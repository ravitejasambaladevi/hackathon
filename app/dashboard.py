import streamlit as st
import cv2
import math
import time
import os
import numpy as np
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import av
from queue import Queue
from streamlit_autorefresh import st_autorefresh

# ---------------- AUTO REFRESH ----------------
st_autorefresh(interval=1000, key="refresh")

# ---------------- PAGE ----------------
st.set_page_config(page_title="AI Emergency Detection", layout="wide")

st.title("🚨 AI Emergency Detection System")
st.info("Allow camera access")

# ---------------- MODEL ----------------
@st.cache_resource
def load_model():
    return YOLO("yolov8s")  # Change to your model path if custom

model = load_model()

IMPORTANT_CLASSES = ["person", "car", "motorcycle", "bus", "truck"]

# ---------------- OUTPUT ----------------
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------- QUEUE ----------------
if "save_queue" not in st.session_state:
    st.session_state.save_queue = Queue()

save_queue = st.session_state.save_queue

# ---------------- SESSION ----------------
if "saved_images" not in st.session_state:
    st.session_state.saved_images = []

# ---------------- RTC ----------------
RTC_CONFIGURATION = RTCConfiguration({
    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
})

# ---------------- FIRE ----------------
def detect_fire(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv, (0,120,200), (35,255,255))
    mask2 = cv2.inRange(hsv, (0,0,230), (180,40,255))
    combined = cv2.bitwise_or(mask1, mask2)
    return cv2.countNonZero(combined) / (frame.shape[0]*frame.shape[1])

# ---------------- PROCESSOR ----------------
class VideoProcessor(VideoProcessorBase):

    def __init__(self):
        self.person_history = []
        self.movement_buffer = []
        self.prev_gray = None
        self.still_start_time = None
        self.movement_start_time = None
        self.accident_flag = False
        self.accident_time = None
        self.fire_start_time = None
        self.last_frame_time = 0
        self.last_saved_time = 0

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        frame_resized = cv2.resize(img, (640, 360))

        # FPS control
        if time.time() - self.last_frame_time < 0.05:
            return av.VideoFrame.from_ndarray(frame_resized, format="bgr24")
        self.last_frame_time = time.time()

        results = model(frame_resized, conf=0.3)

        current_centers = []
        vehicle_detected = False
        risk_score = 0

        # YOLO Detection
        for box in results[0].boxes:
            cls = model.names[int(box.cls[0])]
            if cls in IMPORTANT_CLASSES:
                x1,y1,x2,y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame_resized,(x1,y1),(x2,y2),(0,255,0),2)

                if cls == "person":
                    cx = (x1+x2)//2
                    cy = (y1+y2)//2
                    current_centers.append((cx,cy))

                if cls in ["car","bus","truck","motorcycle"]:
                    vehicle_detected = True

        # UNCONSCIOUS
        if current_centers:
            self.person_history.append(current_centers[0])

        if len(self.person_history) > 20:
            self.person_history.pop(0)

        if len(self.person_history) >= 2:
            px,py = self.person_history[-2]
            cx,cy = self.person_history[-1]

            dist = math.sqrt((cx-px)**2 + (cy-py)**2)
            self.movement_buffer.append(dist)

            if len(self.movement_buffer) > 10:
                self.movement_buffer.pop(0)

            avg = np.median(self.movement_buffer)

            if avg < 35:
                if self.still_start_time is None:
                    self.still_start_time = time.time()

                elapsed = time.time() - self.still_start_time

                cv2.putText(frame_resized,f"Still: {int(elapsed)}s",(20,40),
                            cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,255),2)

                if elapsed > 15:
                    cv2.putText(frame_resized,"UNCONSCIOUS",(20,80),
                                cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),3)
                    risk_score += 70

            elif avg > 40:
                if self.movement_start_time is None:
                    self.movement_start_time = time.time()
                elif time.time()-self.movement_start_time > 2:
                    self.still_start_time = None
                    self.movement_start_time = None

        # ACCIDENT
        gray = cv2.cvtColor(frame_resized,cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray,(21,21),0)

        motion_score = 0
        if self.prev_gray is not None:
            diff = cv2.absdiff(self.prev_gray,gray)
            thresh = cv2.threshold(diff,25,255,cv2.THRESH_BINARY)[1]
            motion_score = cv2.countNonZero(thresh)

        self.prev_gray = gray

        if motion_score > 12000 and vehicle_detected:
            if not self.accident_flag:
                self.accident_flag = True
                self.accident_time = time.time()

        if self.accident_flag:
            if motion_score < 15000:
                if time.time()-self.accident_time > 2:
                    cv2.putText(frame_resized,"ACCIDENT",(20,120),
                                cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),3)
                    risk_score += 50
            else:
                self.accident_flag = False

        # FIRE
        fire_ratio = detect_fire(frame_resized)

        cv2.putText(frame_resized,f"Fire:{fire_ratio:.2f}",(20,260),
                    cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,255),2)

        if fire_ratio > 0.1:
            if self.fire_start_time is None:
                self.fire_start_time = time.time()

            if time.time()-self.fire_start_time > 1:
                cv2.putText(frame_resized,"FIRE ALERT",(20,160),
                            cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),3)
                risk_score += 100
        else:
            self.fire_start_time = None

        # SAVE
        if risk_score >= 50:
            if time.time() - self.last_saved_time > 10:
                filename = f"{OUTPUT_DIR}/emergency_{int(time.time())}.jpg"
                if cv2.imwrite(filename, frame_resized):
                    save_queue.put(filename)
                    st.session_state.saved_images.append(filename)
                    self.last_saved_time = time.time()

        # DEBUG
        cv2.putText(frame_resized,f"Motion:{motion_score}",(20,300),
                    cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,0,0),2)
        cv2.putText(frame_resized,f"Risk:{risk_score}",(20,330),
                    cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2)

        return av.VideoFrame.from_ndarray(frame_resized,format="bgr24")

# ---------------- UI ----------------
col1,col2 = st.columns([2,1])

with col1:
    webrtc_streamer(
        key="cam",
        video_processor_factory=VideoProcessor,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={"video":True,"audio":False}
    )

with col2:
    count = len(st.session_state.saved_images)

    st.subheader("Status")
    st.write(f"Saved Images: {count}")

    if count > 0:
        st.error("⚠️ Emergency Detected")

    st.subheader("Captured Images")
    for img in st.session_state.saved_images:
        # Convert BGR to RGB for Streamlit display
        st.image(cv2.cvtColor(cv2.imread(img), cv2.COLOR_BGR2RGB), width=200)
