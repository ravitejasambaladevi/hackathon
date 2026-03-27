import streamlit as st
import cv2
import math
import time
import os
import numpy as np
from ultralytics import YOLO

st.set_page_config(page_title="AI Emergency Detection", layout="wide")
st.title("🚨 AI Emergency Detection System")

# Load model
model = YOLO("yolov8s.pt")

IMPORTANT_CLASSES = ["person", "car", "motorcycle", "bus", "truck"]

# ---------------- SESSION ----------------
if "run" not in st.session_state:
    st.session_state.run = False

if "saved_count" not in st.session_state:
    st.session_state.saved_count = 0

if "last_saved_time" not in st.session_state:
    st.session_state.last_saved_time = 0

# ---------------- UI ----------------
col1, col2 = st.columns(2)

with col1:
    if st.button("▶ Start"):
        st.session_state.run = True

with col2:
    if st.button("⏹ Stop"):
        st.session_state.run = False

video_placeholder = st.empty()
status_placeholder = st.empty()

# ---------------- PARAMETERS ----------------
person_history = []
movement_buffer = []

MAX_HISTORY = 20
BUFFER_SIZE = 10

MOVEMENT_THRESHOLD = 25
STILL_TIME_THRESHOLD = 20

still_start_time = None

# Movement smoothing (FIX)
movement_start_time = None
RESET_TIME_THRESHOLD = 2

# Fire
fire_start_time = None
FIRE_TIME_THRESHOLD = 1

# Accident
prev_gray = None
accident_flag = False
accident_time = None

# Save
SAVE_COOLDOWN = 10
MAX_SAVED_IMAGES = 2

os.makedirs("outputs", exist_ok=True)

# ---------------- FIRE DETECTION ----------------
def detect_fire(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_fire = (0, 120, 200)
    upper_fire = (35, 255, 255)
    mask_fire = cv2.inRange(hsv, lower_fire, upper_fire)

    lower_white = (0, 0, 230)
    upper_white = (180, 40, 255)
    mask_white = cv2.inRange(hsv, lower_white, upper_white)

    combined = cv2.bitwise_or(mask_fire, mask_white)

    fire_pixels = cv2.countNonZero(combined)
    total_pixels = frame.shape[0] * frame.shape[1]

    return fire_pixels / total_pixels


# ---------------- MAIN LOOP ----------------
if st.session_state.run:

    cap = cv2.VideoCapture(0)

    for _ in range(1000):

        ret, frame = cap.read()
        if not ret:
            st.error("Camera not working")
            break

        frame = cv2.resize(frame, (640, 480))

        results = model(frame, conf=0.3)

        person_detected = False
        vehicle_detected = False
        current_centers = []
        risk_score = 0

        # -------- YOLO DETECTION --------
        for box in results[0].boxes:
            class_id = int(box.cls[0])
            class_name = model.names[class_id]
            conf = float(box.conf[0])

            if class_name in IMPORTANT_CLASSES:
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{class_name} {conf:.2f}",
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                if class_name == "person":
                    person_detected = True
                    cx = (x1 + x2) // 2
                    cy = (y1 + y2) // 2
                    current_centers.append((cx, cy))

                if class_name in ["car", "motorcycle", "bus", "truck"]:
                    vehicle_detected = True

        # -------- UNCONSCIOUS DETECTION --------
        if current_centers:
            person_history.append(current_centers[0])

        if len(person_history) > MAX_HISTORY:
            person_history.pop(0)

        if len(person_history) >= 2:
            px, py = person_history[-2]
            cx, cy = person_history[-1]

            dist = math.sqrt((cx - px) ** 2 + (cy - py) ** 2)

            movement_buffer.append(dist)
            if len(movement_buffer) > BUFFER_SIZE:
                movement_buffer.pop(0)

            avg_movement = np.median(movement_buffer)

            if person_detected:
                risk_score += 20

            if avg_movement < MOVEMENT_THRESHOLD:
                risk_score += 30

                if still_start_time is None:
                    still_start_time = time.time()

                elapsed = time.time() - still_start_time

                cv2.putText(frame, f"Still: {int(elapsed)}s",
                            (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

                if elapsed > STILL_TIME_THRESHOLD:
                    cv2.putText(frame, "UNCONSCIOUS",
                                (20, 80),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                    risk_score += 50

            elif avg_movement > MOVEMENT_THRESHOLD:
                if movement_start_time is None:
                    movement_start_time = time.time()

                elif time.time() - movement_start_time > RESET_TIME_THRESHOLD:
                    still_start_time = None
                    movement_start_time = None
            else:
                movement_start_time = None

        # -------- ACCIDENT DETECTION --------
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        motion_score = 0

        if prev_gray is not None:
            diff = cv2.absdiff(prev_gray, gray)
            thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)[1]
            motion_score = cv2.countNonZero(thresh)

        prev_gray = gray

        if motion_score > 20000 and vehicle_detected:
            if not accident_flag:
                accident_flag = True
                accident_time = time.time()

        if accident_flag:
            if motion_score < 10000:
                if time.time() - accident_time > 2:
                    cv2.putText(frame, "🚗 ACCIDENT",
                                (20, 120),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                    risk_score += 50
            else:
                accident_flag = False

        # -------- FIRE DETECTION --------
        fire_ratio = detect_fire(frame)

        cv2.putText(frame, f"Fire: {fire_ratio:.3f}",
                    (20, 280),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        if fire_ratio > 0.1:
            if fire_start_time is None:
                fire_start_time = time.time()

            if time.time() - fire_start_time > FIRE_TIME_THRESHOLD:
                cv2.putText(frame, "🔥 FIRE ALERT",
                            (20, 160),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                risk_score += 80
        else:
            fire_start_time = None

        # -------- DEBUG --------
        cv2.putText(frame, f"Motion: {motion_score}",
                    (20, 320),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        # -------- RISK --------
        cv2.putText(frame, f"Risk: {risk_score}",
                    (20, 200),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        # -------- SAVE IMAGE --------
        if risk_score >= 80:
            current_time = time.time()

            if (
                current_time - st.session_state.last_saved_time > SAVE_COOLDOWN
                and st.session_state.saved_count < MAX_SAVED_IMAGES
            ):
                filename = f"outputs/emergency_{int(current_time)}.jpg"

                if cv2.imwrite(filename, frame):
                    st.session_state.saved_count += 1
                    st.session_state.last_saved_time = current_time

        # -------- DISPLAY --------
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        video_placeholder.image(frame_rgb, channels="RGB")

        status_placeholder.markdown(f"""
        ### Status
        - Risk Score: **{risk_score}**
        - Saved Images: **{st.session_state.saved_count}**
        """)

    cap.release()