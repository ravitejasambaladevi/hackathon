# 🚨 AI Emergency Detection System

## 📌 Overview

AI Emergency Detection System is a real-time computer vision project that detects critical situations such as:

* Unconscious person
* Vehicle accidents
* Fire incidents

The system uses intelligent logic (not just object detection) to reduce false alerts and provide reliable emergency detection.

---

## 🎯 Features

* 🔍 Real-time detection using webcam
* 🧍 Unconscious person detection (based on no movement)
* 🚗 Accident detection (based on motion spikes)
* 🔥 Fire detection (color-based analysis)
* ⚠️ Risk scoring system
* 📸 Automatic image capture during emergencies
* 📊 Live dashboard using Streamlit

---

## ⚙️ Tech Stack

* Python
* YOLOv8 (Ultralytics)
* OpenCV
* Streamlit
* NumPy

---

## ▶️ How to Run

### 1. Install requirements

```
pip install -r requirements.txt
```

### 2. Run the application

```
streamlit run app/dashboard.py
```

---

## 📂 Project Structure

```
ai_emergency_detection/
│
├── app/
│   └── dashboard.py
│
├── outputs/        # saved images
├── requirements.txt
├── yolov8s.pt
```

---

## 💡 Use Cases

* Smart city surveillance
* Night-time road monitoring
* Security systems
* Industrial safety

---

## 🚀 Future Improvements

* Custom trained fire & accident model
* SMS/Email alert system
* Cloud deployment
* Multi-camera support

---

## 👨‍💻 Author

Keshapogu Mohan Krishna 
Raviteja Sambaladevi
