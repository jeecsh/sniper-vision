# 🎯 Sniper Vision

![Demo GIF](media/demo.mp4)

Sniper Vision is a Raspberry Pi-based AI system that uses real-time **face recognition**, **optical flow**, and **Kalman filtering** to detect and track a human forehead. This project is a prototype aimed at educational and research purposes.

> 🚨 **Disclaimer**: This project is strictly for research and academic purposes. No real weapons or harm are involved.

---


## 🧠 Technologies Used

- **Raspberry Pi** (for live camera feed)
- **Face Recognition** (via `face_recognition` Python library)
- **Optical Flow** (Lucas-Kanade method)
- **Kalman Filter** (for predictive tracking)
- **OpenCV** (for image processing & drawing)
- **Socket Programming** (for frame transmission)

---

## ⚙️ How It Works

1. **Capture Frames:** Raspberry Pi captures real-time frames from the camera.
2. **Face Detection:** The system detects faces, focusing on the forehead area.
3. **Optical Flow:** Tracks movement from frame to frame using optical flow techniques.
4. **Kalman Filtering:** Smoothens the tracking of the forehead, reducing jitter and improving precision.
5. **Coordinate Calculation:** The system uses detected coordinates for simulated targeting.

---

## 📦 Installation

To get started with Sniper Vision, clone the repository and install the necessary dependencies:

```bash
git clone https://github.com/jeecsh/sniper-vision.git
cd sniper-vision
pip install -r requirements.txt
