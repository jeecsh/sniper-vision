# 🎯 Sniper Vision

<video width="100%" controls>
  <source src="media/demo.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

Sniper Vision is a Raspberry Pi-based AI system that uses real-time **head detection**, **optical flow**, and **Kalman filtering** to detect and track a human forehead. This project is a prototype aimed at educational and research purposes.

> 🚨 **Disclaimer**: This project is strictly for research and academic purposes. No real weapons or harm are involved.

---

## 🧠 Technologies Used

- **Raspberry Pi** (for live camera feed)
- **YOLOv4-tiny** (for head detection)
- **Optical Flow** (Lucas-Kanade method with RANSAC)
- **Kalman Filter** (for predictive tracking)
- **OpenCV** (for image processing & drawing)
- **Socket Programming** (for frame transmission)

---

## ⚙️ How It Works

1. **Capture Frames:** Raspberry Pi captures real-time frames from the camera.
2. **Head Detection:** System detects heads using YOLOv4-tiny model, focusing on the forehead area.
3. **Optical Flow:** Tracks movement from frame to frame using Lucas-Kanade optical flow with RANSAC for better accuracy.
4. **Kalman Filtering:** Smoothens the tracking of the forehead, reducing jitter and improving precision.
5. **Coordinate Calculation:** The system uses detected coordinates for simulated targeting with sniper scope overlay.

---

## 📦 Installation

To get started with Sniper Vision, clone the repository and install the necessary dependencies:

```bash
git clone https://github.com/jeecsh/sniper-vision.git
cd sniper-vision
pip install -r requirements.txt
