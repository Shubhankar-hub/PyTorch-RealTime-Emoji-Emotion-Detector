<h1 align="center">
  <img src="A_logo_features_the_project_title_PyTorch_Emojify.png" width="320"><br>
  🤖 PyTorch Emojify  
  <br>
  <sub>Real-Time Emotion Detection with Emoji Overlay</sub>
</h1>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/PyTorch-2.x-red?logo=pytorch&logoColor=white" />
  <img src="https://img.shields.io/badge/OpenCV-4.x-green?logo=opencv&logoColor=white" />
  <img src="https://img.shields.io/badge/Status-Active-success?style=flat" />
  <img src="https://img.shields.io/github/license/YourUsername/PyTorch-Emojify" />
  <img src="https://img.shields.io/github/stars/YourUsername/PyTorch-Emojify?style=social" />
</p>

---

## 🎭 Overview

**PyTorch Emojify** is a real-time deep-learning project that:

- Detects **facial emotions** from your webcam  
- Uses a **PyTorch CNN model**  
- Overlays **transparent PNG emojis** on your face (Snapchat-style)  
- Supports **7 emotions**  
- Works on CPU & GPU  
- Includes **GUI** support  

Perfect for learning computer vision, deep learning, PyTorch, or making cool AI projects.

---

## 😄 Supported Emotions

| Emotion    | Emoji |
|-----------|-------|
| Angry     | 😡 |
| Disgusted | 🤢 |
| Fearful   | 😨 |
| Happy     | 😀 |
| Neutral   | 😐 |
| Sad       | 😢 |
| Surprised | 😲 |

---

## 🎥 Demo Preview (GIF Placeholder)

> Replace `demo.gif` when you upload your GIF.

<p align="center">
  <img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/5db240f0-339d-4a82-abed-930853298c11" />
  <img width="1920" height="1080" alt="Screenshot 2025-11-15 203640" src="https://github.com/user-attachments/assets/a5293fd0-b825-4554-9f6c-b6d1f668e782" />

</p>

---

## 📂 Project Structure
```
PyTorch-Emojify/
│── model.py
│── train.py
│── realtime.py
│── emojify.py
│── gui.py
│── emotion.pth
│── README.md
│── requirements.txt
│── assets/
│      angry.png
│      disgusted.png
│      fearful.png
│      happy.png
│      neutral.png
│      sad.png
│      surprised.png
│
└── data

```

Install all dependencies:

```
pip install -r requirements.txt
pip install torch torchvision torchaudio numpy

```
🚀 Run Real-Time Emotion Detection
python realtime.py

😃 Run Emoji Overlay (Emojify Mode!)
python emojifywithname.py

🧠 Model Details

Framework: PyTorch

Architecture: Custom CNN

Input size: 48×48 grayscale

Output: 7 emotion classes

Loss: CrossEntropyLoss

Optimizer: Adam (lr=0.0001)
