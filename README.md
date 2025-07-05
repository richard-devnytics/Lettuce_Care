# 🥬 Lettuce Care – CNN-Based Lettuce Disease Detection App

**Lettuce Care** is a machine learning application that uses Convolutional Neural Networks (CNNs) to detect common diseases in lettuce from images. It represents a significant advancement in agricultural technology by enabling early, automated, and reliable detection of plant diseases through image classification.

---

## 🧠 About the Project

CNNs are deep learning algorithms especially suited for image recognition and classification. In this app, a CNN model is trained on a dataset of lettuce leaf images — including both healthy and diseased samples — to detect patterns such as discoloration, leaf spots, and deformities.

Once deployed, the model can classify new lettuce images and provide both the disease label and basic care instructions, helping reduce crop losses and improve farm management.

---

## 🌿 Why It Matters

- 🧪 **Accurate Early Detection:** Detect diseases before they spread.
- 👨‍🌾 **Reduce Manual Labor:** Automates visual inspection.
- 📱 **Mobile-Friendly:** Ideal for deployment in Android apps or handheld devices.
- ⚡ **Real-Time Diagnosis:** Fast prediction from camera or uploaded image.

---

## 🛠️ Tech Stack

- `Python`
- `Keras` + `TensorFlow` – CNN model training and inference
- `KivyMD` – UI interface (Material Design)
- `OpenCV` or `Pillow` – for image preprocessing (if used)

---

## 🗂️ Project Structure

lettuce-care/
├── model/ # Trained Keras CNN model (.h5)
├── app/ # Python + KivyMD application
│ ├── main.py
│ └── ui.kv
├── data/ # Sample input images
├── requirements.txt
├── README.md
└── .gitignore


---

## 🧪 Model Overview

- **Architecture:** CNN with Conv2D, MaxPooling, Flatten, Dense layers
- **Input Size:** 224x224 or 128x128 (depending on training)
- **Dataset:** Labeled lettuce leaves (healthy and diseased)

---

## 🥬 Disease Classes

This version of the model can detect the following classes:

- ✅ Healthy  
- ⚠️ Septoria Leaf Spot  
- ⚠️ Downy Mildew

*Additional classes can be added as more data becomes available.*

---

## ▶️ How to Run

### 🖥️ Local Desktop (Test Mode)

1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Or venv\Scripts\activate on Windows
