# 🌿 Plant Disease Recognition App

[🚀 Live App](https://plant-disease-recognition-szwjhwyjfkmxfxgtqhjbzc.streamlit.app/) | 

This project uses a trained CNN model to detect plant diseases from uploaded leaf images using TensorFlow and Streamlit.

---

## 🚀 Features

- Built using "TensorFlow" and "Streamlit"
- Classifies into "38 different plant disease classes"
- Provides **treatment suggestions**
- Clean, user-friendly UI with background banner

---

## 📸 Screenshot

![App Screenshot](assests)


## 🧠 Model Details

| Layer (Type)           | Output Shape        | Parameters |
|------------------------|---------------------|------------|
| Conv2D, MaxPooling2D   | 6 Convolution Blocks| ✔️          |
| Dense + Dropout        | 1500 ➝ 38 classes   | ✔️          |
| Model Size             | ~8 Million Params   | ✔️          |
| Accuracy               | ~97.5% (Train), ~94.1% (Validation) | ✔️ |

---

## 📁 Dataset Info

- Total Images: ~87,000
- Classes: 38 Plant Diseases
- Split:
  - Train: 70,295 images
  - Validation: 17,572 images
  - Test: 33 images

📚 Original dataset from: [PlantVillage GitHub Repo](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset)

---

## 🛠️ Tech Stack

- 🐍 Python
- 🧠 TensorFlow
- 🌐 Streamlit

---

## 📁 Project Structure

plant-disease-recognition/
├── main1.py # Streamlit app script
├── trained_plant_disease_model.h5 # Trained CNN model (<100MB)
├── bn1.jpeg # Banner image
├── requirements.txt # Python dependencies
├── .gitignore # Files to ignore in repo
└── README.md # You're reading it!

## ▶️ Run Locally

```bash
https://github.com/Asif-770/plant-disease-recognition.git
cd plant-disease-recognition
pip install -r requirements.txt
streamlit run main1.py

---

## 👨‍💻 Developed By

"Md Asif Khan"  
📫 Email: ak2130251@gmail.com  
