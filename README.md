# Multi-Digit Image Detection using RNN

This project focuses on detecting and recognizing multi-digit numbers from vehicle plate images using deep learning techniques. The model utilizes **Recurrent Neural Networks (RNN)** to learn digit sequences from image data, based on the **CAR-A** and **CAR-B** datasets.

## 🚀 Features
- Processes vehicle plate images and extracts digit labels.  
- Uses TensorFlow/Keras for model training.  
- Supports both training and testing datasets.  
- Designed for multi-digit number recognition tasks.  

## 🧠 Tech Stack
- Python 3.x  
- TensorFlow / Keras  
- NumPy, Pandas, Matplotlib, PIL  

## 📁 Dataset Structure

ORAND-CAR-2014/
├── CAR-A/
│   ├── a_train_images/
│   ├── a_test_images/
│   ├── a_train_gt.txt
│   └── a_test_gt.txt
└── CAR-B/
├── b_train_images/
├── b_test_images/
├── b_train_gt.txt
└── b_test_gt.txt


## ⚙️ How to Run

pip install -r requirements.txt
jupyter notebook checked.ipynb

## 📊 Results

The model learns to recognize multi-digit sequences with high accuracy, demonstrating the effectiveness of RNNs in sequential image-to-text tasks.

## 📄 License

This project is open-source under the [MIT License](LICENSE).
