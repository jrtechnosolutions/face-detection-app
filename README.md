# Face Detection & Recognition App

[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.31.0-FF4B4B.svg)](https://streamlit.io/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8.0-5C3EE8.svg)](https://opencv.org/)
[![DeepFace](https://img.shields.io/badge/DeepFace-0.0.79-FFA500.svg)](https://github.com/serengil/deepface)

## 📋 Description

This application is a comprehensive facial analysis tool built with OpenCV, DeepFace, and Streamlit. It provides advanced capabilities for face detection, facial feature recognition, face comparison, and face recognition with identity management.

## ✨ Features

- **Face Detection:** Utilizes OpenCV's DNN module for accurate face detection in images and video streams
- **Feature Detection:** Identifies facial features like eyes and smiles using Haar Cascade classifiers
- **Attribute Analysis:** Estimates age, gender, and emotion using DeepFace's pre-trained models
- **Face Comparison:** Compares faces between images using both HOG features and neural embedding models
- **Face Recognition:** Registers and recognizes facial identities with a persistent database
- **Real-time Processing:** Works with both static images and live camera feeds

## 🛠️ Technologies

- **Streamlit:** Interactive web interface
- **OpenCV:** Computer vision algorithms and video processing
- **DeepFace:** Deep learning models for facial analysis
- **TensorFlow/Keras:** Deep learning framework
- **NumPy/Pandas:** Data processing and manipulation
- **Scikit-learn:** Machine learning utilities

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/face-detection-app.git
cd face-detection-app

# Install dependencies
pip install -r requirements.txt

# Additional system requirements (for Ubuntu/Debian)
apt-get update && apt-get install -y $(cat packages.txt)
```

## 🚀 Usage

```bash
# Run the application
streamlit run app.py
```

### Application Modes

1. **About:** Overview and instructions for using the application
2. **Face Detection:** Detect faces in images or video streams
3. **Feature Detection:** Identify facial features like eyes and smiles
4. **Comparison Mode:** Compare faces between two images
5. **Face Recognition:** Register and recognize faces with identity management

## 📸 Image Processing Capabilities

- Face detection with confidence scoring
- Eye and smile detection with adjustable sensitivity
- Age, gender, and emotion estimation
- Facial comparison with similarity scoring
- Identity recognition against registered faces

## 📊 Data Management

- Register new faces with names
- Update existing facial profiles with multiple samples
- Export and import face databases
- Secure local storage of facial embeddings

## 🔧 Configuration

The application supports various configuration options through the UI:
- Confidence thresholds for detection
- Feature detection sensitivity
- Comparison model selection
- Recognition threshold adjustment

## 💻 Development

This project is built with a modular architecture:
- `app.py`: Main application entry point
- `streamlit_app.py`: Core application with UI components
- `face_comparison.py`: Face comparison algorithms
- `face_database_utils.py`: Database management utilities
- `deepface_patch.py`: Custom patches for DeepFace integration

## 📄 License

[MIT License](LICENSE)

## 🙏 Acknowledgements

- OpenCV team for their computer vision libraries
- Streamlit team for their easy-to-use web framework
- DeepFace project for pre-trained facial analysis models
- All open-source contributors who made this project possible
