import os
import sys

# Intenta importar streamlit
try:
    import streamlit as st
except ImportError:
    print("Error: No se pudo importar streamlit. Instalando...")
    os.system("pip install streamlit>=1.31.0")
    import streamlit as st

# Configura variables de entorno para TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reducir mensajes de TensorFlow
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Forzar CPU para evitar problemas con GPU en entornos cloud

# Configura mensaje de error personalizado para dlib
try:
    import dlib
    DLIB_AVAILABLE = True
except ImportError:
    DLIB_AVAILABLE = False
    print("Warning: dlib no está disponible. Algunas funciones pueden estar limitadas.")

# Verificar TensorFlow y configurar ambiente
try:
    import tensorflow as tf
    tf_version = tf.__version__
    print(f"TensorFlow version: {tf_version}")
except Exception as e:
    print(f"Warning: TensorFlow initialization error: {e}")

# Aplicar parches para DeepFace y RetinaFace
try:
    # Intenta importar y aplicar parches
    import deepface_patch
    deepface_patch.apply_patches()
except Exception as e:
    print(f"Warning: Failed to apply patches: {e}")

# Asegurar que los archivos necesarios estén disponibles
required_model_files = [
    "deploy.prototxt",
    "res10_300x300_ssd_iter_140000.caffemodel",
    "shape_predictor_68_face_landmarks.dat"
]

for model_file in required_model_files:
    if not os.path.exists(model_file):
        print(f"Note: {model_file} will be downloaded automatically when needed")

# Importa la aplicación principal
from streamlit_app import main

if __name__ == "__main__":
    # Imprime información del sistema para debugging
    print(f"Python version: {sys.version}")
    print(f"DLIB available: {DLIB_AVAILABLE}")
    
    # Ejecuta la aplicación principal
    main() 