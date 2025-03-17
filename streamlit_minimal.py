import streamlit as st

def main():
    st.title("Face Detection App")
    st.write("Esta es una versión mínima para probar la configuración.")
    st.write("Si ves este mensaje, la aplicación está configurada correctamente.")
    
    st.info("Versiones de dependencias:")
    try:
        import tensorflow as tf
        st.success(f"TensorFlow version: {tf.__version__}")
    except Exception as e:
        st.error(f"Error de TensorFlow: {e}")
    
    try:
        import keras
        st.success(f"Keras version: {keras.__version__}")
    except Exception as e:
        st.error(f"Error de Keras: {e}")
    
    try:
        import cv2
        st.success(f"OpenCV version: {cv2.__version__}")
    except Exception as e:
        st.error(f"Error de OpenCV: {e}")
    
if __name__ == "__main__":
    main() 