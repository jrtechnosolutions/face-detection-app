"""
Minimal Streamlit Application for System Testing

This module provides a simplified version of the face detection application
for testing environment configuration and dependency installation.

It displays basic information about installed dependencies and their versions,
helping to diagnose potential compatibility issues before running the full app.
"""
import streamlit as st

def main():
    """
    Main function that creates a minimal Streamlit interface.
    
    Displays a simple UI with version information for key dependencies,
    allowing users to verify that the environment is correctly configured.
    """
    # Set up the page header
    st.title("Face Detection App")
    st.write("This is a minimal version to test the configuration.")
    st.write("If you can see this message, the application is configured correctly.")
    
    # Display dependency versions
    st.info("Dependency Versions:")
    
    # Check TensorFlow installation
    try:
        import tensorflow as tf
        st.success(f"TensorFlow version: {tf.__version__}")
    except Exception as e:
        st.error(f"TensorFlow error: {e}")
    
    # Check Keras installation
    try:
        import keras
        st.success(f"Keras version: {keras.__version__}")
    except Exception as e:
        st.error(f"Keras error: {e}")
    
    # Check OpenCV installation
    try:
        import cv2
        st.success(f"OpenCV version: {cv2.__version__}")
    except Exception as e:
        st.error(f"OpenCV error: {e}")
    
if __name__ == "__main__":
    main() 