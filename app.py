"""
Main application entry point for the Face Detection & Recognition App.

This module initializes the application, checks for required model files, 
and launches the main Streamlit interface.

Author: CVJR01
Version: 1.0.0
License: MIT
"""
import os
import streamlit as st

# Check for required model files and download them if missing
try:
    # Define the required model files for face detection
    model_files = [
        "deploy.prototxt.txt",                    # Caffe model configuration
        "res10_300x300_ssd_iter_140000.caffemodel"  # Pre-trained model weights
    ]
    
    # Identify any missing model files
    missing_files = [f for f in model_files if not os.path.exists(f)]
    
    # If any files are missing, download them automatically
    if missing_files:
        st.warning("Missing required model files. Downloading...")
        import download_models
        download_models.main()
        st.success("Model files successfully downloaded!")
except Exception as e:
    st.error(f"Error checking/downloading models: {e}")

# Launch the main application
try:
    # Import the main application function
    print("Starting Face Detection Application...")
    from streamlit_app import main
    
    # Application entry point
    if __name__ == "__main__":
        main()
except Exception as e:
    st.error(f"Error starting application: {e}")
    st.error("Please check the logs for more information.") 