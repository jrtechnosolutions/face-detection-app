# Simple app.py for Face Detection
import streamlit as st

try:
    # Importar la aplicación principal
    print("Starting Face Detection Application...")
    from streamlit_app import main
    
    # Main entry point
    if __name__ == "__main__":
        main()
except Exception as e:
    st.error(f"Error al iniciar la aplicación: {e}")
    st.error("Por favor, revise los logs para más información.") 