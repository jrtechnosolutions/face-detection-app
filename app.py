# Simple app.py for Face Detection
import os
import streamlit as st

# Comprobar y descargar modelos necesarios
try:
    # Verificar si los archivos de modelo existen
    model_files = [
        "deploy.prototxt.txt",
        "res10_300x300_ssd_iter_140000.caffemodel"
    ]
    
    missing_files = [f for f in model_files if not os.path.exists(f)]
    
    if missing_files:
        st.warning("Faltan archivos de modelo. Descargando...")
        import download_models
        download_models.main()
        st.success("¡Archivos de modelo descargados correctamente!")
except Exception as e:
    st.error(f"Error al comprobar/descargar modelos: {e}")

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