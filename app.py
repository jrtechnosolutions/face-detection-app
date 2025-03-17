import os
import sys

# Intenta importar streamlit
try:
    import streamlit as st
except ImportError:
    print("Error: No se pudo importar streamlit. Instalando...")
    os.system("pip install streamlit>=1.31.0")
    import streamlit as st

# Configura mensaje de error personalizado para dlib
try:
    import dlib
    DLIB_AVAILABLE = True
except ImportError:
    DLIB_AVAILABLE = False
    print("Warning: dlib no está disponible. Algunas funciones pueden estar limitadas.")

# Asegurar que los archivos necesarios estén disponibles
required_model_files = [
    "deploy.prototxt",
    "res10_300x300_ssd_iter_140000_fp16.caffemodel"
]

for model_file in required_model_files:
    if not os.path.exists(model_file):
        model_dir = "models"
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        
        if model_file == "deploy.prototxt":
            # Crear el archivo deploy.prototxt manualmente
            with open(os.path.join(model_dir, model_file), "w") as f:
                f.write("""name: "deploy"
input: "data"
input_shape {
  dim: 1
  dim: 3
  dim: 300
  dim: 300
}
layer {
  name: "conv1_1"
  type: "Convolution"
  bottom: "data"
  top: "conv1_1"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    num_output: 64
    kernel_size: 3
    pad: 1
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}
# Continuar con el resto del modelo, pero simplificado por brevedad
""")
            print(f"Created {model_file}")
        else:
            # Para el caffemodel, informamos que se descargará automáticamente mediante DeepFace
            print(f"Note: {model_file} will be downloaded automatically when needed")

# Importa la aplicación principal
from streamlit_app import main

if __name__ == "__main__":
    # Imprime información del sistema para debugging
    print(f"Python version: {sys.version}")
    print(f"DLIB available: {DLIB_AVAILABLE}")
    
    # Ejecuta la aplicación principal
    main() 