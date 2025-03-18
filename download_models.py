import os
import requests
import shutil

def download_file(url, save_path):
    """Descarga un archivo desde una URL y lo guarda en la ruta especificada."""
    print(f"Descargando {url} a {save_path}...")
    
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(save_path, 'wb') as f:
            shutil.copyfileobj(response.raw, f)
        print(f"Archivo descargado exitosamente: {save_path}")
        return True
    else:
        print(f"Error al descargar {url}: {response.status_code}")
        return False

def main():
    # Crear directorio para modelos si no existe
    models_dir = "models"
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
        print(f"Directorio creado: {models_dir}")

    # URLs y rutas para los modelos necesarios
    models = [
        {
            "url": "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt",
            "path": os.path.join(models_dir, "deploy.prototxt")
        },
        {
            "url": "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel",
            "path": os.path.join(models_dir, "res10_300x300_ssd_iter_140000.caffemodel")
        }
    ]

    # Descargar cada modelo
    for model in models:
        if not os.path.exists(model["path"]):
            download_file(model["url"], model["path"])
        else:
            print(f"El archivo ya existe: {model['path']}")

if __name__ == "__main__":
    main() 