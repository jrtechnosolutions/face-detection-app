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
    # URLs y rutas para los modelos necesarios
    models = [
        {
            "url": "https://raw.githubusercontent.com/sr6033/face-detection-with-OpenCV-and-DNN/master/deploy.prototxt.txt",
            "path": "deploy.prototxt.txt"
        },
        {
            "url": "https://raw.githubusercontent.com/sr6033/face-detection-with-OpenCV-and-DNN/master/res10_300x300_ssd_iter_140000.caffemodel",
            "path": "res10_300x300_ssd_iter_140000.caffemodel"
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