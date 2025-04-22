"""
Model Downloader Utility

This module handles the automatic download of pre-trained model files
required for face detection functionality in the application.

It retrieves the necessary model configuration and weights files from
public repositories and saves them to the local file system.
"""
import os
import requests
import shutil

def download_file(url, save_path):
    """
    Downloads a file from a URL and saves it to the specified path.
    
    Args:
        url (str): The URL of the file to download
        save_path (str): The local file path where the file will be saved
        
    Returns:
        bool: True if download was successful, False otherwise
    """
    print(f"Downloading {url} to {save_path}...")
    
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(save_path, 'wb') as f:
            shutil.copyfileobj(response.raw, f)
        print(f"File successfully downloaded: {save_path}")
        return True
    else:
        print(f"Error downloading {url}: {response.status_code}")
        return False

def main():
    """
    Main function that downloads all required model files.
    
    Specifies the model files needed for face detection and downloads
    them if they don't already exist in the local directory.
    """
    # URLs and paths for required model files
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

    # Download each model file if it doesn't exist
    for model in models:
        if not os.path.exists(model["path"]):
            download_file(model["url"], model["path"])
        else:
            print(f"File already exists: {model['path']}")

if __name__ == "__main__":
    main() 