"""
Utilidades para manejar la persistencia de la base de datos de rostros.
"""
import os
import pickle
import streamlit as st
import json
import base64
import numpy as np

# Configurar ruta para la base de datos
DATABASE_FILE = "face_database.pkl"

def save_face_database(database):
    """
    Guarda la base de datos de rostros en un archivo persistente.
    
    Args:
        database (dict): La base de datos de rostros a guardar
    """
    try:
        # Convertir numpy arrays a listas para poder serializarlas
        serializable_db = {}
        for name, info in database.items():
            serializable_db[name] = {}
            # Manejar diferentes formatos de la base de datos
            if 'embeddings' in info:
                serializable_db[name]['embeddings'] = [emb.tolist() if isinstance(emb, np.ndarray) else emb for emb in info['embeddings']]
                serializable_db[name]['models'] = info['models']
                serializable_db[name]['count'] = info['count']
            elif 'embedding' in info:
                # Formato antiguo
                serializable_db[name]['embedding'] = info['embedding'].tolist() if isinstance(info['embedding'], np.ndarray) else info['embedding']
                serializable_db[name]['count'] = info.get('count', 1)
        
        # Guardar en un archivo pickle
        with open(DATABASE_FILE, 'wb') as f:
            pickle.dump(serializable_db, f)
        return True
    except Exception as e:
        st.error(f"Error al guardar la base de datos: {str(e)}")
        return False

def load_face_database():
    """
    Carga la base de datos de rostros desde un archivo persistente.
    
    Returns:
        dict: La base de datos de rostros cargada, o un diccionario vacío si no existe el archivo.
    """
    if not os.path.exists(DATABASE_FILE):
        return {}
    
    try:
        with open(DATABASE_FILE, 'rb') as f:
            database = pickle.load(f)
        
        # Convertir listas a numpy arrays
        for name, info in database.items():
            if 'embeddings' in info:
                database[name]['embeddings'] = [np.array(emb) if isinstance(emb, list) else emb for emb in info['embeddings']]
            elif 'embedding' in info:
                database[name]['embedding'] = np.array(info['embedding']) if isinstance(info['embedding'], list) else info['embedding']
        
        return database
    except Exception as e:
        st.error(f"Error al cargar la base de datos: {str(e)}")
        return {}

def export_database_json():
    """
    Exporta la base de datos a un archivo JSON para compartir o hacer backup.
    
    Returns:
        str: Ruta al archivo JSON exportado.
    """
    try:
        if 'face_database' in st.session_state and st.session_state.face_database:
            # Crear una versión serializable de la base de datos
            serializable_db = {}
            for name, info in st.session_state.face_database.items():
                serializable_db[name] = {}
                if 'embeddings' in info:
                    serializable_db[name]['embeddings'] = [
                        base64.b64encode(np.array(emb).tobytes()).decode('utf-8') 
                        for emb in info['embeddings']
                    ]
                    serializable_db[name]['models'] = info['models']
                    serializable_db[name]['count'] = info['count']
                elif 'embedding' in info:
                    serializable_db[name]['embedding'] = base64.b64encode(
                        np.array(info['embedding']).tobytes()
                    ).decode('utf-8')
                    serializable_db[name]['count'] = info.get('count', 1)
            
            # Guardar en un archivo JSON
            export_file = "face_database_export.json"
            with open(export_file, 'w') as f:
                json.dump(serializable_db, f, indent=2)
            
            return export_file
        return None
    except Exception as e:
        st.error(f"Error al exportar la base de datos: {str(e)}")
        return None

def import_database_json(json_file):
    """
    Importa una base de datos desde un archivo JSON.
    
    Args:
        json_file: El archivo JSON a importar
        
    Returns:
        dict: La base de datos importada.
    """
    try:
        content = json_file.read()
        imported_db = json.loads(content)
        
        # Convertir datos codificados en base64 a numpy arrays
        for name, info in imported_db.items():
            if 'embeddings' in info:
                imported_db[name]['embeddings'] = [
                    np.frombuffer(base64.b64decode(emb), dtype=np.float32) 
                    for emb in info['embeddings']
                ]
            elif 'embedding' in info:
                imported_db[name]['embedding'] = np.frombuffer(
                    base64.b64decode(info['embedding']), dtype=np.float32
                )
        
        return imported_db
    except Exception as e:
        st.error(f"Error al importar la base de datos: {str(e)}")
        return {} 