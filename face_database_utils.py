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
        # Verificar si hay datos para guardar
        if not database:
            # Si la base de datos está vacía, eliminar el archivo si existe
            if os.path.exists(DATABASE_FILE):
                os.remove(DATABASE_FILE)
                st.sidebar.write("Database was empty - removed existing file")
            return True
        
        # Convertir numpy arrays a listas para poder serializarlas
        serializable_db = {}
        for name, info in database.items():
            serializable_db[name] = {}
            # Manejar diferentes formatos de la base de datos
            if 'embeddings' in info:
                serializable_db[name]['embeddings'] = [emb.tolist() if isinstance(emb, np.ndarray) else emb for emb in info['embeddings']]
                serializable_db[name]['models'] = info['models']
                serializable_db[name]['count'] = info['count']
                
                # Guardar imagen facial si existe
                if 'face_image' in info:
                    serializable_db[name]['face_image'] = info['face_image'].tolist() if isinstance(info['face_image'], np.ndarray) else info['face_image']
            elif 'embedding' in info:
                # Formato antiguo
                serializable_db[name]['embedding'] = info['embedding'].tolist() if isinstance(info['embedding'], np.ndarray) else info['embedding']
                serializable_db[name]['count'] = info.get('count', 1)
                
                # Guardar imagen facial si existe
                if 'face_image' in info:
                    serializable_db[name]['face_image'] = info['face_image'].tolist() if isinstance(info['face_image'], np.ndarray) else info['face_image']
        
        # Guardar en un archivo pickle
        with open(DATABASE_FILE, 'wb') as f:
            pickle.dump(serializable_db, f)
            
        # Verificar que el archivo se creó correctamente
        if os.path.exists(DATABASE_FILE):
            st.sidebar.write(f"Database saved to {DATABASE_FILE} ({len(serializable_db)} entries)")
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
                # Cargar imagen facial si existe
                if 'face_image' in info:
                    database[name]['face_image'] = np.array(info['face_image']) if isinstance(info['face_image'], list) else info['face_image']
            elif 'embedding' in info:
                database[name]['embedding'] = np.array(info['embedding']) if isinstance(info['embedding'], list) else info['embedding']
                # Cargar imagen facial si existe
                if 'face_image' in info:
                    database[name]['face_image'] = np.array(info['face_image']) if isinstance(info['face_image'], list) else info['face_image']
        
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
                    
                    # Incluir imagen facial si existe
                    if 'face_image' in info:
                        serializable_db[name]['face_image'] = base64.b64encode(
                            np.array(info['face_image']).tobytes()
                        ).decode('utf-8')
                        serializable_db[name]['face_image_shape'] = info['face_image'].shape
                elif 'embedding' in info:
                    serializable_db[name]['embedding'] = base64.b64encode(
                        np.array(info['embedding']).tobytes()
                    ).decode('utf-8')
                    serializable_db[name]['count'] = info.get('count', 1)
                    
                    # Incluir imagen facial si existe
                    if 'face_image' in info:
                        serializable_db[name]['face_image'] = base64.b64encode(
                            np.array(info['face_image']).tobytes()
                        ).decode('utf-8')
                        serializable_db[name]['face_image_shape'] = info['face_image'].shape
            
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
                
                # Importar imagen facial si existe
                if 'face_image' in info and 'face_image_shape' in info:
                    face_data = np.frombuffer(base64.b64decode(info['face_image']), dtype=np.uint8)
                    shape = info['face_image_shape']
                    imported_db[name]['face_image'] = face_data.reshape(shape)
            elif 'embedding' in info:
                imported_db[name]['embedding'] = np.frombuffer(
                    base64.b64decode(info['embedding']), dtype=np.float32
                )
                
                # Importar imagen facial si existe
                if 'face_image' in info and 'face_image_shape' in info:
                    face_data = np.frombuffer(base64.b64decode(info['face_image']), dtype=np.uint8)
                    shape = info['face_image_shape']
                    imported_db[name]['face_image'] = face_data.reshape(shape)
        
        return imported_db
    except Exception as e:
        st.error(f"Error al importar la base de datos: {str(e)}")
        return {}

def print_database_info():
    """
    Imprime información sobre la base de datos actual para depuración.
    """
    if 'face_database' in st.session_state:
        db = st.session_state.face_database
        st.sidebar.write("--- Database Debug Info ---")
        st.sidebar.write(f"Database contains {len(db)} entries")
        
        # Mostrar nombres en la base de datos
        if db:
            names = list(db.keys())
            st.sidebar.write(f"Names in database: {', '.join(names)}")
            
            # Mostrar detalles del primer elemento
            if names:
                first_entry = db[names[0]]
                st.sidebar.write(f"Sample entry for '{names[0]}':")
                if 'embeddings' in first_entry:
                    st.sidebar.write(f"- Has {len(first_entry['embeddings'])} embeddings")
                    st.sidebar.write(f"- Models: {', '.join(first_entry['models'])}")
                    st.sidebar.write(f"- Count: {first_entry['count']}")
                    
                    # Mostrar si tiene imagen
                    if 'face_image' in first_entry:
                        st.sidebar.write(f"- Has reference face image: {first_entry['face_image'].shape}")
                    else:
                        st.sidebar.write("- No reference image")
                elif 'embedding' in first_entry:
                    st.sidebar.write("- Has single embedding (old format)")
                    
                    # Mostrar si tiene imagen
                    if 'face_image' in first_entry:
                        st.sidebar.write(f"- Has reference face image: {first_entry['face_image'].shape}")
                    else:
                        st.sidebar.write("- No reference image")
        else:
            st.sidebar.write("Database is empty") 