"""
Face Database Utilities

This module provides functionality for managing a persistent database of facial
embeddings and identities for facial recognition capabilities.

It includes functions for saving, loading, importing, and exporting face data,
with support for multiple embedding models and reference face images.
"""
import os
import pickle
import streamlit as st
import json
import base64
import numpy as np

# Configure path for the database file
DATABASE_FILE = "face_database.pkl"

def save_face_database(database):
    """
    Save the face database to a persistent file.
    
    Handles serialization of facial embeddings, converting numpy arrays to a
    format suitable for pickle storage. Creates a new file or updates an existing
    one, with proper handling of empty databases.
    
    Args:
        database (dict): The face database to save
        
    Returns:
        bool: True if the save operation was successful, False otherwise
    """
    try:
        # Check if there is data to save
        if not database:
            # If database is empty, delete the file if it exists
            if os.path.exists(DATABASE_FILE):
                os.remove(DATABASE_FILE)
                st.sidebar.write("Database was empty - removed existing file")
            return True
        
        # Convert numpy arrays to lists for serialization
        serializable_db = {}
        for name, info in database.items():
            serializable_db[name] = {}
            # Handle different database formats
            if 'embeddings' in info:
                serializable_db[name]['embeddings'] = [emb.tolist() if isinstance(emb, np.ndarray) else emb for emb in info['embeddings']]
                serializable_db[name]['models'] = info['models']
                serializable_db[name]['count'] = info['count']
                
                # Save facial image if it exists
                if 'face_image' in info:
                    serializable_db[name]['face_image'] = info['face_image'].tolist() if isinstance(info['face_image'], np.ndarray) else info['face_image']
            elif 'embedding' in info:
                # Legacy format
                serializable_db[name]['embedding'] = info['embedding'].tolist() if isinstance(info['embedding'], np.ndarray) else info['embedding']
                serializable_db[name]['count'] = info.get('count', 1)
                
                # Save facial image if it exists
                if 'face_image' in info:
                    serializable_db[name]['face_image'] = info['face_image'].tolist() if isinstance(info['face_image'], np.ndarray) else info['face_image']
        
        # Save to a pickle file
        with open(DATABASE_FILE, 'wb') as f:
            pickle.dump(serializable_db, f)
            
        # Verify that the file was created successfully
        if os.path.exists(DATABASE_FILE):
            st.sidebar.write(f"Database saved to {DATABASE_FILE} ({len(serializable_db)} entries)")
        return True
    except Exception as e:
        st.error(f"Error saving database: {str(e)}")
        return False

def load_face_database():
    """
    Load the face database from a persistent file.
    
    Handles deserialization of facial embeddings, converting lists back to
    numpy arrays for efficient processing.
    
    Returns:
        dict: The loaded face database, or an empty dictionary if the file
              doesn't exist or an error occurs
    """
    if not os.path.exists(DATABASE_FILE):
        return {}
    
    try:
        with open(DATABASE_FILE, 'rb') as f:
            database = pickle.load(f)
        
        # Convert lists back to numpy arrays
        for name, info in database.items():
            if 'embeddings' in info:
                database[name]['embeddings'] = [np.array(emb) if isinstance(emb, list) else emb for emb in info['embeddings']]
                # Load facial image if it exists
                if 'face_image' in info:
                    database[name]['face_image'] = np.array(info['face_image']) if isinstance(info['face_image'], list) else info['face_image']
            elif 'embedding' in info:
                database[name]['embedding'] = np.array(info['embedding']) if isinstance(info['embedding'], list) else info['embedding']
                # Load facial image if it exists
                if 'face_image' in info:
                    database[name]['face_image'] = np.array(info['face_image']) if isinstance(info['face_image'], list) else info['face_image']
        
        return database
    except Exception as e:
        st.error(f"Error loading database: {str(e)}")
        return {}

def export_database_json():
    """
    Export the face database to a JSON file for sharing or backup.
    
    Converts numpy arrays to base64-encoded strings for JSON compatibility.
    
    Returns:
        str: Path to the exported JSON file, or None if an error occurs
    """
    try:
        if 'face_database' in st.session_state and st.session_state.face_database:
            # Create a serializable version of the database
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
                    
                    # Include facial image if it exists
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
                    
                    # Include facial image if it exists
                    if 'face_image' in info:
                        serializable_db[name]['face_image'] = base64.b64encode(
                            np.array(info['face_image']).tobytes()
                        ).decode('utf-8')
                        serializable_db[name]['face_image_shape'] = info['face_image'].shape
            
            # Save to a JSON file
            export_file = "face_database_export.json"
            with open(export_file, 'w') as f:
                json.dump(serializable_db, f, indent=2)
            
            return export_file
        return None
    except Exception as e:
        st.error(f"Error exporting database: {str(e)}")
        return None

def import_database_json(json_file):
    """
    Import a face database from a JSON file.
    
    Converts base64-encoded strings back to numpy arrays.
    
    Args:
        json_file: The JSON file to import
        
    Returns:
        dict: The imported face database, or an empty dictionary if an error occurs
    """
    try:
        content = json_file.read()
        imported_db = json.loads(content)
        
        # Convert base64-encoded data to numpy arrays
        for name, info in imported_db.items():
            if 'embeddings' in info:
                imported_db[name]['embeddings'] = [
                    np.frombuffer(base64.b64decode(emb), dtype=np.float32) 
                    for emb in info['embeddings']
                ]
                
                # Import facial image if it exists
                if 'face_image' in info and 'face_image_shape' in info:
                    face_data = np.frombuffer(base64.b64decode(info['face_image']), dtype=np.uint8)
                    shape = info['face_image_shape']
                    imported_db[name]['face_image'] = face_data.reshape(shape)
            elif 'embedding' in info:
                imported_db[name]['embedding'] = np.frombuffer(
                    base64.b64decode(info['embedding']), dtype=np.float32
                )
                
                # Import facial image if it exists
                if 'face_image' in info and 'face_image_shape' in info:
                    face_data = np.frombuffer(base64.b64decode(info['face_image']), dtype=np.uint8)
                    shape = info['face_image_shape']
                    imported_db[name]['face_image'] = face_data.reshape(shape)
        
        return imported_db
    except Exception as e:
        st.error(f"Error importing database: {str(e)}")
        return {}

def print_database_info():
    """
    Print information about the current face database for debugging.
    
    Displays the number of entries, names in the database, and details about
    the first entry to help with troubleshooting.
    """
    if 'face_database' in st.session_state:
        db = st.session_state.face_database
        st.sidebar.write("--- Database Debug Info ---")
        st.sidebar.write(f"Database contains {len(db)} entries")
        
        # Show names in the database
        if db:
            names = list(db.keys())
            st.sidebar.write(f"Names in database: {', '.join(names)}")
            
            # Show details of the first element
            if names:
                first_entry = db[names[0]]
                st.sidebar.write(f"Sample entry for '{names[0]}':")
                if 'embeddings' in first_entry:
                    st.sidebar.write(f"- Has {len(first_entry['embeddings'])} embeddings")
                    st.sidebar.write(f"- Models: {', '.join(first_entry['models'])}")
                    st.sidebar.write(f"- Count: {first_entry['count']}")
                    
                    # Show if it has an image
                    if 'face_image' in first_entry:
                        st.sidebar.write(f"- Has reference face image: {first_entry['face_image'].shape}")
                    else:
                        st.sidebar.write("- No reference image")
                elif 'embedding' in first_entry:
                    st.sidebar.write("- Has single embedding (old format)")
                    
                    # Show if it has an image
                    if 'face_image' in first_entry:
                        st.sidebar.write(f"- Has reference face image: {first_entry['face_image'].shape}")
                    else:
                        st.sidebar.write("- No reference image")
        else:
            st.sidebar.write("Database is empty") 