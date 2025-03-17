"""
Módulo para parchear DeepFace y RetinaFace para compatibilidad con diferentes versiones de TensorFlow
"""
import os
import sys
import importlib
import warnings

def patch_retina_face():
    """Parchea RetinaFace para funcionar con Keras independiente o integrado en TensorFlow"""
    try:
        # Verificar si podemos importar retina_face
        import retina_face
        
        # Verificar la versión de TensorFlow
        import tensorflow as tf
        tf_version = tf.__version__
        
        # Para TF 2.15.x, usamos keras estándar
        if tf_version.startswith('2.15'):
            try:
                import keras
                print(f"Using standard Keras {keras.__version__} with TensorFlow {tf_version}")
                
                # Monkeypatch para RetinaFace si es necesario
                try:
                    # Intentar importar el módulo que podría usar keras
                    from retina_face.commons import postprocess
                    if not hasattr(postprocess, '_keras_patched'):
                        # Verificar si está usando tf.keras
                        if hasattr(postprocess, 'keras') and postprocess.keras.__name__ == 'tensorflow.keras':
                            print("Patching RetinaFace to use standard keras instead of tf.keras")
                            postprocess.keras = keras
                            postprocess._keras_patched = True
                except ImportError:
                    pass
            except ImportError:
                print("Standard Keras not found, using tf.keras")
        
        # Para TF 2.19.x, necesitamos tf-keras
        elif tf_version.startswith('2.19'):
            try:
                import tf_keras
                print(f"Using tf-keras with TensorFlow {tf_version}")
                
                # Monkeypatch para RetinaFace si es necesario
                try:
                    from retina_face.commons import postprocess
                    if not hasattr(postprocess, '_keras_patched'):
                        if hasattr(postprocess, 'keras'):
                            print("Patching RetinaFace to use tf-keras")
                            postprocess.keras = tf_keras
                            postprocess._keras_patched = True
                except ImportError:
                    pass
            except ImportError:
                print("Warning: tf-keras not installed. RetinaFace may not work properly.")
    
    except ImportError as e:
        print(f"Warning: Could not patch RetinaFace: {e}")

def patch_deepface():
    """Parchea DeepFace para funcionar con diferentes versiones de TensorFlow"""
    try:
        import deepface
        import tensorflow as tf
        tf_version = tf.__version__
        
        if tf_version.startswith('2.19'):
            try:
                import tf_keras
                # Intentar parchear los módulos relevantes de DeepFace
                deepface_modules = [
                    'deepface.commons.functions',
                    'deepface.detectors.RetinaFaceWrapper',
                    'deepface.detectors.FaceDetector'
                ]
                
                for module_name in deepface_modules:
                    try:
                        module = importlib.import_module(module_name)
                        if hasattr(module, 'keras') and module.keras.__name__ == 'tensorflow.keras':
                            module.keras = tf_keras
                            print(f"Patched {module_name} to use tf-keras")
                    except (ImportError, AttributeError):
                        pass
            except ImportError:
                print("Warning: tf-keras not installed. DeepFace may not work properly with TF 2.19")
    except ImportError as e:
        print(f"Warning: Could not patch DeepFace: {e}")

def apply_patches():
    """Aplica todos los parches necesarios"""
    warnings.filterwarnings('ignore')  # Reducir mensajes de advertencia
    patch_retina_face()
    patch_deepface()
    print("Patches applied successfully")

if __name__ == "__main__":
    apply_patches() 