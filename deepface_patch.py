"""
DeepFace and RetinaFace Compatibility Patch Module

This module provides patches to ensure compatibility between DeepFace, RetinaFace,
and different versions of TensorFlow and Keras.

It addresses issues that arise from TensorFlow's changing Keras integration
across different versions by dynamically patching the necessary modules at runtime.
"""
import os
import sys
import importlib
import warnings

def patch_retina_face():
    """
    Patch RetinaFace library to work with standalone Keras or TensorFlow-integrated Keras.
    
    This function detects the current TensorFlow version and modifies RetinaFace's
    internal references to use the appropriate Keras implementation:
    - For TF 2.15.x: Uses standard standalone Keras
    - For TF 2.19.x: Uses tf-keras package
    
    The patches are applied using monkey patching to avoid modifying source files.
    """
    try:
        # Check if RetinaFace is available
        import retina_face
        
        # Check TensorFlow version
        import tensorflow as tf
        tf_version = tf.__version__
        
        # For TF 2.15.x, use standard keras
        if tf_version.startswith('2.15'):
            try:
                import keras
                print(f"Using standard Keras {keras.__version__} with TensorFlow {tf_version}")
                
                # Monkey patch RetinaFace if necessary
                try:
                    # Try to import the module that might use keras
                    from retina_face.commons import postprocess
                    if not hasattr(postprocess, '_keras_patched'):
                        # Check if it's using tf.keras
                        if hasattr(postprocess, 'keras') and postprocess.keras.__name__ == 'tensorflow.keras':
                            print("Patching RetinaFace to use standard keras instead of tf.keras")
                            postprocess.keras = keras
                            postprocess._keras_patched = True
                except ImportError:
                    pass
            except ImportError:
                print("Standard Keras not found, using tf.keras")
        
        # For TF 2.19.x, we need tf-keras
        elif tf_version.startswith('2.19'):
            try:
                import tf_keras
                print(f"Using tf-keras with TensorFlow {tf_version}")
                
                # Monkey patch RetinaFace if necessary
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
    """
    Patch DeepFace library to work with different TensorFlow versions.
    
    This function identifies and patches key DeepFace modules that depend on
    Keras, ensuring they use the appropriate Keras implementation for the
    current TensorFlow version.
    
    Particularly important for TensorFlow 2.19+, where Keras is no longer
    bundled with TensorFlow.
    """
    try:
        import deepface
        import tensorflow as tf
        tf_version = tf.__version__
        
        if tf_version.startswith('2.19'):
            try:
                import tf_keras
                # Try to patch relevant DeepFace modules
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
    """
    Apply all necessary compatibility patches.
    
    This function serves as the main entry point to apply all patches
    for ensuring compatibility between face detection libraries and
    the current TensorFlow/Keras environment.
    """
    warnings.filterwarnings('ignore')  # Reduce warning messages
    patch_retina_face()
    patch_deepface()
    print("Patches applied successfully")

if __name__ == "__main__":
    apply_patches() 