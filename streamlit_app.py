import streamlit as st
import cv2
import numpy as np
from PIL import Image
from io import BytesIO
import base64
import tempfile
import os
import time
import urllib.request
import matplotlib.pyplot as plt
import pickle
from sklearn.metrics.pairwise import cosine_similarity # type: ignore
import pandas as pd

# Importar las utilidades para la base de datos de rostros
try:
    from face_database_utils import save_face_database, load_face_database, export_database_json, import_database_json, print_database_info
    DATABASE_UTILS_AVAILABLE = True
except ImportError:
    DATABASE_UTILS_AVAILABLE = False
    st.warning("Database utilities are not available. Face recognition data will not be persistent between sessions.")

# Importar DeepFace para reconocimiento facial avanzado
try:
    from deepface import DeepFace
    DEEPFACE_AVAILABLE = True
except ImportError:
    DEEPFACE_AVAILABLE = False

# Import functions for face comparison
try:
    from face_comparison import compare_faces, compare_faces_embeddings, generate_comparison_report_english, draw_face_matches, extract_face_embeddings, extract_face_embeddings_all_models
    FACE_COMPARISON_AVAILABLE = True
except ImportError:
    FACE_COMPARISON_AVAILABLE = False
    st.warning("Face comparison functions are not available. Please check your installation.")

# Función principal que encapsula toda la aplicación
def main():
    # Set page config with custom title and layout
    st.set_page_config(
        page_title="Advanced Face & Feature Detection",
        page_icon="👤",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Sidebar for navigation and controls
    st.sidebar.title("Controls & Settings")

    # Initialize session_state to store original image and camera state
    if 'original_image' not in st.session_state:
        st.session_state.original_image = None
    if 'camera_running' not in st.session_state:
        st.session_state.camera_running = False
    if 'feature_camera_running' not in st.session_state:
        st.session_state.feature_camera_running = False

    # Navigation menu
    app_mode = st.sidebar.selectbox(
        "Choose the app mode",
        ["About", "Face Detection", "Feature Detection", "Comparison Mode", "Face Recognition"]
    )

    # Function to load DNN models with caching and auto-download
    @st.cache_resource
    def load_face_model():
        # No need to create directory as we're using the root directory
        #
            #
        
        # Correct model file names
        modelFile = "res10_300x300_ssd_iter_140000.caffemodel"
        configFile = "deploy.prototxt.txt"
        
        # Check if files exist
        missing_files = []
        if not os.path.exists(modelFile):
            missing_files.append(modelFile)
        if not os.path.exists(configFile):
            missing_files.append(configFile)
        
        if missing_files:
            st.error("Missing model files: " + ", ".join(missing_files))
            st.error("Please manually download the following files:")
            st.code("""
            1. Download the model file:
               URL: https://raw.githubusercontent.com/sr6033/face-detection-with-OpenCV-and-DNN/master/res10_300x300_ssd_iter_140000.caffemodel
               Save as: res10_300x300_ssd_iter_140000.caffemodel
               
            2. Download the configuration file:
               URL: https://raw.githubusercontent.com/sr6033/face-detection-with-OpenCV-and-DNN/master/deploy.prototxt.txt
               Save as: deploy.prototxt.txt
            """)
            st.stop()
        
        # Load model
        try:
            net = cv2.dnn.readNetFromCaffe(configFile, modelFile)
            return net
        except Exception as e:
            st.error(f"Error loading model: {e}")
            st.stop()

    @st.cache_resource
    def load_feature_models():
        # Load pre-trained models for eye and smile detection
        eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
        return eye_cascade, smile_cascade

    # Function for detecting faces in an image
    def detect_face_dnn(net, frame, conf_threshold=0.5):
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], False, False)
        net.setInput(blob)
        detections = net.forward()
        
        # Procesar las detecciones para devolver una lista de bounding boxes
        bboxes = []
        frame_h = frame.shape[0]
        frame_w = frame.shape[1]
        
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > conf_threshold:
                x1 = int(detections[0, 0, i, 3] * frame_w)
                y1 = int(detections[0, 0, i, 4] * frame_h)
                x2 = int(detections[0, 0, i, 5] * frame_w)
                y2 = int(detections[0, 0, i, 6] * frame_h)
                
                # Asegurarse de que las coordenadas estén dentro de los límites de la imagen
                x1 = max(0, min(x1, frame_w - 1))
                y1 = max(0, min(y1, frame_h - 1))
                x2 = max(0, min(x2, frame_w - 1))
                y2 = max(0, min(y2, frame_h - 1))
                
                # Añadir el bounding box y la confianza
                bboxes.append([x1, y1, x2, y2, confidence])
        
        return bboxes

    # Function for processing face detections
    def process_face_detections(frame, detections, conf_threshold=0.5, bbox_color=(0, 255, 0)):
        # Create a copy for drawing on
        result_frame = frame.copy()
        
        # Filtrar detecciones por umbral de confianza
        bboxes = []
        for detection in detections:
            if len(detection) == 5:  # Asegurarse de que la detección tiene el formato correcto
                x1, y1, x2, y2, confidence = detection
                if confidence >= conf_threshold:
                    # Dibujar el bounding box
                    cv2.rectangle(result_frame, (x1, y1), (x2, y2), bbox_color, 2)
                    
                    # Añadir texto con la confianza
                    label = f"{confidence:.2f}"
                    cv2.putText(result_frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, bbox_color, 2)
                    
                    # Añadir a la lista de bounding boxes
                    bboxes.append([x1, y1, x2, y2, confidence])
        
        return result_frame, bboxes

    # Function to detect facial features (eyes, smile) with improved profile face handling
    def detect_facial_features(frame, bboxes, eye_cascade, smile_cascade, detect_eyes=True, detect_smile=True, smile_sensitivity=15, eye_sensitivity=5):
        result_frame = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Counters for detection summary
        eye_count = 0
        smile_count = 0
        
        for bbox in bboxes:
            x1, y1, x2, y2, _ = bbox
            face_width = x2 - x1
            face_height = y2 - y1
            
            # Detect eyes if enabled
            if detect_eyes:
                # Adjust region of interest to focus on the upper part of the face
                upper_face_y1 = y1
                upper_face_y2 = y1 + int(face_height * 0.45)  # Reduced to focus more on the eye area
                
                # Extract ROI for eyes
                eye_roi_gray = gray[upper_face_y1:upper_face_y2, x1:x2]
                eye_roi_color = result_frame[upper_face_y1:upper_face_y2, x1:x2]
                
                if eye_roi_gray.size > 0:
                    # Enhance contrast for better detection
                    eye_roi_gray = cv2.equalizeHist(eye_roi_gray)
                    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                    eye_roi_gray = clahe.apply(eye_roi_gray)
                    
                    # Detect eyes with adjusted parameters
                    eyes = eye_cascade.detectMultiScale(
                        eye_roi_gray,
                        scaleFactor=1.05,
                        minNeighbors=max(3, eye_sensitivity),
                        minSize=(int(face_width * 0.1), int(face_width * 0.1)),
                        maxSize=(int(face_width * 0.25), int(face_width * 0.25))
                    )
                    
                    # Process detected eyes
                    if len(eyes) > 0:
                        # Sort by size and position
                        eyes = sorted(eyes, key=lambda e: (e[2] * e[3], -e[1]))  # Sort by size and vertical position
                        eyes = eyes[:2]  # Take at most 2 largest eyes
                        
                        for (ex, ey, ew, eh) in eyes:
                            # Validate eye size and position
                            if ew * eh > (face_width * face_height * 0.01):  # Minimum size threshold
                                eye_count += 1
                                cv2.rectangle(eye_roi_color, (ex, ey), (ex+ew, ey+eh), (255, 0, 0), 2)
                                cv2.putText(eye_roi_color, "Eye", (ex, ey-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            
            # Detect smile if enabled
            if detect_smile:
                # Adjust region of interest for smile detection
                lower_face_y1 = y1 + int(face_height * 0.5)  # Start from middle of face
                lower_face_y2 = y2
                
                # Extract ROI for smile
                smile_roi_gray = gray[lower_face_y1:lower_face_y2, x1:x2]
                smile_roi_color = result_frame[lower_face_y1:lower_face_y2, x1:x2]
                
                if smile_roi_gray.size > 0:
                    # Enhance contrast for better detection
                    smile_roi_gray = cv2.equalizeHist(smile_roi_gray)
                    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                    smile_roi_gray = clahe.apply(smile_roi_gray)
                    
                    # Detect smiles with adjusted parameters
                    smiles = smile_cascade.detectMultiScale(
                        smile_roi_gray,
                        scaleFactor=1.1,
                        minNeighbors=max(5, smile_sensitivity),
                        minSize=(int(face_width * 0.3), int(face_width * 0.15)),
                        maxSize=(int(face_width * 0.6), int(face_width * 0.3))
                    )
                    
                    # Process detected smiles
                    if len(smiles) > 0:
                        # Sort by size and take the largest
                        smiles = sorted(smiles, key=lambda s: s[2] * s[3], reverse=True)
                        sx, sy, sw, sh = smiles[0]
                        
                        # Validate smile size and position
                        if sw * sh > (face_width * face_height * 0.05):  # Minimum size threshold
                            smile_count += 1
                            cv2.rectangle(smile_roi_color, (sx, sy), (sx+sw, sy+sh), (0, 0, 255), 2)
                            cv2.putText(smile_roi_color, "Smile", (sx, sy-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        return result_frame, eye_count, smile_count

    # Función para detectar atributos faciales (edad, género, emoción)
    def detect_face_attributes(image, bbox):
        """
        Detecta atributos faciales como edad, género y emoción usando DeepFace.
        
        Args:
            image: Imagen en formato OpenCV (BGR)
            bbox: Bounding box de la cara [x1, y1, x2, y2, conf]
            
        Returns:
            Diccionario con los atributos detectados
        """
        if not DEEPFACE_AVAILABLE:
            return None
        
        try:
            x1, y1, x2, y2, _ = bbox
            face_img = image[y1:y2, x1:x2]
            
            # Convertir de BGR a RGB para DeepFace
            face_img_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            
            # Analyze atributos faciales
            attributes = DeepFace.analyze(
                img_path=face_img_rgb,
                actions=['age', 'gender', 'emotion'],
                enforce_detection=False,
                detector_backend="opencv"
            )
            
            return attributes[0]
        
        except Exception as e:
            st.error(f"Error detecting facial attributes: {str(e)}")
            return None

    # Function to apply age and gender detection (placeholder - would need additional models)
    def detect_age_gender(frame, bboxes):
        # Versión mejorada que usa DeepFace si está disponible
        result_frame = frame.copy()
        
        for i, bbox in enumerate(bboxes):
            x1, y1, x2, y2, _ = bbox
            
            if DEEPFACE_AVAILABLE:
                # Intentar usar DeepFace para análisis facial
                attributes = detect_face_attributes(frame, bbox)
                
                if attributes:
                    # Extraer información de atributos
                    age = attributes.get('age', 'Unknown')
                    gender = attributes.get('gender', 'Unknown')
                    emotion = attributes.get('dominant_emotion', 'Unknown').capitalize()
                    gender_prob = attributes.get('gender', {}).get('Woman', 0)
                    
                    # Determinar color basado en confianza
                    if gender == 'Woman':
                        gender_color = (255, 0, 255)  # Magenta para mujer
                    else:
                        gender_color = (255, 0, 0)    # Azul para hombre
                    
                    # Añadir texto con información
                    cv2.putText(result_frame, f"Age: {age}", (x1, y2+20), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
                    cv2.putText(result_frame, f"Gender: {gender}", (x1, y2+40), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, gender_color, 2)
                    cv2.putText(result_frame, f"Emotion: {emotion}", (x1, y2+60), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                else:
                    # Fallback si DeepFace falla
                    cv2.putText(result_frame, "Age: Unknown", (x1, y2+20), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
                    cv2.putText(result_frame, "Gender: Unknown", (x1, y2+40), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
            else:
                # Usar texto placeholder si DeepFace no está disponible
                cv2.putText(result_frame, "Age: 25-35", (x1, y2+20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
                cv2.putText(result_frame, "Gender: Unknown", (x1, y2+40), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
        
        return result_frame

    # Function to generate download link for processed image
    def get_image_download_link(img, filename, text):
        buffered = BytesIO()
        img.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        href = f'<a href="data:file/txt;base64,{img_str}" download="{filename}">{text}</a>'
        return href

    # Function to process video frames
    def process_video(video_path, face_net, eye_cascade, smile_cascade, conf_threshold=0.5, detect_eyes=True, detect_smile=True, bbox_color=(0, 255, 0), smile_sensitivity=15, eye_sensitivity=5):
        cap = cv2.VideoCapture(video_path)
        
        # Get video properties
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        # Create temporary output file
        temp_dir = tempfile.mkdtemp()
        temp_output_path = os.path.join(temp_dir, "processed_video.mp4")
        
        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(temp_output_path, fourcc, fps, (frame_width, frame_height))
        
        # Create a progress bar
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Process video frames
        current_frame = 0
        processing_times = []
        
        # Total counters for statistics
        total_faces = 0
        total_eyes = 0
        total_smiles = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Start timing for performance metrics
            start_time = time.time()
            
            # Detect faces
            detections = detect_face_dnn(face_net, frame, conf_threshold)
            processed_frame, bboxes = process_face_detections(frame, detections, conf_threshold, bbox_color)
            
            # Update face counter
            total_faces += len(bboxes)
            
            # Detect facial features if enabled
            if detect_eyes or detect_smile:
                processed_frame, eye_count, smile_count = detect_facial_features(
                    processed_frame, 
                    bboxes, 
                    eye_cascade, 
                    smile_cascade,
                    detect_eyes,
                    detect_smile,
                    smile_sensitivity,
                    eye_sensitivity
                )
                # Update counters
                total_eyes += eye_count
                total_smiles += smile_count
            
            # End timing
            processing_times.append(time.time() - start_time)
            
            # Write the processed frame
            out.write(processed_frame)
            
            # Update progress
            current_frame += 1
            progress_bar.progress(current_frame / frame_count)
            status_text.text(f"Processing frame {current_frame}/{frame_count}")
        
        # Release resources
        cap.release()
        out.release()
        
        # Calculate and display performance metrics
        if processing_times:
            avg_time = sum(processing_times) / len(processing_times)
            status_text.text(f"Processing complete! Average processing time: {avg_time:.4f}s per frame")
        
        # Return detection statistics
        detection_stats = {
            "faces": total_faces // max(1, current_frame),  # Average per frame
            "eyes": total_eyes // max(1, current_frame),    # Average per frame
            "smiles": total_smiles // max(1, current_frame) # Average per frame
        }
        
        return temp_output_path, temp_dir, detection_stats

    # Camera control functions
    def start_camera():
        st.session_state.camera_running = True

    def stop_camera():
        st.session_state.camera_running = False
        st.session_state.camera_stopped = True

    def start_feature_camera():
        st.session_state.feature_camera_running = True

    def stop_feature_camera():
        st.session_state.feature_camera_running = False
        st.session_state.feature_camera_stopped = True

    if app_mode == "About":
        st.markdown("""
        ## About This App
        
        This application uses OpenCV's Deep Neural Network (DNN) module and Haar Cascade classifiers to detect faces and facial features in images and videos.
        
        ### Features:
        - Face detection using OpenCV DNN
        - Eye and smile detection using Haar Cascades
        - Support for both image and video processing
        - Adjustable confidence threshold
        - Download options for processed media
        - Performance metrics
        
        ### How to use:
        1. Select a mode from the sidebar
        2. Upload an image or video
        3. Adjust settings as needed
        4. View and download the results
        
        ### Technologies Used:
        - Streamlit for the web interface
        - OpenCV for computer vision operations
        - Python for backend processing
        
        ### Models:
        - SSD MobileNet for face detection
        - Haar Cascades for facial features
        """)
        
        # Display a sample image or GIF
        st.image("https://opencv.org/wp-content/uploads/2019/07/detection.gif", caption="Sample face detection", use_container_width=True)

    elif app_mode == "Face Detection":
        # Load the face detection model
        face_net = load_face_model()
        
        # Input type selection (Image or Video)
        input_type = st.sidebar.radio("Select Input Type", ["Image", "Video"])
        
        # Confidence threshold slider
        conf_threshold = st.sidebar.slider(
            "Confidence Threshold", 
            min_value=0.0, 
            max_value=1.0, 
            value=0.5, 
            step=0.05,
            help="Adjust the threshold for face detection confidence (higher = fewer detections but more accurate)"
        )
        
        # Style options
        bbox_color = st.sidebar.color_picker("Bounding Box Color", "#00FF00")
        # Convert hex color to BGR for OpenCV
        bbox_color_rgb = tuple(int(bbox_color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
        bbox_color_bgr = (bbox_color_rgb[2], bbox_color_rgb[1], bbox_color_rgb[0])  # Convert RGB to BGR
        
        # Display processing metrics
        show_metrics = st.sidebar.checkbox("Show Processing Metrics", True)
        
        if input_type == "Image":
            # File uploader for images
            file_buffer = st.file_uploader("Upload an image", type=['jpg', 'jpeg', 'png'])
            
            if file_buffer is not None:
                # Read the file and convert it to OpenCV format
                raw_bytes = np.asarray(bytearray(file_buffer.read()), dtype=np.uint8)
                image = cv2.imdecode(raw_bytes, cv2.IMREAD_COLOR)
                
                # Save la imagen original en session_state para reprocesarla cuando cambie el umbral
                # Usar un identificador único para cada archivo para detectar cambios
                file_id = file_buffer.name + str(file_buffer.size)
                
                if 'file_id' not in st.session_state or st.session_state.file_id != file_id:
                    st.session_state.file_id = file_id
                    st.session_state.original_image = image.copy()
                
                # Display original image
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Original Image")
                    st.image(st.session_state.original_image, channels='BGR', use_container_width=True)
                
                # Start timing for performance metrics
                start_time = time.time()
                
                # Detect faces
                detections = detect_face_dnn(face_net, st.session_state.original_image, conf_threshold)
                processed_image, bboxes = process_face_detections(st.session_state.original_image, detections, conf_threshold, bbox_color_bgr)
                
                # Calculate processing time
                processing_time = time.time() - start_time
                
                # Display the processed image
                with col2:
                    st.subheader("Processed Image")
                    st.image(processed_image, channels='BGR', use_container_width=True)
                    
                    # Convert OpenCV image to PIL for download
                    pil_img = Image.fromarray(processed_image[:, :, ::-1])
                    st.markdown(
                        get_image_download_link(pil_img, "face_detection_result.jpg", "📥 Download Processed Image"),
                        unsafe_allow_html=True
                    )
                
                # Show metrics if enabled
                if show_metrics:
                    st.subheader("Processing Metrics")
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Processing Time", f"{processing_time:.4f} seconds")
                    col2.metric("Faces Detected", len(bboxes))
                    col3.metric("Confidence Threshold", f"{conf_threshold:.2f}")
                    
                    # Display detailed metrics in an expandable section
                    with st.expander("Detailed Detection Information"):
                        if bboxes:
                            st.write("Detected faces with confidence scores:")
                            for i, bbox in enumerate(bboxes):
                                st.write(f"Face #{i+1}: Confidence = {bbox[4]:.4f}")
                        else:
                            st.write("No faces detected in the image.")
        
        else:  # Video mode
            # Video mode options
            video_source = st.radio("Select video source", ["Upload video", "Use webcam"])
            
            if video_source == "Upload video":
                # File uploader for videos
                file_buffer = st.file_uploader("Upload a video", type=['mp4', 'avi', 'mov'])
                
                if file_buffer is not None:
                    # Save uploaded video to temporary file
                    temp_dir = tempfile.mkdtemp()
                    temp_path = os.path.join(temp_dir, "input_video.mp4")
                    
                    with open(temp_path, "wb") as f:
                        f.write(file_buffer.read())
                    
                    # Display original video
                    st.subheader("Original Video")
                    st.video(temp_path)
                    
                    # Load models for feature detection (will be used in the processing)
                    eye_cascade, smile_cascade = load_feature_models()
                    
                    # Process video button
                    if st.button("Process Video"):
                        with st.spinner("Processing video... This may take a while depending on the video length."):
                            # Process the video
                            output_path, output_dir, detection_stats = process_video(
                                temp_path, 
                                face_net, 
                                eye_cascade,
                                smile_cascade,
                                conf_threshold,
                                detect_eyes=True,
                                detect_smile=True,
                                bbox_color=bbox_color_bgr,
                                eye_sensitivity=5
                            )
                            
                            # Display processed video
                            st.subheader("Processed Video")
                            st.video(output_path)
                            
                            # Mostrar estadísticas de detección
                            st.subheader("Detection Summary")
                            summary_col1, summary_col2, summary_col3 = st.columns(3)
                            summary_col1.metric("Avg. Faces per Frame", detection_stats["faces"])
                            
                            if detect_eyes: # type: ignore
                                summary_col2.metric("Avg. Eyes per Frame", detection_stats["eyes"])
                            else:
                                summary_col2.metric("Eyes Detected", "N/A")
                            
                            if detect_smile: # type: ignore
                                summary_col3.metric("Avg. Smiles per Frame", detection_stats["smiles"])
                            else:
                                summary_col3.metric("Smiles Detected", "N/A")
                            
                            # Provide download link
                            with open(output_path, 'rb') as f:
                                video_bytes = f.read()
                            
                            st.download_button(
                                label="📥 Download Processed Video",
                                data=video_bytes,
                                file_name="processed_video.mp4",
                                mime="video/mp4"
                            )
                            
                            # Clean up temporary files
                            try:
                                os.remove(temp_path)
                                os.remove(output_path)
                                os.rmdir(temp_dir)
                                os.rmdir(output_dir)
                            except:
                                pass
            else:  # Use webcam
                st.subheader("Real-time face detection")
                st.write("Click 'Start Camera' to begin real-time face detection.")
                
                # Placeholder for webcam video
                camera_placeholder = st.empty()
                
                # Buttons to control the camera
                col1, col2 = st.columns(2)
                start_button = col1.button("Start Camera", on_click=start_camera)
                stop_button = col2.button("Stop Camera", on_click=stop_camera)
                
                # Show message when camera is stopped
                if 'camera_stopped' in st.session_state and st.session_state.camera_stopped:
                    st.info("Camera stopped. Click 'Start Camera' to activate it again.")
                    st.session_state.camera_stopped = False
                
                if st.session_state.camera_running:
                    st.info("Camera activated. Processing real-time video...")
                    # Initialize webcam
                    cap = cv2.VideoCapture(0)  # 0 is typically the main webcam
                    
                    if not cap.isOpened():
                        st.error("Could not access webcam. Make sure it's connected and not being used by another application.")
                        st.warning("⚠️ Note: If you're using this app on Hugging Face Spaces, webcam access is not supported. Try running this app locally for webcam features.")
                        st.session_state.camera_running = False
                    else:
                        # Display real-time video with face detection
                        try:
                            while st.session_state.camera_running:
                                ret, frame = cap.read()
                                if not ret:
                                    st.error("Error reading frame from camera.")
                                    break
                                
                                # Detect faces
                                detections = detect_face_dnn(face_net, frame, conf_threshold)
                                processed_frame, bboxes = process_face_detections(frame, detections, conf_threshold, bbox_color_bgr)
                                
                                # Display the processed frame
                                camera_placeholder.image(processed_frame, channels="BGR", use_container_width=True)
                                
                                # Small pause to avoid overloading the CPU
                                time.sleep(0.01)
                        finally:
                            # Release the camera when stopped
                            cap.release()

    elif app_mode == "Feature Detection":
        # Load all required models
        face_net = load_face_model()
        eye_cascade, smile_cascade = load_feature_models()
        
        # Feature selection checkboxes
        st.sidebar.subheader("Feature Detection Options")
        detect_eyes = st.sidebar.checkbox("Detect Eyes", True)
        
        # Add controls for eye detection sensitivity
        eye_sensitivity = 5  # Default value
        if detect_eyes:
            eye_sensitivity = st.sidebar.slider(
                "Eye Detection Sensitivity", 
                min_value=1, 
                max_value=10, 
                value=5, 
                step=1,
                help="Adjust the sensitivity of eye detection (lower value = more detections)"
            )
        
        detect_smile = st.sidebar.checkbox("Detect Smile", True)
        
        # Add controls for smile detection sensitivity
        smile_sensitivity = 15  # Default value
        if detect_smile:
            smile_sensitivity = st.sidebar.slider(
                "Smile Detection Sensitivity", 
                min_value=5, 
                max_value=30, 
                value=15, 
                step=1,
                help="Adjust the sensitivity of smile detection (lower value = more detections)"
            )
        
        detect_age_gender_option = st.sidebar.checkbox("Detect Age/Gender (Demo)", False)
        
        # Confidence threshold slider
        conf_threshold = st.sidebar.slider(
            "Face Detection Confidence", 
            min_value=0.0, 
            max_value=1.0, 
            value=0.5, 
            step=0.05
        )
        
        # Style options
        bbox_color = st.sidebar.color_picker("Bounding Box Color", "#00FF00")
        # Convert hex color to BGR for OpenCV
        bbox_color_rgb = tuple(int(bbox_color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
        bbox_color_bgr = (bbox_color_rgb[2], bbox_color_rgb[1], bbox_color_rgb[0])  # Convert RGB to BGR
        
        # Input type selection
        input_type = st.sidebar.radio("Select Input Type", ["Image", "Video"])
        
        if input_type == "Image":
            # File uploader for images
            file_buffer = st.file_uploader("Upload an image", type=['jpg', 'jpeg', 'png'])
            
            if file_buffer is not None:
                # Read the file and convert it to OpenCV format
                raw_bytes = np.asarray(bytearray(file_buffer.read()), dtype=np.uint8)
                image = cv2.imdecode(raw_bytes, cv2.IMREAD_COLOR)
                
                # Save la imagen original en session_state para reprocesarla cuando cambie el umbral
                # Usar un identificador único para cada archivo para detectar cambios
                file_id = file_buffer.name + str(file_buffer.size)
                
                if 'feature_file_id' not in st.session_state or st.session_state.feature_file_id != file_id:
                    st.session_state.feature_file_id = file_id
                    st.session_state.feature_original_image = image.copy()
                
                # Display original image
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Original Image")
                    st.image(st.session_state.feature_original_image, channels='BGR', use_container_width=True)
                
                # Start processing with face detection
                detections = detect_face_dnn(face_net, st.session_state.feature_original_image, conf_threshold)
                processed_image, bboxes = process_face_detections(st.session_state.feature_original_image, detections, conf_threshold, bbox_color_bgr)
                
                # Inicializar contadores
                eye_count = 0
                smile_count = 0
                
                # Detect facial features if any options are enabled
                if detect_eyes or detect_smile:
                    processed_image, eye_count, smile_count = detect_facial_features(
                        processed_image, 
                        bboxes,
                        eye_cascade,
                        smile_cascade,
                        detect_eyes,
                        detect_smile,
                        smile_sensitivity,
                        eye_sensitivity
                    )
                    
                # Apply age/gender detection if enabled (demo purpose)
                if detect_age_gender_option:
                    processed_image = detect_age_gender(processed_image, bboxes)
                
                # Display the processed image
                with col2:
                    st.subheader("Processed Image")
                    st.image(processed_image, channels='BGR', use_container_width=True)
                    
                    # Convert OpenCV image to PIL for download
                    pil_img = Image.fromarray(processed_image[:, :, ::-1])
                    st.markdown(
                        get_image_download_link(pil_img, "feature_detection_result.jpg", "📥 Download Processed Image"),
                        unsafe_allow_html=True
                    )
                
                # Display detection summary
                st.subheader("Detection Summary")
                summary_col1, summary_col2, summary_col3 = st.columns(3)
                summary_col1.metric("Faces Detected", len(bboxes))
                
                if detect_eyes:
                    summary_col2.metric("Eyes Detected", eye_count)
                else:
                    summary_col2.metric("Eyes Detected", "N/A")
                
                if detect_smile:
                    summary_col3.metric("Smiles Detected", smile_count)
                else:
                    summary_col3.metric("Smiles Detected", "N/A")
        
        else:  # Video mode
            st.write("Facial feature detection in video")
            
            # Video mode options
            video_source = st.radio("Select video source", ["Upload video", "Use webcam"])
            
            if video_source == "Upload video":
                st.write("Upload a video to process with facial feature detection.")
                # Similar implementation to Face Detection mode for uploaded videos
                file_buffer = st.file_uploader("Upload a video", type=['mp4', 'avi', 'mov'])
                
                if file_buffer is not None:
                    # Save uploaded video to temporary file
                    temp_dir = tempfile.mkdtemp()
                    temp_path = os.path.join(temp_dir, "input_video.mp4")
                    
                    with open(temp_path, "wb") as f:
                        f.write(file_buffer.read())
                    
                    # Display original video
                    st.subheader("Original Video")
                    st.video(temp_path)
                    
                    # Process video button
                    if st.button("Process Video"):
                        with st.spinner("Processing video... This may take a while depending on the video length."):
                            # Process the video with feature detection
                            output_path, output_dir, detection_stats = process_video(
                                temp_path, 
                                face_net, 
                                eye_cascade,
                                smile_cascade,
                                conf_threshold,
                                detect_eyes=True,
                                detect_smile=True,
                                bbox_color=bbox_color_bgr,
                                smile_sensitivity=smile_sensitivity,
                                eye_sensitivity=eye_sensitivity
                            )
                            
                            # Display processed video
                            st.subheader("Processed Video")
                            st.video(output_path)
                            
                            # Mostrar estadísticas de detección
                            st.subheader("Detection Summary")
                            summary_col1, summary_col2, summary_col3 = st.columns(3)
                            summary_col1.metric("Avg. Faces per Frame", detection_stats["faces"])
                            
                            if detect_eyes:
                                summary_col2.metric("Avg. Eyes per Frame", detection_stats["eyes"])
                            else:
                                summary_col2.metric("Eyes Detected", "N/A")
                            
                            if detect_smile:
                                summary_col3.metric("Avg. Smiles per Frame", detection_stats["smiles"])
                            else:
                                summary_col3.metric("Smiles Detected", "N/A")
                            
                            # Provide download link
                            with open(output_path, 'rb') as f:
                                video_bytes = f.read()
                            
                            st.download_button(
                                label="📥 Download Processed Video",
                                data=video_bytes,
                                file_name="feature_detection_video.mp4",
                                mime="video/mp4"
                            )
                            
                            # Clean up temporary files
                            try:
                                os.remove(temp_path)
                                os.remove(output_path)
                                os.rmdir(temp_dir)
                                os.rmdir(output_dir)
                            except:
                                pass
            else:  # Usar cámara web
                st.subheader("Real-time facial feature detection")
                st.write("Click 'Start Camera' to begin real-time detection.")
                
                # Placeholder for webcam video
                camera_placeholder = st.empty()
                
                # Buttons to control the camera
                col1, col2 = st.columns(2)
                start_button = col1.button("Start Camera", on_click=start_feature_camera)
                stop_button = col2.button("Stop Camera", on_click=stop_feature_camera)
                
                # Show message when camera is stopped
                if 'feature_camera_stopped' in st.session_state and st.session_state.feature_camera_stopped:
                    st.info("Camera stopped. Click 'Start Camera' to activate it again.")
                    st.session_state.feature_camera_stopped = False
                
                if st.session_state.feature_camera_running:
                    st.info("Camera activated. Processing real-time video with feature detection...")
                    # Initialize webcam
                    cap = cv2.VideoCapture(0)  # 0 is typically the main webcam
                    
                    if not cap.isOpened():
                        st.error("Could not access webcam. Make sure it's connected and not being used by another application.")
                        st.warning("⚠️ Note: If you're using this app on Hugging Face Spaces, webcam access is not supported. Try running this app locally for webcam features.")
                        st.session_state.feature_camera_running = False
                    else:
                        # Display real-time video with face and feature detection
                        try:
                            # Create placeholders for metrics
                            metrics_placeholder = st.empty()
                            metrics_col1, metrics_col2, metrics_col3 = metrics_placeholder.columns(3)
                            
                            # Initialize counters
                            face_count_total = 0
                            eye_count_total = 0
                            smile_count_total = 0
                            frame_count = 0
                            
                            while st.session_state.feature_camera_running:
                                ret, frame = cap.read()
                                if not ret:
                                    st.error("Error reading frame from camera.")
                                    break
                                
                                # Detect faces
                                detections = detect_face_dnn(face_net, frame, conf_threshold)
                                processed_frame, bboxes = process_face_detections(frame, detections, conf_threshold, bbox_color_bgr)
                                
                                # Update face counter
                                face_count = len(bboxes)
                                face_count_total += face_count
                                
                                # Initialize counters for this frame
                                eye_count = 0
                                smile_count = 0
                                
                                # Detect facial features if enabled
                                if detect_eyes or detect_smile:
                                    processed_frame, eye_count, smile_count = detect_facial_features(
                                        processed_frame, 
                                        bboxes,
                                        eye_cascade,
                                        smile_cascade,
                                        detect_eyes,
                                        detect_smile,
                                        smile_sensitivity,
                                        eye_sensitivity
                                    )
                                    
                                    # Update total counters
                                    eye_count_total += eye_count
                                    smile_count_total += smile_count
                                
                                # Apply age/gender detection if enabled
                                if detect_age_gender_option:
                                    processed_frame = detect_age_gender(processed_frame, bboxes)
                                
                                # Display the processed frame
                                camera_placeholder.image(processed_frame, channels="BGR", use_container_width=True)
                                
                                # Update frame counter
                                frame_count += 1
                                
                                # Update metrics every 5 frames to avoid overloading the interface
                                if frame_count % 5 == 0:
                                    metrics_col1.metric("Faces Detected", face_count)
                                    
                                    if detect_eyes:
                                        metrics_col2.metric("Eyes Detected", eye_count)
                                    else:
                                        metrics_col2.metric("Eyes Detected", "N/A")
                                    
                                    if detect_smile:
                                        metrics_col3.metric("Smiles Detected", smile_count)
                                    else:
                                        metrics_col3.metric("Smiles Detected", "N/A")
                                
                                # Small pause to avoid overloading the CPU
                                time.sleep(0.01)
                        finally:
                            # Release the camera when stopped
                            cap.release()

    elif app_mode == "Comparison Mode":
        st.subheader("Face Comparison")
        st.write("Upload two images to compare faces between them.")
        
        # Añadir explicación sobre la interpretación de resultados
        with st.expander("📌 How to interpret similarity results"):
            st.markdown("""
            ### Facial Similarity Interpretation Guide
            
            The system calculates similarity between faces based on multiple facial features and characteristics.
            
            **Similarity Ranges:**
            - **70-100%**: HIGH Similarity - Very likely to be the same person or identical twins
            - **50-70%**: MEDIUM Similarity - Possible match, requires verification
            - **30-50%**: LOW Similarity - Different people with some similar features
            - **0-30%**: VERY LOW Similarity - Completely different people
            
            **Enhanced Comparison System:**
            The system uses a sophisticated approach that:
            1. Analyzes multiple facial characteristics with advanced precision
            2. Evaluates hair style/color, facial structure, texture patterns, and expressions with improved accuracy
            3. Applies a balanced differentiation between similar and different individuals
            4. Creates a clear gap between similar and different people's scores
            5. Reduces scores for people with different facial structures
            6. Applies penalty factors for critical differences in facial features
            
            **Features Analyzed:**
            - Facial texture patterns (HOG features)
            - Eye region characteristics (highly weighted)
            - Nose bridge features
            - Hair style and color patterns (enhanced detection)
            - Precise facial proportions and structure
            - Texture and edge patterns
            - Facial expressions
            - Critical difference markers (aspect ratio, brightness patterns, texture variance)
            
            **Factors affecting similarity:**
            - Face angle and expression
            - Lighting conditions
            - Age differences
            - Image quality
            - Gender characteristics (with stronger weighting)
            - Critical facial structure differences
            
            **Important note:** This system is designed to provide highly accurate similarity scores that create a clear distinction between different individuals while still recognizing truly similar people. The algorithm now applies multiple reduction factors to ensure that different people receive appropriately low similarity scores. For official identification, always use certified systems.
            """)
        
        # Load face detection model
        face_net = load_face_model()
        
        # Side-by-side file uploaders
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("First Image")
            file1 = st.file_uploader("Upload first image", type=['jpg', 'jpeg', 'png'], key="file1")
        
        with col2:
            st.write("Second Image")
            file2 = st.file_uploader("Upload second image", type=['jpg', 'jpeg', 'png'], key="file2")
        
        # Set confidence threshold
        conf_threshold = st.slider("Face Detection Confidence", min_value=0.0, max_value=1.0, value=0.5, step=0.05)
        
        # Similarity threshold for considering a match
        similarity_threshold = st.slider("Similarity Threshold (%)", min_value=35.0, max_value=95.0, value=45.0, step=5.0,
                                        help="Minimum percentage of similarity to consider two faces as a match")
        
        # Selección del método de comparación
        comparison_method = st.radio(
            "Facial Comparison Method",
            ["HOG (Fast, effective)", "Embeddings (Slow, more precise)"],
            help="HOG uses histograms of oriented gradients for quick comparison. Embeddings use deep neural networks for greater precision."
        )
        
        # Si se selecciona embeddings, mostrar opciones de modelos y advertencia
        embedding_model = "VGG-Face"
        if comparison_method == "Embeddings (Slow, more precise)" and DEEPFACE_AVAILABLE:
            st.warning("WARNING: The current version of TensorFlow (2.19) may have incompatibilities with some models. It is recommended to use HOG if you experience problems.")
            
            embedding_model = st.selectbox(
                "Embedding model",
                ["VGG-Face", "Facenet", "OpenFace", "ArcFace"],  # Eliminado "DeepFace" de la lista
                help="Select the neural network model to extract facial embeddings"
            )
        elif comparison_method == "Embeddings (Slow, more precise)" and not DEEPFACE_AVAILABLE:
            st.warning("The DeepFace library is not available. Please install with 'pip install deepface' to use embeddings.")
            st.info("Using HOG method by default.")
            comparison_method = "HOG (Fast, effective)"
        
        # Style options
        bbox_color = st.color_picker("Bounding Box Color", "#00FF00")
        # Convert hex color to BGR for OpenCV
        bbox_color_rgb = tuple(int(bbox_color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
        bbox_color_bgr = (bbox_color_rgb[2], bbox_color_rgb[1], bbox_color_rgb[0])  # Convert RGB to BGR
        
        # Process the images when both are uploaded
        if file1 is not None and file2 is not None:
            # Read both images
            raw_bytes1 = np.asarray(bytearray(file1.read()), dtype=np.uint8)
            image1 = cv2.imdecode(raw_bytes1, cv2.IMREAD_COLOR)
            
            raw_bytes2 = np.asarray(bytearray(file2.read()), dtype=np.uint8)
            image2 = cv2.imdecode(raw_bytes2, cv2.IMREAD_COLOR)
            
            # Save original images in session_state
            # Use a unique identifier for each file to detect changes
            file1_id = file1.name + str(file1.size)
            file2_id = file2.name + str(file2.size)
            
            if 'file1_id' not in st.session_state or st.session_state.file1_id != file1_id:
                st.session_state.file1_id = file1_id
                st.session_state.original_image1 = image1.copy()
            
            if 'file2_id' not in st.session_state or st.session_state.file2_id != file2_id:
                st.session_state.file2_id = file2_id
                st.session_state.original_image2 = image2.copy()
            
            # Display original images
            with col1:
                st.image(st.session_state.original_image1, channels='BGR', use_container_width=True, caption="Image 1")
            
            with col2:
                st.image(st.session_state.original_image2, channels='BGR', use_container_width=True, caption="Image 2")
            
            # Detect faces in both images
            detections1 = detect_face_dnn(face_net, st.session_state.original_image1, conf_threshold)
            processed_image1, bboxes1 = process_face_detections(st.session_state.original_image1, detections1, conf_threshold, bbox_color_bgr)
            
            detections2 = detect_face_dnn(face_net, st.session_state.original_image2, conf_threshold)
            processed_image2, bboxes2 = process_face_detections(st.session_state.original_image2, detections2, conf_threshold, bbox_color_bgr)
            
            # Display processed images
            st.subheader("Detected Faces")
            proc_col1, proc_col2 = st.columns(2)
            
            with proc_col1:
                st.image(processed_image1, channels='BGR', use_container_width=True, caption="Processed Image 1")
                st.write(f"Faces detected: {len(bboxes1)}")
            
            with proc_col2:
                st.image(processed_image2, channels='BGR', use_container_width=True, caption="Processed Image 2")
                st.write(f"Faces detected: {len(bboxes2)}")
            
            # Compare faces
            if len(bboxes1) == 0 or len(bboxes2) == 0:
                st.warning("Cannot compare: One or both images have no faces detected.")
            else:
                with st.spinner("Comparing faces..."):
                    # Perform face comparison based on selected method
                    if comparison_method == "Embeddings (Slow, more precise)" and DEEPFACE_AVAILABLE:
                        try:
                            st.info(f"Using embedding model: {embedding_model}")
                            comparison_results = compare_faces_embeddings(
                                st.session_state.original_image1, bboxes1,
                                st.session_state.original_image2, bboxes2,
                                model_name=embedding_model
                            )
                        except Exception as e:
                            st.error(f"Error using embeddings: {str(e)}")
                            st.info("Automatically switching to HOG method...")
                            comparison_results = compare_faces(
                                st.session_state.original_image1, bboxes1,
                                st.session_state.original_image2, bboxes2
                            )
                    else:
                        # Usar método HOG tradicional
                        if comparison_method == "Embeddings (Slow, more precise)":
                            st.warning("Using HOG method because DeepFace is not available.")
                        comparison_results = compare_faces(
                            st.session_state.original_image1, bboxes1,
                            st.session_state.original_image2, bboxes2
                        )
                    
                    # Generate comparison report
                    report = generate_comparison_report_english(comparison_results, bboxes1, bboxes2)
                    
                    # Create combined image with match lines
                    combined_image = draw_face_matches(
                        st.session_state.original_image1, bboxes1,
                        st.session_state.original_image2, bboxes2,
                        comparison_results,
                        threshold=similarity_threshold
                    )
                    
                    # Show results
                    st.subheader("Comparison Results")
                    
                    # Show combined image
                    st.image(combined_image, channels='BGR', use_container_width=True, 
                            caption="Visual Comparison (red lines indicate matches above threshold)")
                    
                    # Show similarity statistics
                    st.subheader("Similarity Statistics")
                    
                    # Calculate general statistics
                    all_similarities = []
                    for face_comparisons in comparison_results:
                        for comp in face_comparisons:
                            all_similarities.append(float(comp["similarity"]))
                    
                    if all_similarities:
                        avg_similarity = sum(all_similarities) / len(all_similarities)
                        max_similarity = max(all_similarities)
                        min_similarity = min(all_similarities)
                        
                        # Determinar el nivel de similitud promedio
                        if avg_similarity >= 70:  # Updated from 80 to 70
                            avg_level = "HIGH"
                            avg_color = "normal"
                        elif avg_similarity >= 50:  # Updated from 65 to 50
                            avg_level = "MEDIUM"
                            avg_color = "normal"
                        elif avg_similarity >= 30:  # Updated from 35 to 30
                            avg_level = "LOW"
                            avg_color = "inverse"
                        else:
                            avg_level = "VERY LOW"
                            avg_color = "inverse"
                        
                        # Determinar el nivel de similitud máxima
                        if max_similarity >= 70:  # Updated from 80 to 70
                            max_level = "HIGH"
                            max_color = "normal"
                        elif max_similarity >= 50:  # Updated from 65 to 50
                            max_level = "MEDIUM"
                            max_color = "normal"
                        elif max_similarity >= 30:  # Updated from 35 to 30
                            max_level = "LOW"
                            max_color = "inverse"
                        else:
                            max_level = "VERY LOW"
                            max_color = "inverse"
                        
                        # Show metrics with color coding
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Average Similarity", f"{avg_similarity:.2f}%", 
                                   delta=avg_level, delta_color=avg_color)
                        col2.metric("Maximum Similarity", f"{max_similarity:.2f}%", 
                                   delta=max_level, delta_color=max_color)
                        col3.metric("Minimum Similarity", f"{min_similarity:.2f}%")
                        
                        # Count matches above threshold
                        matches_above_threshold = sum(1 for s in all_similarities if s >= similarity_threshold)
                        st.metric(f"Matches above threshold ({similarity_threshold}%)", matches_above_threshold)
                        
                        # Determine if there are significant matches
                        best_matches = [face_comp[0] for face_comp in comparison_results if face_comp]
                        if any(float(match["similarity"]) >= similarity_threshold for match in best_matches):
                            if any(float(match["similarity"]) >= 70 for match in best_matches):  # Updated from 80 to 70
                                st.success("CONCLUSION: HIGH similarity matches found between images.")
                            elif any(float(match["similarity"]) >= 50 for match in best_matches):  # Updated from 65 to 50
                                st.info("CONCLUSION: MEDIUM similarity matches found between images.")
                            else:
                                st.warning("CONCLUSION: LOW similarity matches found between images.")
                        else:
                            st.error("CONCLUSION: No significant matches found between images.")
                        
                        # Añadir gráfico de distribución de similitud
                        st.subheader("Similarity Distribution")
                        
                        # Crear histograma de similitudes
                        fig, ax = plt.subplots(figsize=(10, 4))
                        bins = [0, 30, 50, 70, 100]  # Updated from [0, 35, 65, 80, 100]
                        labels = ['Very Low', 'Low', 'Medium', 'High']
                        colors = ['darkred', 'red', 'orange', 'green']
                        
                        # Contar cuántos valores caen en cada rango
                        hist_data = [sum(1 for s in all_similarities if bins[i] <= s < bins[i+1]) for i in range(len(bins)-1)]
                        
                        # Crear gráfico de barras
                        bars = ax.bar(labels, hist_data, color=colors)
                        
                        # Añadir etiquetas
                        ax.set_xlabel('Similarity Level')
                        ax.set_ylabel('Number of Comparisons')
                        ax.set_title('Similarity Level Distribution')
                        
                        # Añadir valores sobre las barras
                        for bar in bars:
                            height = bar.get_height()
                            ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                                   f'{int(height)}', ha='center', va='bottom')
                        
                        st.pyplot(fig)
                    
                    # Show detailed report in an expandable section
                    with st.expander("View Detailed Report"):
                        st.write(report)
                    
                    # Provide option to download the report
                    st.download_button(
                        label="📥 Download Comparison Report",
                        data=report,
                        file_name="face_comparison_report.txt",
                        mime="text/plain"
                    )
                    
                    # Provide option to download the combined image
                    pil_combined_img = Image.fromarray(combined_image[:, :, ::-1])
                    buf = BytesIO()
                    pil_combined_img.save(buf, format="JPEG")
                    byte_im = buf.getvalue()
                    
                    st.download_button(
                        label="📥 Download Comparison Image",
                        data=byte_im,
                        file_name="face_comparison.jpg",
                        mime="image/jpeg"
                    )

    # Add a help text for eye detection sensitivity in the Feature Detection mode
    if app_mode == "Feature Detection":
        st.sidebar.markdown("**Eye Detection Settings**")
        st.sidebar.info("Adjust the slider to change the sensitivity of eye detection. A higher value will detect more eyes but may generate false positives.")

    elif app_mode == "Face Recognition":
        st.title("Face Recognition System")
        st.markdown("""
        Este módulo permite registrar rostros y reconocerlos posteriormente en tiempo real o en imágenes.
        Utiliza embeddings faciales para una identificación precisa.
        """)
        
        # Verificar si DeepFace está disponible
        if not DEEPFACE_AVAILABLE:
            st.error("DeepFace is not available. Please install the library with 'pip install deepface'")
            st.stop()
        
        # Load el modelo de detección facial
        face_net = load_face_model()
        
        # Inicializar base de datos de rostros si no existe
        if 'face_database' not in st.session_state:
            if DATABASE_UTILS_AVAILABLE:
                # Cargar la base de datos desde el archivo persistente
                st.session_state.face_database = load_face_database()
                st.sidebar.write(f"Loaded face database with {len(st.session_state.face_database)} entries")
            else:
                st.session_state.face_database = {}
        
        # Imprimir información de depuración
        if DATABASE_UTILS_AVAILABLE:
            print_database_info()
        
        # Crear pestañas para las diferentes funcionalidades
        tab1, tab2, tab3 = st.tabs(["Register Face", "Image Recognition", "Real-time Recognition"])
        
        with tab1:
            st.header("Register New Face")
            
            # Añadir el file_uploader para la imagen
            uploaded_file = st.file_uploader("Upload an image", type=['jpg', 'jpeg', 'png'], key="register_face_image")
            
            # Limpiar el nombre cuando se carga una imagen nueva
            if uploaded_file and 'last_uploaded_file' in st.session_state and st.session_state.last_uploaded_file != uploaded_file.name:
                st.session_state.person_name = ""
            
            if uploaded_file:
                # Guardar el nombre del archivo actual para comparar en la próxima carga
                st.session_state.last_uploaded_file = uploaded_file.name
            
            # Formulario de registro
            with st.form("face_registration_form"):
                person_name = st.text_input("Person's name", key="person_name")
                
                # Selector de modelo
                model_choice = st.selectbox(
                    "Embedding model",
                    ["VGG-Face", "Facenet", "OpenFace", "ArcFace"],
                    index=0
                )
                
                # Ajuste de umbral de confianza
                confidence_threshold = st.slider(
                    "Detection Confidence",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.5,
                    step=0.01
                )
                
                # Opción para añadir a persona existente
                add_to_existing = st.checkbox(
                    "Add to existing person"
                )
                
                # Botón de registro
                register_button = st.form_submit_button("Register Face")
            
            if register_button:
                # Validar que se haya proporcionado un nombre
                if not person_name:
                    st.error("Person's name is required. Please enter a name.")
                elif uploaded_file is None:
                    st.error("Please upload an image.")
                else:
                    # Mostrar spinner durante el procesamiento
                    with st.spinner('Processing image and extracting facial features...'):
                        # Process imagen
                        raw_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                        image = cv2.imdecode(raw_bytes, cv2.IMREAD_COLOR)
                        
                        # Detect rostros
                        face_net = load_face_model()
                        detections = detect_face_dnn(face_net, image, conf_threshold=confidence_threshold)
                        
                        # Procesar detecciones y obtener bounding boxes
                        processed_image, bboxes = process_face_detections(image, detections, confidence_threshold)
                        
                        if not bboxes:
                            st.error("No faces detected in the image. Please upload another image.")
                        elif len(bboxes) > 1:
                            st.warning("Multiple faces detected. The first one will be used.")
                            
                            # Extraer embeddings del primer rostro
                            if bboxes and len(bboxes) > 0 and len(bboxes[0]) == 5:
                                embeddings_all_models = extract_face_embeddings_all_models(image, bboxes[0])
                                
                                if embeddings_all_models:
                                    # Guardar la imagen del rostro para referencia
                                    x1, y1, x2, y2, _ = bboxes[0]
                                    # Validar coordenadas
                                    x1, y1 = max(0, x1), max(0, y1)
                                    x2, y2 = min(image.shape[1], x2), min(image.shape[0], y2)
                                    
                                    if x2 > x1 and y2 > y1:
                                        face_crop = image[y1:y2, x1:x2].copy()
                                        # Asegurar un tamaño mínimo para el rostro
                                        if face_crop.size > 0:
                                            min_size = 64
                                            face_h, face_w = face_crop.shape[:2]
                                            if face_h < min_size or face_w < min_size:
                                                scale = max(min_size/face_h, min_size/face_w)
                                                face_crop = cv2.resize(face_crop, 
                                                                     (max(min_size, int(face_w * scale)), 
                                                                      max(min_size, int(face_h * scale))))
                                    else:
                                        st.error("Invalid face region detected. Please try again with a clearer image.")
                                        return
                                    
                                    # Guardar en la base de datos
                                    if add_to_existing and person_name in st.session_state.face_database:
                                        # Añadir a persona existente
                                        if 'embeddings' in st.session_state.face_database[person_name]:
                                            # Formato nuevo con múltiples embeddings
                                            for embedding in embeddings_all_models:
                                                model_name = embedding['model']
                                                model_idx = -1
                                                
                                                # Buscar si ya existe un embedding de este modelo
                                                for i, model in enumerate(st.session_state.face_database[person_name]['models']):
                                                    if model == model_name:
                                                        model_idx = i
                                                        break
                                                
                                                if model_idx >= 0:
                                                    # Actualizar embedding existente
                                                    st.session_state.face_database[person_name]['embeddings'][model_idx] = embedding['embedding']
                                                else:
                                                    # Añadir nuevo modelo
                                                    st.session_state.face_database[person_name]['models'].append(model_name)
                                                    st.session_state.face_database[person_name]['embeddings'].append(embedding['embedding'])
                                            
                                            # Actualizar imagen de referencia
                                            st.session_state.face_database[person_name]['face_image'] = face_crop
                                        
                                        # Incrementar contador
                                        st.session_state.face_database[person_name]['count'] += 1
                                    else:
                                        # Crear nueva entrada en la base de datos
                                        st.sidebar.write(f"Creating new entry for {person_name}")
                                        
                                        models = []
                                        embeddings = []
                                        
                                        for embedding in embeddings_all_models:
                                            models.append(embedding['model'])
                                            embeddings.append(embedding['embedding'])
                                        
                                        st.session_state.face_database[person_name] = {
                                            'embeddings': embeddings,
                                            'models': models,
                                            'count': 1,
                                            'face_image': face_crop
                                        }
                                    
                                    st.success(f"Face registered successfully for {person_name}!")
                                    
                                    # Guardar la base de datos actualizada
                                    if DATABASE_UTILS_AVAILABLE:
                                        save_success = save_face_database(st.session_state.face_database)
                                        if save_success:
                                            st.info("Face database saved successfully!")
                                            # Mostrar información actualizada de la base de datos
                                            print_database_info()
                                        else:
                                            st.error("Error saving face database!")
                                    
                                    # Mostrar la imagen con el rostro detectado
                                    processed_image, _ = process_face_detections(image, [bboxes[0]], confidence_threshold)
                                    st.image(cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB), caption=f"Registered face: {person_name}")
                                    
                                    # Forzar recarga de la interfaz para mostrar el rostro registrado
                                    st.rerun()
                                else:
                                    st.error("Failed to extract embeddings. Please try again with a clearer image.")
                        else:
                            # Solo un rostro detectado
                            embeddings_all_models = extract_face_embeddings_all_models(image, bboxes[0])
                            
                            if embeddings_all_models:
                                # Extraer la región del rostro para guardarla
                                x1, y1, x2, y2, _ = bboxes[0]
                                # Validar coordenadas
                                x1, y1 = max(0, x1), max(0, y1)
                                x2, y2 = min(image.shape[1], x2), min(image.shape[0], y2)
                                
                                if x2 > x1 and y2 > y1:
                                    face_crop = image[y1:y2, x1:x2].copy()
                                    # Asegurar un tamaño mínimo para el rostro
                                    if face_crop.size > 0:
                                        min_size = 64
                                        face_h, face_w = face_crop.shape[:2]
                                        if face_h < min_size or face_w < min_size:
                                            scale = max(min_size/face_h, min_size/face_w)
                                            face_crop = cv2.resize(face_crop, 
                                                                 (max(min_size, int(face_w * scale)), 
                                                                  max(min_size, int(face_h * scale))))
                                    else:
                                        st.error("Invalid face region detected. Please try again with a clearer image.")
                                        return
                                    
                                    # Guardar en la base de datos
                                    if add_to_existing and person_name in st.session_state.face_database:
                                        # Añadir a persona existente
                                        if 'embeddings' in st.session_state.face_database[person_name]:
                                            # Formato nuevo con múltiples embeddings
                                            for embedding in embeddings_all_models:
                                                model_name = embedding['model']
                                                model_idx = -1
                                                
                                                # Buscar si ya existe un embedding de este modelo
                                                for i, model in enumerate(st.session_state.face_database[person_name]['models']):
                                                    if model == model_name:
                                                        model_idx = i
                                                        break
                                                
                                                if model_idx >= 0:
                                                    # Actualizar embedding existente
                                                    st.session_state.face_database[person_name]['embeddings'][model_idx] = embedding['embedding']
                                                else:
                                                    # Añadir nuevo modelo
                                                    st.session_state.face_database[person_name]['models'].append(model_name)
                                                    st.session_state.face_database[person_name]['embeddings'].append(embedding['embedding'])
                                            
                                            # Actualizar imagen de referencia
                                            st.session_state.face_database[person_name]['face_image'] = face_crop
                                        
                                        # Incrementar contador
                                        st.session_state.face_database[person_name]['count'] += 1
                                    else:
                                        # Crear nueva entrada en la base de datos
                                        st.sidebar.write(f"Creating new entry for {person_name}")
                                        
                                        models = []
                                        embeddings = []
                                        
                                        for embedding in embeddings_all_models:
                                            models.append(embedding['model'])
                                            embeddings.append(embedding['embedding'])
                                        
                                        st.session_state.face_database[person_name] = {
                                            'embeddings': embeddings,
                                            'models': models,
                                            'count': 1,
                                            'face_image': face_crop
                                        }
                                    
                                    st.success(f"Face registered successfully for {person_name}!")
                                
                                # Guardar la base de datos actualizada
                                if DATABASE_UTILS_AVAILABLE:
                                    save_success = save_face_database(st.session_state.face_database)
                                    if save_success:
                                        st.info("Face database saved successfully!")
                                        # Mostrar información actualizada de la base de datos
                                        print_database_info()
                                    else:
                                        st.error("Error saving face database!")
                                
                                # Mostrar la imagen con el rostro detectado
                                processed_image, _ = process_face_detections(image, [bboxes[0]], confidence_threshold)
                                st.image(cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB), caption=f"Registered face: {person_name}")
                                
                                # Forzar recarga de la interfaz para mostrar el rostro registrado
                                st.rerun()
                            else:
                                st.error("Failed to extract embeddings. Please try again with a clearer image.")
            
            # Mostrar tabla de rostros registrados
            st.subheader("Registered Faces")
            
            # Debug de contenido de base de datos
            st.sidebar.write(f"Face database contains {len(st.session_state.face_database)} entries at display time")
            
            if 'face_database' in st.session_state and st.session_state.face_database:
                # Inicializar variables para la tabla
                data = []
                
                # Preparar datos para la tabla
                for name, info in st.session_state.face_database.items():
                    # Determinar el número de embeddings
                    if 'embeddings' in info:
                        num_embeddings = len(info['embeddings'])
                        models = ', '.join(info['models'])
                    else:
                        num_embeddings = 1
                        models = 'VGG-Face'  # Modelo por defecto para formato antiguo
                    
                    # Determinar el número de imágenes
                    num_images = info.get('count', 1)
                    
                    # Añadir a los datos
                    data.append({
                        "Name": name,
                        "Images": num_images,
                        "Embeddings": num_embeddings,
                        "Models": models,
                        "Face": info.get('face_image', None)
                    })
                
                # Debug de los datos procesados
                st.sidebar.write(f"Processed {len(data)} entries for display")
                
                # Verificar si hay datos para mostrar
                if data:
                    # Crear cabeceras de la tabla
                    col_thumb, col1, col2, col3, col4, col5 = st.columns([2, 3, 2, 2, 4, 2])
                    
                    with col_thumb:
                        st.write("**Thumbnail**")
                    with col1:
                        st.write("**Name**")
                    with col2:
                        st.write("**Images**")
                    with col3:
                        st.write("**Embeddings**")
                    with col4:
                        st.write("**Models**")
                    with col5:
                        st.write("**Actions**")
                    
                    # Mostrar tabla con botones de eliminación
                    for i, row in enumerate(data):
                        col_thumb, col1, col2, col3, col4, col5 = st.columns([2, 3, 2, 2, 4, 2])
                        
                        # Mostrar miniatura si está disponible
                        with col_thumb:
                            if row["Face"] is not None and row["Face"].size > 0:
                                try:
                                    # Redimensionar para crear miniatura
                                    face_img = row["Face"]
                                    h, w = face_img.shape[:2]
                                    if h > 0 and w > 0:
                                        thumbnail = cv2.resize(face_img, (max(1, w//4), max(1, h//4)))
                                        st.image(cv2.cvtColor(thumbnail, cv2.COLOR_BGR2RGB), width=50)
                                    else:
                                        st.write("Invalid image")
                                except Exception as e:
                                    st.write("Error displaying image")
                                    st.error(f"Error: {str(e)}")
                            else:
                                st.write("No image")
                                
                        with col1:
                            st.write(row["Name"])
                        with col2:
                            st.write(row["Images"])
                        with col3:
                            st.write(row["Embeddings"])
                        with col4:
                            st.write(row["Models"])
                        with col5:
                            if st.button("Delete", key=f"delete_{row['Name']}"):
                                # Eliminar el registro
                                if row["Name"] in st.session_state.face_database:
                                    del st.session_state.face_database[row["Name"]]
                                    
                                    # Guardar la base de datos actualizada
                                    if DATABASE_UTILS_AVAILABLE:
                                        save_face_database(st.session_state.face_database)
                                    
                                    st.success(f"Deleted {row['Name']} from the database.")
                                    st.rerun()
                    
                    # Botón para eliminar todos los registros
                    if st.button("Delete All Registered Faces"):
                        # Mostrar confirmación
                        if 'confirm_delete_all' not in st.session_state:
                            st.session_state.confirm_delete_all = False
                        
                        if not st.session_state.confirm_delete_all:
                            st.warning("Are you sure you want to delete all registered faces? This action cannot be undone.")
                            col1, col2 = st.columns(2)
                            with col1:
                                if st.button("Yes, delete all"):
                                    st.session_state.face_database = {}
                                    
                                    # Guardar la base de datos vacía
                                    if DATABASE_UTILS_AVAILABLE:
                                        save_face_database({})
                                    
                                    st.session_state.confirm_delete_all = False
                                    st.success("All registered faces have been deleted.")
                                    st.rerun()
                            with col2:
                                if st.button("Cancel"):
                                    st.session_state.confirm_delete_all = False
                                    st.rerun()
                else:
                    st.info("No faces registered yet. Use the form above to register faces.")
            else:
                st.info("No faces registered yet. Use the form above to register faces.")
            
            # Añadir botones para importar/exportar la base de datos
            if DATABASE_UTILS_AVAILABLE:
                st.subheader("Database Management")
                col1, col2 = st.columns(2)
                
                with col1:
                    # Exportar base de datos
                    if st.button("Export Face Database") and st.session_state.face_database:
                        export_file = export_database_json()
                        if export_file:
                            with open(export_file, "rb") as f:
                                st.download_button(
                                    label="Download JSON Database",
                                    data=f,
                                    file_name="face_database.json",
                                    mime="application/json"
                                )
                
                with col2:
                    # Importar base de datos
                    uploaded_json = st.file_uploader("Import Face Database", type=["json"], key="import_database")
                    if uploaded_json is not None:
                        if st.button("Process Import"):
                            with st.spinner("Importing database..."):
                                imported_db = import_database_json(uploaded_json)
                                if imported_db:
                                    # Actualizar la base de datos actual
                                    st.session_state.face_database.update(imported_db)
                                    
                                    # Guardar la base de datos actualizada
                                    if save_face_database(st.session_state.face_database):
                                        st.success("Database imported and saved successfully!")
                                        st.rerun()
        
        with tab2:
            st.header("Image Recognition")
            
            # Verificar si hay rostros registrados
            if not st.session_state.face_database:
                st.warning("No faces registered. Please register at least one face first.")
            else:
                # Subir imagen para reconocimiento
                uploaded_file = st.file_uploader("Subir imagen para reconocimiento", type=['jpg', 'jpeg', 'png'], key="recognition_image")
                
                # Configuración avanzada
                with st.expander("Configuración avanzada", expanded=False):
                    # Configuración de umbral de similitud
                    similarity_threshold = st.slider(
                        "Similarity threshold (%)", 
                        min_value=35.0, 
                        max_value=95.0, 
                        value=45.0, 
                        step=5.0,
                        help="Porcentaje mínimo de similitud para considerar una coincidencia"
                    )
                    
                    confidence_threshold = st.slider(
                        "Detection Confidence", 
                        min_value=0.3, 
                        max_value=0.9, 
                        value=0.5, 
                        step=0.05,
                        help="Un valor más alto es más restrictivo pero más preciso"
                    )
                    
                    model_choice = st.selectbox(
                        "Embedding model", 
                        ["VGG-Face", "Facenet", "OpenFace", "ArcFace"],
                        help="Diferentes modelos pueden dar resultados distintos según las características faciales"
                    )
                    
                    voting_method = st.radio(
                        "Método de votación para múltiples embeddings",
                        ["Promedio", "Mejor coincidencia", "Votación ponderada"],
                        help="Cómo combinar resultados cuando hay múltiples imágenes de una persona"
                    )
                    
                    show_all_matches = st.checkbox(
                        "Mostrar todas las coincidencias", 
                        value=False,
                        help="Mostrar las 3 mejores coincidencias para cada rostro"
                    )
                
                if uploaded_file is not None:
                    # Mostrar spinner durante el procesamiento
                    with st.spinner('Processing image and analyzing faces...'):
                        # Process la imagen subida
                        raw_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                        image = cv2.imdecode(raw_bytes, cv2.IMREAD_COLOR)
                        
                        # Detect rostros
                        detections = detect_face_dnn(face_net, image, confidence_threshold)
                        processed_image, bboxes = process_face_detections(image, detections, confidence_threshold)
                        
                        if not bboxes:
                            st.error("No se detectaron rostros en la imagen.")
                            # Inicializar result_image aunque no haya rostros
                            result_image = image.copy()
                        else:
                            # Mostrar imagen con rostros detectados
                            st.image(processed_image, channels='BGR', caption="Faces detected")
                            
                            # Reconocer cada rostro
                            result_image = image.copy()
                            
                            # Crear columnas para mostrar estadísticas
                            stats_cols = st.columns(len(bboxes) if len(bboxes) <= 3 else 3)
                            
                            for i, bbox in enumerate(bboxes):
                                # Extraer embedding del rostro
                                embedding = extract_face_embeddings(image, bbox, model_name=model_choice)
                                
                                if embedding is not None:
                                    # Compare con rostros registrados
                                    matches = []
                                    
                                    for name, info in st.session_state.face_database.items():
                                        if 'embeddings' in info:
                                            # Nuevo formato con múltiples embeddings
                                            similarities = []
                                            
                                            for idx, registered_embedding in enumerate(info['embeddings']):
                                                # Usar el mismo modelo si es posible
                                                if info['models'][idx] == model_choice:
                                                    weight = 1.0  # Dar más peso a embeddings del mismo modelo
                                                else:
                                                    weight = 0.8  # Peso menor para embeddings de otros modelos
                                                    
                                                # Asegurarse de que los embeddings sean compatibles
                                                try:
                                                    similarity = cosine_similarity([embedding["embedding"]], [registered_embedding])[0][0] * 100 * weight
                                                    similarities.append(similarity)
                                                except ValueError as e:
                                                    # Si hay error de dimensiones incompatibles, omitir esta comparación
                                                    # Modelos incompatibles: {info['models'][idx]} vs {embedding['model']}
                                                    continue
                                            
                                            # Aplicar método de votación seleccionado
                                            if voting_method == "Promedio":
                                                if similarities:  # Verificar que la lista no esté vacía
                                                    final_similarity = sum(similarities) / len(similarities)
                                                else:
                                                    final_similarity = 0.0  # Valor predeterminado si no hay similitudes
                                            elif voting_method == "Mejor coincidencia":
                                                if similarities:  # Verificar que la lista no esté vacía
                                                    final_similarity = max(similarities)
                                                else:
                                                    final_similarity = 0.0  # Valor predeterminado si no hay similitudes
                                            else:  # Votación ponderada
                                                if similarities:  # Verificar que la lista no esté vacía
                                                    # Dar más peso a similitudes más altas
                                                    weighted_sum = sum(s * (i+1) for i, s in enumerate(sorted(similarities)))
                                                    weights_sum = sum(i+1 for i in range(len(similarities)))
                                                    final_similarity = weighted_sum / weights_sum
                                                else:
                                                    final_similarity = 0.0  # Valor predeterminado si no hay similitudes
                                            
                                            matches.append({"name": name, "similarity": final_similarity, "count": info['count']})
                                        else:
                                            # Formato antiguo con un solo embedding
                                            registered_embedding = info['embedding']
                                            try:
                                                similarity = cosine_similarity([embedding["embedding"]], [registered_embedding])[0][0] * 100
                                                matches.append({"name": name, "similarity": similarity, "count": 1})
                                            except ValueError as e:
                                                # Si hay error de dimensiones incompatibles, omitir esta comparación
                                                # Modelos incompatibles: {embedding['model']} vs formato antiguo
                                                continue
                                
                                # Ordenar coincidencias por similitud
                                matches.sort(key=lambda x: x["similarity"], reverse=True)
                                
                                # Dibujar resultado en la imagen
                                x1, y1, x2, y2, _ = bbox
                                
                                if matches and matches[0]["similarity"] >= similarity_threshold:
                                    # Coincidencia encontrada
                                    best_match = matches[0]
                                    
                                    # Color basado en nivel de similitud
                                    if best_match["similarity"] >= 80:
                                        color = (0, 255, 0)  # Verde para alta similitud
                                    elif best_match["similarity"] >= 65:
                                        color = (0, 255, 255)  # Amarillo para media similitud
                                    else:
                                        color = (0, 165, 255)  # Naranja para baja similitud
                                    
                                    # Dibujar rectángulo y etiqueta principal
                                    label = f"{best_match['name']}: {best_match['similarity']:.1f}%"
                                    cv2.rectangle(result_image, (x1, y1), (x2, y2), color, 2)
                                    cv2.putText(result_image, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                                    
                                    # Mostrar coincidencias adicionales si está activado
                                    if show_all_matches and len(matches) > 1:
                                        for j, match in enumerate(matches[1:3]):  # Mostrar las siguientes 2 mejores coincidencias
                                            sub_label = f"#{j+2}: {match['name']}: {match['similarity']:.1f}%"
                                            cv2.putText(result_image, sub_label, (x1, y1-(j+2)*20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
                                    
                                    # Mostrar estadísticas en columnas
                                    col_idx = i % 3
                                    with stats_cols[col_idx]:
                                        st.metric(
                                            f"Rostro {i+1}", 
                                            f"{best_match['name']}",
                                            f"{best_match['similarity']:.1f}%"
                                        )
                                        
                                        # Guardar información para mostrar la imagen de referencia después
                                        if 'matched_faces' not in st.session_state:
                                            st.session_state.matched_faces = []
                                        
                                        # Extraer la región del rostro para mostrarla
                                        face_crop = image[y1:y2, x1:x2].copy()
                                        
                                        # Guardar información de la coincidencia
                                        st.session_state.matched_faces.append({
                                            "face_crop": face_crop,
                                            "matched_name": best_match['name'],
                                            "similarity": best_match['similarity'],
                                            "bbox": (x1, y1, x2, y2)
                                        })
                                        
                                        if show_all_matches and len(matches) > 1:
                                            st.write("Otras coincidencias:")
                                            for j, match in enumerate(matches[1:3]):
                                                st.write(f"- {match['name']}: {match['similarity']:.1f}%")
                                else:
                                    # No hay coincidencia
                                    label = "Desconocido"
                                    if matches:
                                        label += f": {matches[0]['similarity']:.1f}%"
                                    
                                    cv2.rectangle(result_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
                                    cv2.putText(result_image, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                                    
                                    # Mostrar estadísticas en columnas
                                    col_idx = i % 3
                                    with stats_cols[col_idx]:
                                        st.metric(
                                            f"Rostro {i+1}", 
                                            "Desconocido",
                                            f"{matches[0]['similarity']:.1f}%" if matches else "N/A"
                                        )
                
                # Mostrar resultado solo si hay una imagen cargada
                if uploaded_file is not None:
                    st.subheader("Recognition Result")
                    st.image(result_image, channels='BGR', use_container_width=True)
                    
                    # Mostrar comparación lado a lado de cada rostro con su coincidencia
                    if 'matched_faces' in st.session_state and st.session_state.matched_faces:
                        st.subheader("Face Comparison")
                        st.write("Below you can see each detected face alongside its match in the database:")
                        
                        for idx, match_info in enumerate(st.session_state.matched_faces):
                            # Crear contenedor para la comparación
                            comparison_container = st.container()
                            
                            # Crear columnas dentro del contenedor
                            with comparison_container:
                                comp_col1, comp_col2 = st.columns(2)
                                
                                # Mostrar el rostro detectado
                                with comp_col1:
                                    st.write(f"**Detected Face #{idx+1}**")
                                    st.image(
                                        cv2.cvtColor(match_info["face_crop"], cv2.COLOR_BGR2RGB),
                                        width=250  # Usar ancho fijo en lugar de use_column_width
                                    )
                                
                                # Mostrar imagen de referencia si existe
                                with comp_col2:
                                    reference_name = match_info["matched_name"]
                                    st.write(f"**Match: {reference_name}** ({match_info['similarity']:.1f}%)")
                                    
                                    # Intentar mostrar la imagen de referencia guardada
                                    if reference_name in st.session_state.face_database and 'face_image' in st.session_state.face_database[reference_name]:
                                        reference_image = st.session_state.face_database[reference_name]['face_image']
                                        st.image(
                                            cv2.cvtColor(reference_image, cv2.COLOR_BGR2RGB),
                                            width=250  # Usar ancho fijo en lugar de use_column_width
                                        )
                                    else:
                                        # Mensaje de error simplificado
                                        st.info(f"No reference image available for {reference_name}. Please re-register this person.")
                          
                          # Limpiar el estado para la próxima ejecución
                        del st.session_state.matched_faces
        
        with tab3:
            st.header("Real-time Recognition")
            
            # Verificar si hay rostros registrados
            if not st.session_state.face_database:
                st.warning("No faces registered. Please register at least one face first.")
            else:
                # Configuración avanzada
                with st.expander("Configuración avanzada", expanded=False):
                    # Configuración de umbral de similitud
                    similarity_threshold = st.slider(
                        "Similarity threshold (%)", 
                        min_value=35.0, 
                        max_value=95.0, 
                        value=45.0, 
                        step=5.0,
                        key="realtime_threshold",
                        help="Porcentaje mínimo de similitud para considerar una coincidencia"
                    )
                    
                    confidence_threshold = st.slider(
                        "Detection Confidence", 
                        min_value=0.3, 
                        max_value=0.9, 
                        value=0.5, 
                        step=0.05,
                        key="realtime_confidence",
                        help="Un valor más alto es más restrictivo pero más preciso"
                    )
                    
                    model_choice = st.selectbox(
                        "Embedding model", 
                        ["VGG-Face", "Facenet", "OpenFace", "ArcFace"],
                        key="realtime_model",
                        help="Diferentes modelos pueden dar resultados distintos según las características faciales"
                    )
                    
                    voting_method = st.radio(
                        "Método de votación para múltiples embeddings",
                        ["Promedio", "Mejor coincidencia", "Votación ponderada"],
                        key="realtime_voting",
                        help="Cómo combinar resultados cuando hay múltiples imágenes de una persona"
                    )
                    
                    show_confidence = st.checkbox(
                        "Mostrar porcentaje de confianza", 
                        value=True,
                        help="Mostrar el porcentaje de similitud junto al nombre"
                    )
                    
                    stabilize_results = st.checkbox(
                        "Estabilizar resultados", 
                        value=True,
                        help="Reduce fluctuaciones en la identificación usando un promedio temporal"
                    )
                    
                    fps_limit = st.slider(
                        "Límite de FPS", 
                        min_value=5, 
                        max_value=30, 
                        value=15, 
                        step=1,
                        help="Limitar los frames por segundo para reducir uso de CPU"
                    )
                
                # Inicializar estado de la cámara
                if 'recognition_camera_running' not in st.session_state:
                    st.session_state.recognition_camera_running = False
                    
                # Inicializar historial de reconocimiento para estabilización
                if 'recognition_history' not in st.session_state:
                    st.session_state.recognition_history = {}
                
                # Botones para controlar la cámara
                col1, col2 = st.columns(2)
                start_button = col1.button("Iniciar Cámara", key="start_recognition_camera", 
                                          on_click=lambda: setattr(st.session_state, 'recognition_camera_running', True))
                stop_button = col2.button("Detener Cámara", key="stop_recognition_camera", 
                                         on_click=lambda: setattr(st.session_state, 'recognition_camera_running', False))
                
                # Placeholder para el video
                video_placeholder = st.empty()
                
                # Placeholder para métricas
                metrics_cols = st.columns(3)
                with metrics_cols[0]:
                    faces_metric = st.empty()
                with metrics_cols[1]:
                    fps_metric = st.empty()
                with metrics_cols[2]:
                    time_metric = st.empty()
                
                if st.session_state.recognition_camera_running:
                    st.info("Cámara activada. Procesando video en tiempo real...")
                    
                    # Inicializar webcam
                    cap = cv2.VideoCapture(0)
                    
                    if not cap.isOpened():
                        st.error("No se pudo acceder a la cámara. Asegúrese de que esté conectada y no esté siendo utilizada por otra aplicación.")
                        st.session_state.recognition_camera_running = False
                    else:
                        try:
                            # Variables para métricas
                            frame_count = 0
                            start_time = time.time()
                            last_frame_time = start_time
                            fps_history = []
                            
                            while st.session_state.recognition_camera_running:
                                # Control de FPS
                                current_time = time.time()
                                elapsed = current_time - last_frame_time
                                if elapsed < 1.0/fps_limit:
                                    time.sleep(0.01)  # Pequeña pausa para no sobrecargar la CPU
                                    continue
                                    
                                last_frame_time = current_time
                                
                                # Leer frame
                                ret, frame = cap.read()
                                if not ret:
                                    st.error("Error al leer frame de la cámara.")
                                    break
                                
                                # Actualizar contador de frames
                                frame_count += 1
                                
                                # Calcular FPS
                                if frame_count % 5 == 0:
                                    fps = 5 / (current_time - start_time)
                                    fps_history.append(fps)
                                    if len(fps_history) > 10:
                                        fps_history.pop(0)
                                    avg_fps = sum(fps_history) / len(fps_history)
                                    start_time = current_time
                                    
                                    # Actualizar métricas
                                    fps_metric.metric("FPS", f"{avg_fps:.1f}")
                                    time_metric.metric("Tiempo activo", f"{int(current_time - time.time() + st.session_state.get('camera_start_time', current_time))}s")
                                
                                # Detect rostros
                                detections = detect_face_dnn(face_net, frame, confidence_threshold)
                                _, bboxes = process_face_detections(frame, detections, confidence_threshold)
                                
                                # Actualizar métrica de rostros
                                if frame_count % 5 == 0:
                                    faces_metric.metric("Faces detected", len(bboxes))
                                
                                # Reconocer cada rostro
                                result_frame = frame.copy()
                                
                                for i, bbox in enumerate(bboxes):
                                    face_id = f"face_{i}"
                                    
                                    # Extraer embedding del rostro
                                    embedding = extract_face_embeddings(frame, bbox, model_name=model_choice)
                                    
                                    if embedding is not None:
                                        # Compare con rostros registrados
                                        matches = []
                                        
                                        for name, info in st.session_state.face_database.items():
                                            if 'embeddings' in info:
                                                # Nuevo formato con múltiples embeddings
                                                similarities = []
                                                
                                                for idx, registered_embedding in enumerate(info['embeddings']):
                                                    # Usar el mismo modelo si es posible
                                                    if info['models'][idx] == model_choice:
                                                        weight = 1.0  # Dar más peso a embeddings del mismo modelo
                                                    else:
                                                        weight = 0.8  # Peso menor para embeddings de otros modelos
                                                        
                                                    # Asegurarse de que los embeddings sean compatibles
                                                    try:
                                                        similarity = cosine_similarity([embedding["embedding"]], [registered_embedding])[0][0] * 100 * weight
                                                        similarities.append(similarity)
                                                    except ValueError as e:
                                                        # Si hay error de dimensiones incompatibles, omitir esta comparación
                                                        continue
                                                
                                                # Aplicar método de votación seleccionado
                                                if voting_method == "Promedio":
                                                    final_similarity = sum(similarities) / len(similarities)
                                                elif voting_method == "Mejor coincidencia":
                                                    final_similarity = max(similarities)
                                                else:  # Votación ponderada
                                                    # Dar más peso a similitudes más altas
                                                    weighted_sum = sum(s * (i+1) for i, s in enumerate(sorted(similarities)))
                                                    weights_sum = sum(i+1 for i in range(len(similarities)))
                                                    final_similarity = weighted_sum / weights_sum
                                                
                                                matches.append({"name": name, "similarity": final_similarity})
                                            else:
                                                # Formato antiguo con un solo embedding
                                                registered_embedding = info['embedding']
                                                try:
                                                    similarity = cosine_similarity([embedding["embedding"]], [registered_embedding])[0][0] * 100
                                                    matches.append({"name": name, "similarity": similarity})
                                                except ValueError as e:
                                                    # Si hay error de dimensiones incompatibles, omitir esta comparación
                                                    # Modelos incompatibles: {embedding['model']} vs formato antiguo
                                                    continue
                                        
                                        # Ordenar coincidencias por similitud
                                        matches.sort(key=lambda x: x["similarity"], reverse=True)
                                        
                                        # Estabilizar resultados si está activado
                                        if stabilize_results and matches:
                                            best_match = matches[0]
                                            
                                            # Inicializar historial para este rostro si no existe
                                            if face_id not in st.session_state.recognition_history:
                                                st.session_state.recognition_history[face_id] = {
                                                    "names": [],
                                                    "similarities": []
                                                }
                                            
                                            # Añadir al historial
                                            history = st.session_state.recognition_history[face_id]
                                            history["names"].append(best_match["name"])
                                            history["similarities"].append(best_match["similarity"])
                                            
                                            # Limitar historial a los últimos 10 frames
                                            if len(history["names"]) > 10:
                                                history["names"].pop(0)
                                                history["similarities"].pop(0)
                                            
                                            # Determinar el nombre más frecuente en el historial
                                            if len(history["names"]) >= 3:  # Necesitamos al menos 3 frames para estabilizar
                                                name_counts = {}
                                                for name in history["names"]:
                                                    if name not in name_counts:
                                                        name_counts[name] = 0
                                                    name_counts[name] += 1
                                                
                                                # Encontrar el nombre más frecuente
                                                stable_name = max(name_counts.items(), key=lambda x: x[1])[0]
                                                
                                                # Calcular similitud promedio para ese nombre
                                                stable_similarities = [
                                                    history["similarities"][i] 
                                                    for i in range(len(history["names"])) 
                                                    if history["names"][i] == stable_name
                                                ]
                                                stable_similarity = sum(stable_similarities) / len(stable_similarities)
                                                
                                                # Reemplazar la mejor coincidencia con el resultado estabilizado
                                                best_match = {"name": stable_name, "similarity": stable_similarity}
                                            else:
                                                best_match = matches[0]
                                        else:
                                            best_match = matches[0] if matches else None
                                        
                                        # Dibujar resultado en la imagen
                                        x1, y1, x2, y2, _ = bbox
                                        
                                        if best_match and best_match["similarity"] >= similarity_threshold:
                                            # Coincidencia encontrada
                                            # Color basado en nivel de similitud
                                            if best_match["similarity"] >= 80:
                                                color = (0, 255, 0)  # Verde para alta similitud
                                            elif best_match["similarity"] >= 65:
                                                color = (0, 255, 255)  # Amarillo para media similitud
                                            else:
                                                color = (0, 165, 255)  # Naranja para baja similitud
                                            
                                            # Dibujar rectángulo y etiqueta
                                            cv2.rectangle(result_frame, (x1, y1), (x2, y2), color, 2)
                                            
                                            if show_confidence:
                                                label = f"{best_match['name']}: {best_match['similarity']:.1f}%"
                                            else:
                                                label = f"{best_match['name']}"
                                                
                                            cv2.putText(result_frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                                        else:
                                            # No hay coincidencia
                                            cv2.rectangle(result_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                                            
                                            if best_match:
                                                label = f"Desconocido: {best_match['similarity']:.1f}%" if show_confidence else "Desconocido"
                                            else:
                                                label = "Desconocido"
                                                
                                            cv2.putText(result_frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                            
                            # Mostrar resultado
                            video_placeholder.image(result_frame, channels="BGR", use_container_width=True)
                        finally:
                            # Liberar la cámara cuando se detenga
                            cap.release()
                            # Limpiar historial de reconocimiento
                            st.session_state.recognition_history = {}
                else:
                    st.info("Haga clic en 'Iniciar Cámara' para comenzar el reconocimiento en tiempo real.")

# Si se ejecuta este archivo directamente, llamar a la función main
if __name__ == "__main__":
    main()








