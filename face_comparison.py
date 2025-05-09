"""
Face Comparison Module

This module provides functions for comparing faces between images using both
traditional computer vision techniques (HOG) and deep learning models (DeepFace).
It includes functions for generating similarity metrics, visual comparison
reports, and extracting facial embeddings.

The module supports fallback mechanisms to ensure robust face comparison
even when certain models or libraries are unavailable.
"""
import cv2
import numpy as np
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity

def compare_faces(image1, bboxes1, image2, bboxes2):
    """
    Compare faces using Histogram of Oriented Gradients (HOG) features.
    
    This method provides a baseline comparison using traditional computer
    vision techniques without requiring deep learning models.
    
    Args:
        image1 (numpy.ndarray): First image containing faces
        bboxes1 (list): List of bounding boxes for faces in the first image
        image2 (numpy.ndarray): Second image containing faces
        bboxes2 (list): List of bounding boxes for faces in the second image
        
    Returns:
        list: Nested list of comparison results with similarity scores
    """
    # Convert images to grayscale
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    
    # Initialize list to store comparison results
    comparison_results = []
    
    # Calculate HOG parameters based on face size
    win_size = (64, 64)
    block_size = (16, 16)
    block_stride = (8, 8)
    cell_size = (8, 8)
    nbins = 9
    
    # Iterate over each face in the first image
    for bbox1 in bboxes1:
        x1_1, y1_1, x2_1, y2_1, _ = bbox1
        
        # Check if the face region is valid
        if x1_1 >= x2_1 or y1_1 >= y2_1:
            continue
            
        # Resize face to a standard size for HOG
        face1_roi = image1[y1_1:y2_1, x1_1:x2_1]
        face1_resized = cv2.resize(face1_roi, win_size)
        face1_gray = cv2.cvtColor(face1_resized, cv2.COLOR_BGR2GRAY)
        
        # Calculate HOG features
        hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins)
        h1 = hog.compute(face1_gray)
        
        # Normalize the feature vector
        h1_norm = h1 / np.linalg.norm(h1)
        
        # Store results for this face
        face_comparisons = []
        
        # Compare with each face in the second image
        for bbox2 in bboxes2:
            x1_2, y1_2, x2_2, y2_2, _ = bbox2
            
            # Check if the face region is valid
            if x1_2 >= x2_2 or y1_2 >= y2_2:
                continue
                
            # Resize face to a standard size for HOG
            face2_roi = image2[y1_2:y2_2, x1_2:x2_2]
            face2_resized = cv2.resize(face2_roi, win_size)
            face2_gray = cv2.cvtColor(face2_resized, cv2.COLOR_BGR2GRAY)
            
            # Calculate HOG features
            h2 = hog.compute(face2_gray)
            
            # Normalize the feature vector
            h2_norm = h2 / np.linalg.norm(h2)
            
            # Calculate cosine similarity
            similarity = np.dot(h1_norm.flatten(), h2_norm.flatten()) * 100
            
            # Add result
            face_comparisons.append({
                "similarity": similarity
            })
        
        comparison_results.append(face_comparisons)
    
    return comparison_results

def compare_faces_embeddings(image1, bboxes1, image2, bboxes2, model_name="VGG-Face"):
    """
    Compare faces using neural network facial embeddings from DeepFace.
    
    This provides a more advanced face comparison than HOG features by using
    deep learning models specifically trained for facial recognition.
    
    Args:
        image1 (numpy.ndarray): First image containing faces
        bboxes1 (list): List of bounding boxes for faces in the first image
        image2 (numpy.ndarray): Second image containing faces
        bboxes2 (list): List of bounding boxes for faces in the second image
        model_name (str): Name of the DeepFace model to use (default: "VGG-Face")
        
    Returns:
        list: Nested list of comparison results with similarity scores
    """
    try:
        from deepface import DeepFace
        import numpy as np
    except ImportError:
        # Fallback to HOG if DeepFace is not available
        return compare_faces(image1, bboxes1, image2, bboxes2)
    
    # Initialize list to store comparison results
    comparison_results = []
    
    # Iterate over each face in the first image
    for bbox1 in bboxes1:
        x1_1, y1_1, x2_1, y2_1, _ = bbox1
        
        # Check if the face region is valid
        if x1_1 >= x2_1 or y1_1 >= y2_1:
            continue
            
        # Extract face region
        face1_roi = image1[y1_1:y2_1, x1_1:x2_1]
        
        # Get embedding for the face
        try:
            embedding1 = DeepFace.represent(face1_roi, model_name=model_name, enforce_detection=False)[0]["embedding"]
        except Exception as e:
            st.warning(f"Error extracting embedding from face 1: {str(e)}")
            # Try with a fallback model
            try:
                embedding1 = DeepFace.represent(face1_roi, model_name="OpenFace", enforce_detection=False)[0]["embedding"]
            except:
                # If still fails, use HOG
                face_comparisons = []
                for bbox2 in bboxes2:
                    face_comparisons.append({"similarity": 0})
                comparison_results.append(face_comparisons)
                continue
        
        # Store results for this face
        face_comparisons = []
        
        # Compare with each face in the second image
        for bbox2 in bboxes2:
            x1_2, y1_2, x2_2, y2_2, _ = bbox2
            
            # Check if the face region is valid
            if x1_2 >= x2_2 or y1_2 >= y2_2:
                continue
                
            # Extract face region
            face2_roi = image2[y1_2:y2_2, x1_2:x2_2]
            
            # Get embedding for the face
            try:
                embedding2 = DeepFace.represent(face2_roi, model_name=model_name, enforce_detection=False)[0]["embedding"]
            except Exception as e:
                st.warning(f"Error extracting embedding from face 2: {str(e)}")
                # Try with a fallback model
                try:
                    embedding2 = DeepFace.represent(face2_roi, model_name="OpenFace", enforce_detection=False)[0]["embedding"]
                except:
                    # If still fails, add a 0 similarity
                    face_comparisons.append({"similarity": 0})
                    continue
            
            # Calculate cosine similarity between embeddings
            embedding1_array = np.array(embedding1).reshape(1, -1)
            embedding2_array = np.array(embedding2).reshape(1, -1)
            similarity = cosine_similarity(embedding1_array, embedding2_array)[0][0] * 100
            
            # Add result
            face_comparisons.append({
                "similarity": similarity
            })
        
        comparison_results.append(face_comparisons)
    
    return comparison_results

def generate_comparison_report_english(comparison_results, bboxes1, bboxes2, threshold=50.0):
    """
    Generate a formatted text report of face comparison results.
    
    Creates a human-readable report detailing the similarity scores between
    faces in two images, highlighting the best matches.
    
    Args:
        comparison_results (list): Results from compare_faces or compare_faces_embeddings
        bboxes1 (list): List of bounding boxes for faces in the first image
        bboxes2 (list): List of bounding boxes for faces in the second image
        threshold (float): Minimum similarity score to consider a match (default: 50.0)
        
    Returns:
        str: Formatted report text with comparison details
    """
    # Skip if no comparison results
    if not comparison_results:
        return "No face comparisons were performed."
    
    # Add header
    report = ["Face Comparison Report:"]
    
    # Add comparison results
    for i, face_comparisons in enumerate(comparison_results):
        report.append(f"\nFace {i+1} from Image 1:")
        
        # Skip if no comparisons for this face
        if not face_comparisons:
            report.append("  No comparisons available for this face.")
            continue
        
        # Find best match
        best_match_idx = max(range(len(face_comparisons)), key=lambda j: face_comparisons[j]["similarity"])
        best_match_similarity = face_comparisons[best_match_idx]["similarity"]
        
        # Add best match info
        if best_match_similarity >= threshold:
            report.append(f"  Best match: Face {best_match_idx+1} from Image 2 (Similarity: {best_match_similarity:.2f}%)")
        else:
            report.append(f"  No strong matches found. Best similarity is with Face {best_match_idx+1} ({best_match_similarity:.2f}%)")
        
        # Add all comparisons
        report.append("  All comparisons:")
        for j, comp in enumerate(face_comparisons):
            report.append(f"    Face {j+1}: Similarity {comp['similarity']:.2f}%")
    
    # Join the list into a single string with line breaks
    return "\n".join(report)

def draw_face_matches(image1, bboxes1, image2, bboxes2, comparison_results, threshold=50.0):
    """
    Create a visual representation of face matches between two images.
    
    Generates a combined image with both input images side by side and
    draws connecting lines between matching faces, with similarity scores.
    
    Args:
        image1 (numpy.ndarray): First image containing faces
        bboxes1 (list): List of bounding boxes for faces in the first image
        image2 (numpy.ndarray): Second image containing faces
        bboxes2 (list): List of bounding boxes for faces in the second image
        comparison_results (list): Results from compare_faces or compare_faces_embeddings
        threshold (float): Minimum similarity score to draw a match line (default: 50.0)
        
    Returns:
        numpy.ndarray: Combined image with visual match indicators
    """
    # Get dimensions
    h1, w1 = image1.shape[:2]
    h2, w2 = image2.shape[:2]
    
    # Create a combined image
    combined_h = max(h1, h2)
    combined_w = w1 + w2
    combined_img = np.zeros((combined_h, combined_w, 3), dtype=np.uint8)
    
    # Copy images
    combined_img[:h1, :w1] = image1
    combined_img[:h2, w1:w1+w2] = image2
    
    # Draw lines between matching faces
    for i, face_comparisons in enumerate(comparison_results):
        # Skip if no comparisons for this face
        if not face_comparisons:
            continue
            
        # Get bbox for this face
        x1_1, y1_1, x2_1, y2_1, _ = bboxes1[i]
        center1_x = (x1_1 + x2_1) // 2
        center1_y = (y1_1 + y2_1) // 2
        
        # For each comparison
        for j, comp in enumerate(face_comparisons):
            similarity = comp["similarity"]
            
            # Only draw lines for matches above threshold
            if similarity >= threshold:
                # Get bbox for the other face
                x1_2, y1_2, x2_2, y2_2, _ = bboxes2[j]
                center2_x = (x1_2 + x2_2) // 2 + w1  # Adjust for offset
                center2_y = (y1_2 + y2_2) // 2
                
                # Calculate color based on similarity (green for high, red for low)
                # Map 50-100% to color scale
                color_val = min(255, max(0, int((similarity - threshold) * 255 / (100 - threshold))))
                line_color = (0, 0, 255)  # Red for all matches
                
                # Draw line
                cv2.line(combined_img, (center1_x, center1_y), (center2_x, center2_y), line_color, 2)
                
                # Add similarity text
                text_x = (center1_x + center2_x) // 2 - 20
                text_y = (center1_y + center2_y) // 2 - 10
                cv2.putText(combined_img, f"{similarity:.1f}%", (text_x, text_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    return combined_img 

def extract_face_embeddings(image, bbox, model_name="VGG-Face"):
    """
    Extract facial embedding vector for a single face using DeepFace.
    
    These embeddings can be used for face recognition and comparison.
    
    Args:
        image (numpy.ndarray): Image containing the face
        bbox (list): Bounding box coordinates [x1, y1, x2, y2, confidence]
        model_name (str): Name of the DeepFace model to use (default: "VGG-Face")
        
    Returns:
        dict: Dictionary containing embedding vector and model information,
              or None if embedding extraction fails
    """
    try:
        from deepface import DeepFace
    except ImportError:
        st.error("DeepFace library is not available. Please install with 'pip install deepface' to use embeddings.")
        return None
    
    # Extract bbox coordinates
    x1, y1, x2, y2, _ = bbox
    
    # Check if the face region is valid
    if x1 >= x2 or y1 >= y2:
        return None
    
    # Extract face region
    face_roi = image[y1:y2, x1:x2]
    
    # Get embedding for the face
    try:
        embedding_info = DeepFace.represent(face_roi, model_name=model_name, enforce_detection=False)[0]
        return {
            "embedding": embedding_info["embedding"],
            "model": model_name
        }
    except Exception as e:
        st.warning(f"Error extracting embedding with {model_name}: {str(e)}")
        # Try with a fallback model
        try:
            fallback_model = "OpenFace"
            embedding_info = DeepFace.represent(face_roi, model_name=fallback_model, enforce_detection=False)[0]
            return {
                "embedding": embedding_info["embedding"],
                "model": fallback_model
            }
        except Exception as e:
            st.error(f"Failed to extract embeddings: {str(e)}")
            return None

def extract_face_embeddings_all_models(image, bbox):
    """
    Extract facial embeddings using multiple models for enhanced recognition.
    
    Uses several different deep learning models to extract embeddings,
    providing more robust recognition capabilities.
    
    Args:
        image (numpy.ndarray): Image containing the face
        bbox (list): Bounding box coordinates [x1, y1, x2, y2, confidence]
        
    Returns:
        list: List of embedding dictionaries from different models,
              or None if all embedding extractions fail
    """
    models = ["VGG-Face", "Facenet", "OpenFace", "ArcFace"]
    embeddings = []
    
    for model_name in models:
        embedding = extract_face_embeddings(image, bbox, model_name=model_name)
        if embedding:
            embeddings.append(embedding)
    
    return embeddings if embeddings else None 