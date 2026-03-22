"""Multi-face collapse detection skill for spatial analysis.

Detects ALL faces in frames and analyzes each face for collapse/anomalies.
Reuses existing Haar Cascade face detection code.
"""
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple


def decode_data_url_to_cv2(data_url: str) -> Optional[np.ndarray]:
    """Decode base64 data URL to OpenCV image."""
    import base64
    try:
        # Remove data URL prefix if present
        if ',' in data_url:
            header, encoded = data_url.split(',', 1)
            # Decode base64
            img_data = base64.b64decode(encoded)
            # Convert to numpy array
            nparr = np.frombuffer(img_data, np.uint8)
            # Decode image
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            return img
        return None
    except Exception:
        return None


def load_image_from_path(image_path: Path) -> np.ndarray:
    """Load image from file path."""
    img = cv2.imread(str(image_path))
    if img is None:
        raise IOError(f"Failed to load image from {image_path}")
    return img


def detect_all_faces(image: np.ndarray) -> List[Tuple[int, int, int, int]]:
    """
    Detect ALL faces in the image using Haar Cascade.
    Reuses existing face detection code from other spatial skills.
    
    Returns:
        List of (x, y, w, h) bounding boxes for all detected faces, sorted by area (largest first)
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    if len(faces) > 0:
        # Sort by area (largest first) and return all faces
        faces_sorted = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
        return [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in faces_sorted]
    return []


def analyze_face_collapse(face_roi: np.ndarray, face_bbox: Tuple[int, int, int, int]) -> Dict[str, Any]:
    """
    Analyze a single face ROI for collapse/anomalies.
    
    Args:
        face_roi: Face region image (cropped from original image)
        face_bbox: Face bounding box (x, y, w, h)
        
    Returns:
        Dict with anomaly_score, severity, anomaly_types, and details
    """
    if face_roi is None or face_roi.size == 0:
        return {
            "anomaly_score": 0.0,
            "severity": "normal",
            "anomaly_types": [],
            "details": "Empty face ROI"
        }
    
    h, w = face_roi.shape[:2]
    if h < 20 or w < 20:  # Too small to analyze
        return {
            "anomaly_score": 0.0,
            "severity": "normal",
            "anomaly_types": [],
            "details": "Face too small to analyze"
        }
    
    gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY) if len(face_roi.shape) == 3 else face_roi
    
    anomaly_types = []
    anomaly_scores = []
    
    # 1. Face structure integrity check (using facial feature detection)
    # Check if face has reasonable proportions and structure
    face_area = h * w
    aspect_ratio = w / h if h > 0 else 1.0
    
    # Normal face aspect ratio is roughly 0.7-0.9 (width/height)
    if aspect_ratio < 0.5 or aspect_ratio > 1.2:
        anomaly_types.append("face_structure_collapse")
        anomaly_scores.append(0.85)  # Severe structural issue
    elif aspect_ratio < 0.6 or aspect_ratio > 1.1:
        anomaly_types.append("facial_feature_distortion")
        anomaly_scores.append(0.70)  # Moderate structural issue
    
    # 2. Texture consistency check (similar to patch_anomaly)
    # Calculate texture features
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    noise_level = np.std(gray)
    
    # Very low sharpness or very high noise suggests texture issues
    if laplacian_var < 50 and noise_level > 30:
        anomaly_types.append("texture_inconsistency")
        anomaly_scores.append(0.75)
    elif laplacian_var < 100 or noise_level > 25:
        anomaly_types.append("texture_inconsistency")
        anomaly_scores.append(0.60)
    
    # 3. Edge coherence check (check face boundaries)
    # Use Canny edge detection to check for unnatural edges
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges > 0) / (h * w)
    
    # Very high or very low edge density might indicate issues
    if edge_density > 0.4 or edge_density < 0.05:
        anomaly_types.append("boundary_anomaly")
        anomaly_scores.append(0.65)
    
    # 4. Color consistency check (for color images)
    if len(face_roi.shape) == 3:
        # Check for color artifacts (unnatural color shifts)
        b, g, r = cv2.split(face_roi)
        color_variance = np.var([np.mean(b), np.mean(g), np.mean(r)])
        
        # Very high color variance might indicate color artifacts
        if color_variance > 1000:
            anomaly_types.append("texture_inconsistency")
            anomaly_scores.append(0.55)
    
    # Calculate overall anomaly score
    if len(anomaly_scores) > 0:
        # Use maximum anomaly score (most severe issue)
        max_anomaly_score = max(anomaly_scores)
        
        # Determine severity
        if max_anomaly_score >= 0.85:
            severity = "critical"
        elif max_anomaly_score >= 0.70:
            severity = "severe"
        elif max_anomaly_score >= 0.50:
            severity = "moderate"
        elif max_anomaly_score >= 0.30:
            severity = "mild"
        else:
            severity = "normal"
        
        # Create details string
        details_parts = []
        if "face_structure_collapse" in anomaly_types:
            details_parts.append("面部结构崩溃")
        if "facial_feature_distortion" in anomaly_types:
            details_parts.append("面部特征扭曲")
        if "texture_inconsistency" in anomaly_types:
            details_parts.append("纹理不一致")
        if "boundary_anomaly" in anomaly_types:
            details_parts.append("边界异常")
        
        details = ", ".join(details_parts) if details_parts else "检测到异常"
        
        return {
            "anomaly_score": float(max_anomaly_score),
            "severity": severity,
            "anomaly_types": anomaly_types,
            "details": details
        }
    else:
        return {
            "anomaly_score": 0.0,
            "severity": "normal",
            "anomaly_types": [],
            "details": "无明显异常"
        }


def analyze_multi_face_collapse(
    selected_frames: List[Dict[str, Any]],
    frames_dir: Path,
    frame_inputs: Optional[List[str]] = None,
    config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Analyze multi-face collapse for selected frames.
    Detects ALL faces and analyzes each face for collapse/anomalies.
    
    Args:
        selected_frames: List of selected frame dicts
        frames_dir: Directory containing frame files
        frame_inputs: Optional list of base64 data URLs
        config: Optional config dict
        
    Returns:
        Dict with frame_results and summary
    """
    frame_results = []
    
    for frame_idx, frame_data in enumerate(selected_frames):
        try:
            # Load image
            img = None
            if frame_inputs:
                if frame_idx < len(frame_inputs) and frame_inputs[frame_idx]:
                    img = decode_data_url_to_cv2(frame_inputs[frame_idx])
            
            if img is None:
                # Fallback to file path
                frame_path = frames_dir / frame_data["file"]
                if not frame_path.exists():
                    continue
                img = load_image_from_path(frame_path)
            
            if img is None:
                continue
            
            h, w = img.shape[:2]
            
            # Detect ALL faces (reusing existing code, but returning all faces)
            all_faces = detect_all_faces(img)
            
            if len(all_faces) == 0:
                frame_results.append({
                    "frame_index": frame_idx,
                    "frame_file": frame_data["file"],
                    "faces_detected": 0,
                    "face_details": []
                })
                continue
            
            # Analyze each face
            face_details = []
            for face_id, (fx, fy, fw, fh) in enumerate(all_faces):
                # Extract face ROI
                face_roi = img[fy:fy+fh, fx:fx+fw]
                
                # Analyze face collapse
                face_analysis = analyze_face_collapse(face_roi, (fx, fy, fw, fh))
                
                face_details.append({
                    "face_id": face_id,
                    "bbox": (fx, fy, fw, fh),
                    "anomaly_score": face_analysis["anomaly_score"],
                    "severity": face_analysis["severity"],
                    "anomaly_types": face_analysis["anomaly_types"],
                    "details": face_analysis["details"]
                })
            
            frame_results.append({
                "frame_index": frame_idx,
                "frame_file": frame_data["file"],
                "faces_detected": len(all_faces),
                "face_details": face_details
            })
            
        except Exception as e:
            frame_results.append({
                "frame_index": frame_idx,
                "frame_file": frame_data.get("file", "unknown"),
                "error": str(e),
                "faces_detected": 0,
                "face_details": []
            })
    
    # Calculate summary
    valid_results = [r for r in frame_results if "error" not in r and r.get("faces_detected", 0) > 0]
    
    if len(valid_results) > 0:
        # Collect all face details
        all_face_details = []
        for result in valid_results:
            all_face_details.extend(result.get("face_details", []))
        
        if len(all_face_details) > 0:
            # Calculate statistics
            anomaly_scores = [f["anomaly_score"] for f in all_face_details]
            max_anomaly_score = max(anomaly_scores) if anomaly_scores else 0.0
            
            anomalous_faces = [f for f in all_face_details if f["anomaly_score"] >= 0.30]
            anomalous_faces_count = len(anomalous_faces)
            total_faces = len(all_face_details)
            anomalous_faces_ratio = anomalous_faces_count / total_faces if total_faces > 0 else 0.0
            
            # Count by severity
            critical_faces = [f for f in all_face_details if f["severity"] == "critical"]
            severe_faces = [f for f in all_face_details if f["severity"] == "severe"]
            moderate_faces = [f for f in all_face_details if f["severity"] == "moderate"]
            mild_faces = [f for f in all_face_details if f["severity"] == "mild"]
            
            avg_anomaly_score = np.mean([f["anomaly_score"] for f in anomalous_faces]) if anomalous_faces else 0.0
            
            # Find frames with anomalies
            frames_with_anomalies = []
            for idx, result in enumerate(valid_results):
                face_details = result.get("face_details", [])
                if any(f["anomaly_score"] >= 0.30 for f in face_details):
                    frames_with_anomalies.append(idx)
        else:
            max_anomaly_score = 0.0
            anomalous_faces_count = 0
            total_faces = 0
            anomalous_faces_ratio = 0.0
            critical_faces = []
            severe_faces = []
            moderate_faces = []
            mild_faces = []
            avg_anomaly_score = 0.0
            frames_with_anomalies = []
    else:
        max_anomaly_score = 0.0
        anomalous_faces_count = 0
        total_faces = 0
        anomalous_faces_ratio = 0.0
        critical_faces = []
        severe_faces = []
        moderate_faces = []
        mild_faces = []
        avg_anomaly_score = 0.0
        frames_with_anomalies = []
    
    return {
        "frame_results": frame_results,
        "summary": {
            "max_anomaly_score": float(max_anomaly_score),
            "anomalous_faces_count": anomalous_faces_count,
            "total_faces": total_faces,
            "anomalous_faces_ratio": float(anomalous_faces_ratio),
            "critical_faces_count": len(critical_faces),
            "severe_faces_count": len(severe_faces),
            "moderate_faces_count": len(moderate_faces),
            "mild_faces_count": len(mild_faces),
            "avg_anomaly_score": float(avg_anomaly_score),
            "frames_with_anomalies": frames_with_anomalies,
            "frames_analyzed": len(frame_results)
        }
    }
