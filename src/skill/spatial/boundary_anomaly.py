"""Boundary anomaly skill for detecting edge melting and transition artifacts.

Analyzes transition areas (hairline, chin, cheeks to background) for anomalies.
"""
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional
import base64


def decode_data_url_to_cv2(data_url: str) -> np.ndarray:
    """Decode base64 data URL to OpenCV image (BGR format)."""
    if "," in data_url:
        header, encoded = data_url.split(",", 1)
    else:
        encoded = data_url
    
    image_data = base64.b64decode(encoded)
    nparr = np.frombuffer(image_data, np.uint8)
    image_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if image_bgr is None:
        raise ValueError(f"Failed to decode image from data URL")
    
    return image_bgr


def load_image_from_path(image_path: Path) -> np.ndarray:
    """Load image from file path as OpenCV BGR array."""
    img = cv2.imread(str(image_path))
    if img is None:
        raise IOError(f"Failed to load image from {image_path}")
    return img


def detect_all_faces(image: np.ndarray) -> List[tuple]:
    """
    Detect ALL faces in the image using Haar Cascade.
    
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


def analyze_boundary_anomaly(
    selected_frames: List[Dict[str, Any]],
    frames_dir: Path,
    frame_inputs: Optional[List[str]] = None,
    config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Analyze boundary anomalies for selected frames.
    
    Args:
        selected_frames: List of selected frame dicts
        frames_dir: Directory containing frame files
        frame_inputs: Optional list of base64 data URLs
        config: Optional configuration dict
        
    Returns:
        Dict with "frame_results" and "summary" keys
    """
    boundary_band_pixels = 15  # Pixels to expand from face bbox for boundary region
    
    if config:
        boundary_band_pixels = int(config.get("boundary_band_pixels", 15))
    
    frame_results = []
    
    for frame_idx, frame_data in enumerate(selected_frames):
        try:
            # Load image
            img = None
            if frame_inputs:
                # Try to match by index in selected_frames list
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
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Detect all faces
            all_faces = detect_all_faces(img)
            
            if len(all_faces) == 0:
                frame_results.append({
                    "frame_index": frame_idx,
                    "frame_file": frame_data["file"],
                    "face_detected": False,
                    "edge_melting_score": 0.0,
                    "hair_boundary_score": 0.0,
                    "jawline_boundary_score": 0.0
                })
                continue
            
            # Calculate edge strength using Canny
            def calculate_edge_score(region):
                if region.size == 0:
                    return 0.0
                edges = cv2.Canny(region, 50, 150)
                edge_density = np.sum(edges > 0) / region.size
                return float(edge_density)
            
            # Calculate gradient smoothness (lower = more melting)
            def calculate_gradient_smoothness(region):
                if region.size == 0:
                    return 1.0
                sobel_x = cv2.Sobel(region, cv2.CV_64F, 1, 0, ksize=3)
                sobel_y = cv2.Sobel(region, cv2.CV_64F, 0, 1, ksize=3)
                gradient_mag = np.sqrt(sobel_x**2 + sobel_y**2)
                # High variance in gradient = inconsistent = melting
                gradient_std = np.std(gradient_mag)
                # Normalize (typical range: 0-50, smooth = low std)
                smoothness = 1.0 / (1.0 + gradient_std / 10.0)
                return float(smoothness)
            
            # Process all faces and take the maximum scores (most severe anomalies)
            max_edge_melting_score = 0.0
            max_hair_boundary_score = 0.0
            max_jawline_boundary_score = 0.0
            
            for fx, fy, fw, fh in all_faces:
                # Expand face bbox to create boundary band
                boundary_x1 = max(0, fx - boundary_band_pixels)
                boundary_y1 = max(0, fy - boundary_band_pixels)
                boundary_x2 = min(w, fx + fw + boundary_band_pixels)
                boundary_y2 = min(h, fy + fh + boundary_band_pixels)
                
                # Extract boundary regions
                # Top boundary (hairline region)
                top_band = gray[max(0, boundary_y1):fy, boundary_x1:boundary_x2]
                # Bottom boundary (jawline region)
                bottom_band = gray[fy+fh:min(h, boundary_y2), boundary_x1:boundary_x2]
                # Left boundary (cheek region)
                left_band = gray[fy:fy+fh, max(0, boundary_x1):fx]
                # Right boundary (cheek region)
                right_band = gray[fy:fy+fh, fx+fw:min(w, boundary_x2)]
                
                # Edge melting score (combines edge strength and gradient smoothness)
                top_edge_score = calculate_edge_score(top_band)
                top_smoothness = calculate_gradient_smoothness(top_band)
                hair_boundary_score = (1.0 - top_smoothness) * top_edge_score  # Low smoothness + high edges = melting
                
                bottom_edge_score = calculate_edge_score(bottom_band)
                bottom_smoothness = calculate_gradient_smoothness(bottom_band)
                jawline_boundary_score = (1.0 - bottom_smoothness) * bottom_edge_score
                
                left_edge_score = calculate_edge_score(left_band)
                left_smoothness = calculate_gradient_smoothness(left_band)
                right_edge_score = calculate_edge_score(right_band)
                right_smoothness = calculate_gradient_smoothness(right_band)
                
                # Overall edge melting score (average of all boundaries)
                edge_melting_score = (
                    hair_boundary_score + jawline_boundary_score +
                    (1.0 - left_smoothness) * left_edge_score +
                    (1.0 - right_smoothness) * right_edge_score
                ) / 4.0
                
                # Track maximum (most severe anomaly)
                if edge_melting_score > max_edge_melting_score:
                    max_edge_melting_score = edge_melting_score
                if hair_boundary_score > max_hair_boundary_score:
                    max_hair_boundary_score = hair_boundary_score
                if jawline_boundary_score > max_jawline_boundary_score:
                    max_jawline_boundary_score = jawline_boundary_score
            
            frame_results.append({
                "frame_index": frame_idx,
                "frame_file": frame_data["file"],
                "face_detected": True,
                "faces_count": len(all_faces),
                "edge_melting_score": float(max_edge_melting_score),
                "hair_boundary_score": float(max_hair_boundary_score),
                "jawline_boundary_score": float(max_jawline_boundary_score)
            })
            
        except Exception as e:
            frame_results.append({
                "frame_index": frame_idx,
                "frame_file": frame_data.get("file", "unknown"),
                "error": str(e),
                "edge_melting_score": 0.0,
                "hair_boundary_score": 0.0,
                "jawline_boundary_score": 0.0
            })
    
    # Calculate summary
    valid_results = [r for r in frame_results if r.get("face_detected", False) and "error" not in r]
    
    if len(valid_results) > 0:
        max_edge_melting = max(r["edge_melting_score"] for r in valid_results)
        max_hair_boundary = max(r["hair_boundary_score"] for r in valid_results)
        max_jawline_boundary = max(r["jawline_boundary_score"] for r in valid_results)
    else:
        max_edge_melting = 0.0
        max_hair_boundary = 0.0
        max_jawline_boundary = 0.0
    
    return {
        "frame_results": frame_results,
        "summary": {
            "max_edge_melting": float(max_edge_melting),
            "max_hair_boundary": float(max_hair_boundary),
            "max_jawline_boundary": float(max_jawline_boundary),
            "frames_analyzed": len(frame_results),
            "faces_detected": len(valid_results)
        }
    }
