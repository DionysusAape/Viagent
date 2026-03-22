"""Patch inconsistency skill for detecting local texture anomalies.

Analyzes patch-level inconsistencies within face regions using traditional CV features.
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


def calculate_patch_features(patch: np.ndarray) -> Dict[str, float]:
    """Calculate CV features for a patch."""
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY) if len(patch.shape) == 3 else patch
    
    # Sharpness (Laplacian variance)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    # Noise level (standard deviation)
    noise_level = np.std(gray)
    
    # High-frequency energy (using Sobel)
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    hf_energy = np.sqrt(sobel_x**2 + sobel_y**2).mean()
    
    # Color distribution (mean and std for each channel if color)
    if len(patch.shape) == 3:
        color_mean = np.mean(patch, axis=(0, 1))
        color_std = np.std(patch, axis=(0, 1))
        color_mismatch = np.std(color_mean)  # How different are the channels
    else:
        color_mismatch = 0.0
    
    return {
        "sharpness": float(laplacian_var),
        "noise_level": float(noise_level),
        "hf_energy": float(hf_energy),
        "color_mismatch": float(color_mismatch)
    }


def analyze_patch_inconsistency(
    selected_frames: List[Dict[str, Any]],
    frames_dir: Path,
    frame_inputs: Optional[List[str]] = None,
    config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Analyze patch-level inconsistencies for selected frames.
    
    Args:
        selected_frames: List of selected frame dicts
        frames_dir: Directory containing frame files
        frame_inputs: Optional list of base64 data URLs
        config: Optional configuration dict
        
    Returns:
        Dict with "frame_results" and "summary" keys
    """
    patch_size = 32  # Patch size in pixels
    if config:
        patch_size = int(config.get("patch_size", 32))
    
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
            
            # Detect all faces
            all_faces = detect_all_faces(img)
            
            if len(all_faces) == 0:
                frame_results.append({
                    "frame_index": frame_idx,
                    "frame_file": frame_data["file"],
                    "face_detected": False,
                    "patch_anomaly_score": 0.0,
                    "max_patch_score": 0.0,
                    "suspicious_regions": []
                })
                continue
            
            # Process all faces and collect patches from all face regions
            all_patch_features = []
            
            for fx, fy, fw, fh in all_faces:
                # Extract patches from face region
                for py in range(fy, fy + fh - patch_size, patch_size // 2):
                    for px in range(fx, fx + fw - patch_size, patch_size // 2):
                        patch = img[py:py+patch_size, px:px+patch_size]
                        if patch.shape[0] < patch_size or patch.shape[1] < patch_size:
                            continue
                        
                        features = calculate_patch_features(patch)
                        all_patch_features.append({
                            "x": px,
                            "y": py,
                            "features": features
                        })
            
            if len(all_patch_features) < 2:
                frame_results.append({
                    "frame_index": frame_idx,
                    "frame_file": frame_data["file"],
                    "face_detected": True,
                    "faces_count": len(all_faces),
                    "patch_anomaly_score": 0.0,
                    "max_patch_score": 0.0,
                    "suspicious_regions": []
                })
                continue
            
            # Calculate inconsistency: compare each patch with its neighbors
            patch_scores = []
            for i, patch1 in enumerate(all_patch_features):
                inconsistency = 0.0
                neighbor_count = 0
                
                for j, patch2 in enumerate(all_patch_features):
                    if i == j:
                        continue
                    
                    # Calculate distance (only consider immediate neighbors for efficiency)
                    dist = np.sqrt((patch1["x"] - patch2["x"])**2 + (patch1["y"] - patch2["y"])**2)
                    if dist > patch_size * 2:
                        continue
                    
                    neighbor_count += 1
                    
                    # Feature differences
                    f1 = patch1["features"]
                    f2 = patch2["features"]
                    
                    sharpness_diff = abs(f1["sharpness"] - f2["sharpness"]) / (f1["sharpness"] + f2["sharpness"] + 1e-6)
                    noise_diff = abs(f1["noise_level"] - f2["noise_level"]) / (f1["noise_level"] + f2["noise_level"] + 1e-6)
                    hf_diff = abs(f1["hf_energy"] - f2["hf_energy"]) / (f1["hf_energy"] + f2["hf_energy"] + 1e-6)
                    
                    inconsistency += (sharpness_diff + noise_diff + hf_diff) / 3.0
                
                if neighbor_count > 0:
                    inconsistency /= neighbor_count
                    patch_scores.append({
                        "x": patch1["x"],
                        "y": patch1["y"],
                        "score": inconsistency
                    })
            
            if len(patch_scores) == 0:
                patch_anomaly_score = 0.0
                max_patch_score = 0.0
                suspicious_regions = []
            else:
                patch_anomaly_score = np.mean([p["score"] for p in patch_scores])
                max_patch_score = max([p["score"] for p in patch_scores])
                # Suspicious regions: patches with score > 0.3
                suspicious_regions = [
                    {"x": p["x"], "y": p["y"], "score": p["score"]}
                    for p in patch_scores if p["score"] > 0.3
                ]
            
            frame_results.append({
                "frame_index": frame_idx,
                "frame_file": frame_data["file"],
                "face_detected": True,
                "faces_count": len(all_faces),
                "patch_anomaly_score": float(patch_anomaly_score),
                "max_patch_score": float(max_patch_score),
                "suspicious_regions": suspicious_regions
            })
            
        except Exception as e:
            frame_results.append({
                "frame_index": frame_idx,
                "frame_file": frame_data.get("file", "unknown"),
                "error": str(e),
                "patch_anomaly_score": 0.0,
                "max_patch_score": 0.0,
                "suspicious_regions": []
            })
    
    # Calculate summary
    valid_results = [r for r in frame_results if r.get("face_detected", False) and "error" not in r]
    
    if len(valid_results) > 0:
        max_patch_anomaly_score = max(r["patch_anomaly_score"] for r in valid_results)
        frames_with_anomalies = [i for i, r in enumerate(valid_results) if r["patch_anomaly_score"] > 0.3]
    else:
        max_patch_anomaly_score = 0.0
        frames_with_anomalies = []
    
    return {
        "frame_results": frame_results,
        "summary": {
            "max_patch_anomaly_score": float(max_patch_anomaly_score),
            "frames_with_anomalies": frames_with_anomalies,
            "frames_analyzed": len(frame_results),
            "faces_detected": len(valid_results)
        }
    }
