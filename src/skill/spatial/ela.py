"""ELA (Error Level Analysis) boundary skill for detecting edge halos.

Analyzes JPEG compression artifacts at face boundaries to detect abnormal halos.
"""
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional
import base64
import io
from PIL import Image


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


def generate_ela_image(image: np.ndarray, quality: int = 95) -> np.ndarray:
    """
    Generate ELA (Error Level Analysis) image.
    
    Args:
        image: Input BGR image
        quality: JPEG quality for recompression (default: 95)
        
    Returns:
        ELA difference image (grayscale)
    """
    # Convert BGR to RGB for PIL
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_rgb)
    
    # Save to memory as JPEG with specified quality
    buffer = io.BytesIO()
    pil_image.save(buffer, format='JPEG', quality=quality)
    buffer.seek(0)
    
    # Reload and convert back to numpy
    recompressed = Image.open(buffer)
    recompressed_array = np.array(recompressed)
    
    # Convert RGB to grayscale for difference calculation
    original_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    recompressed_gray = cv2.cvtColor(cv2.cvtColor(recompressed_array, cv2.COLOR_RGB2BGR), cv2.COLOR_BGR2GRAY)
    
    # Calculate absolute difference
    ela_diff = cv2.absdiff(original_gray.astype(np.float32), recompressed_gray.astype(np.float32))
    
    return ela_diff.astype(np.uint8)


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


def analyze_ela_boundary(
    selected_frames: List[Dict[str, Any]],
    frames_dir: Path,
    frame_inputs: Optional[List[str]] = None,
    config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Analyze ELA boundary artifacts for selected frames.
    
    Args:
        selected_frames: List of selected frame dicts
        frames_dir: Directory containing frame files
        frame_inputs: Optional list of base64 data URLs
        config: Optional configuration dict
        
    Returns:
        Dict with "frame_results" and "summary" keys
    """
    ela_quality = 95
    boundary_band_pixels = 10  # Pixels to expand from face bbox for boundary region
    
    if config:
        ela_quality = int(config.get("ela_quality", 95))
        boundary_band_pixels = int(config.get("boundary_band_pixels", 10))
    
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
            
            # Generate ELA image
            ela_diff = generate_ela_image(img, quality=ela_quality)
            
            # Detect all faces
            all_faces = detect_all_faces(img)
            
            if len(all_faces) == 0:
                # No face detected, skip this frame
                frame_results.append({
                    "frame_index": frame_idx,
                    "frame_file": frame_data["file"],
                    "face_detected": False,
                    "boundary_ela_mean": 0.0,
                    "background_ela_mean": 0.0,
                    "boundary_bg_ratio": 0.0,
                    "halo_score": 0.0
                })
                continue
            
            # Process all faces and take the maximum halo score (most severe anomaly)
            max_halo_score = 0.0
            max_boundary_bg_ratio = 0.0
            max_boundary_ela_mean = 0.0
            max_background_ela_mean = 0.0
            
            for fx, fy, fw, fh in all_faces:
                # Expand face bbox to create boundary band
                boundary_x1 = max(0, fx - boundary_band_pixels)
                boundary_y1 = max(0, fy - boundary_band_pixels)
                boundary_x2 = min(w, fx + fw + boundary_band_pixels)
                boundary_y2 = min(h, fy + fh + boundary_band_pixels)
                
                # Create boundary mask (outer band of expanded bbox)
                boundary_mask = np.zeros((h, w), dtype=np.uint8)
                # Inner face region
                cv2.rectangle(boundary_mask, (fx, fy), (fx + fw, fy + fh), 0, -1)
                # Outer expanded region
                cv2.rectangle(boundary_mask, (boundary_x1, boundary_y1), (boundary_x2, boundary_y2), 255, -1)
                # Boundary band = outer - inner
                inner_mask = np.zeros((h, w), dtype=np.uint8)
                cv2.rectangle(inner_mask, (fx, fy), (fx + fw, fy + fh), 255, -1)
                boundary_mask = cv2.subtract(boundary_mask, inner_mask)
                
                # Background region (everything outside expanded bbox)
                background_mask = np.ones((h, w), dtype=np.uint8) * 255
                cv2.rectangle(background_mask, (boundary_x1, boundary_y1), (boundary_x2, boundary_y2), 0, -1)
                
                # Calculate mean ELA values
                boundary_ela_mean = np.mean(ela_diff[boundary_mask > 0]) if np.sum(boundary_mask > 0) > 0 else 0.0
                background_ela_mean = np.mean(ela_diff[background_mask > 0]) if np.sum(background_mask > 0) > 0 else 0.0
                
                # Calculate ratio
                if background_ela_mean > 0:
                    boundary_bg_ratio = boundary_ela_mean / background_ela_mean
                else:
                    boundary_bg_ratio = 0.0
                
                # Halo score (threshold: 1.5 indicates abnormal halo)
                halo_score = max(0.0, boundary_bg_ratio - 1.0)  # Normalize: >1.5 = high score
                
                # Track maximum (most severe anomaly)
                if halo_score > max_halo_score:
                    max_halo_score = halo_score
                    max_boundary_bg_ratio = boundary_bg_ratio
                    max_boundary_ela_mean = boundary_ela_mean
                    max_background_ela_mean = background_ela_mean
            
            frame_results.append({
                "frame_index": frame_idx,
                "frame_file": frame_data["file"],
                "face_detected": True,
                "faces_count": len(all_faces),
                "boundary_ela_mean": float(max_boundary_ela_mean),
                "background_ela_mean": float(max_background_ela_mean),
                "boundary_bg_ratio": float(max_boundary_bg_ratio),
                "halo_score": float(max_halo_score)
            })
            
        except Exception as e:
            # Skip frame on error
            frame_results.append({
                "frame_index": frame_idx,
                "frame_file": frame_data.get("file", "unknown"),
                "error": str(e),
                "boundary_ela_mean": 0.0,
                "background_ela_mean": 0.0,
                "boundary_bg_ratio": 0.0,
                "halo_score": 0.0
            })
    
    # Calculate summary
    valid_results = [r for r in frame_results if r.get("face_detected", False) and "error" not in r]
    
    if len(valid_results) > 0:
        max_halo_score = max(r["halo_score"] for r in valid_results)
        mean_boundary_bg_ratio = np.mean([r["boundary_bg_ratio"] for r in valid_results])
        anomalous_frames = [i for i, r in enumerate(valid_results) if r["boundary_bg_ratio"] > 1.5]
    else:
        max_halo_score = 0.0
        mean_boundary_bg_ratio = 0.0
        anomalous_frames = []
    
    return {
        "frame_results": frame_results,
        "summary": {
            "max_halo_score": float(max_halo_score),
            "mean_boundary_bg_ratio": float(mean_boundary_bg_ratio),
            "anomalous_frames": anomalous_frames,
            "frames_analyzed": len(frame_results),
            "faces_detected": len(valid_results)
        }
    }
