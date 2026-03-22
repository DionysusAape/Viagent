"""Local edge coherence skill for detecting boundary bleeding / floating objects.

This skill focuses on spatial edge behavior around object boundaries.
It is designed to complement ELA / boundary / patch anomaly by providing
gradient-based evidence about:
- How noisy/incoherent local edges are
- How wide / soft the transition zone is around strong edges

High edge noise + very soft transitions can indicate that foreground
objects (hands, faces, limbs) are "melting" into the background or
floating over it, which is common in AI/interpolated content.
"""
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional
from .ela import decode_data_url_to_cv2, load_image_from_path


def _compute_edge_coherence_metrics(gray: np.ndarray) -> Dict[str, float]:
    """Compute edge coherence metrics on a single grayscale frame.

    Returns:
        Dict with:
            edge_pixel_ratio: fraction of pixels that are edges
            edge_noise_ratio: fraction of edge pixels with high local gradient std
            soft_edge_ratio: ratio of neighbor-gradient to edge-gradient (higher = softer/wider edges)
    """
    h, w = gray.shape[:2]
    if h == 0 or w == 0:
        return {
            "edge_pixel_ratio": 0.0,
            "edge_noise_ratio": 0.0,
            "soft_edge_ratio": 0.0,
        }

    # 1) Detect edges (Canny is relatively robust under low resolution / compression)
    # Thresholds chosen conservatively; can be tuned via config later if needed.
    edges = cv2.Canny(gray, 50, 150)

    edge_count = int(np.count_nonzero(edges))
    total_pixels = h * w
    edge_pixel_ratio = edge_count / float(total_pixels) if total_pixels > 0 else 0.0

    if edge_count == 0:
        return {
            "edge_pixel_ratio": 0.0,
            "edge_noise_ratio": 0.0,
            "soft_edge_ratio": 0.0,
        }

    # 2) Gradient magnitude (Sobel) to characterize edge strength
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = cv2.magnitude(gx, gy)

    # 3) Local gradient std (via mean & mean-of-squares with 3x3 Gaussian blur)
    mean = cv2.GaussianBlur(grad_mag, (3, 3), 0)
    mean_sq = cv2.GaussianBlur(grad_mag * grad_mag, (3, 3), 0)
    var = cv2.max(mean_sq - mean * mean, 0)
    local_std = cv2.sqrt(var)

    edge_mask = edges > 0
    edge_std_vals = local_std[edge_mask]

    if edge_std_vals.size == 0:
        edge_noise_ratio = 0.0
    else:
        # Threshold chosen empirically to mark "noisy/incoherent" edges.
        # Larger std => more chaotic local gradient behavior around edge.
        std_threshold = 10.0
        high_noise = edge_std_vals > std_threshold
        edge_noise_ratio = float(np.count_nonzero(high_noise)) / float(edge_std_vals.size)

    # 4) Soft-edge / wide-transition score:
    # Compare gradient magnitude on edge pixels vs a small neighborhood around them.
    kernel = np.ones((3, 3), dtype=np.uint8)
    dilated_edges = cv2.dilate(edges, kernel, iterations=1)
    neighbor_mask = (dilated_edges > 0) & (~edge_mask)

    edge_grad_vals = grad_mag[edge_mask]
    neighbor_grad_vals = grad_mag[neighbor_mask]

    if edge_grad_vals.size == 0 or neighbor_grad_vals.size == 0:
        soft_edge_ratio = 0.0
    else:
        mean_edge_grad = float(np.mean(edge_grad_vals))
        mean_neighbor_grad = float(np.mean(neighbor_grad_vals))
        if mean_edge_grad > 0:
            soft_edge_ratio = mean_neighbor_grad / mean_edge_grad
        else:
            soft_edge_ratio = 0.0

    return {
        "edge_pixel_ratio": float(edge_pixel_ratio),
        "edge_noise_ratio": float(edge_noise_ratio),
        "soft_edge_ratio": float(soft_edge_ratio),
    }


def analyze_edge_coherence(
    selected_frames: List[Dict[str, Any]],
    frames_dir: Path,
    frame_inputs: Optional[List[str]] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Analyze local edge coherence for selected frames.

    Args:
        selected_frames: List of selected frame dicts
        frames_dir: Directory containing frame files
        frame_inputs: Optional list of base64 data URLs (aligned with selected_frames)
        config: Optional configuration dict (currently unused, kept for future tuning)

    Returns:
        Dict with "frame_results" and "summary" keys.
    """
    frame_results: List[Dict[str, Any]] = []

    for frame_idx, frame_data in enumerate(selected_frames):
        try:
            img = None
            if frame_inputs:
                if frame_idx < len(frame_inputs) and frame_inputs[frame_idx]:
                    img = decode_data_url_to_cv2(frame_inputs[frame_idx])

            if img is None:
                frame_path = frames_dir / frame_data["file"]
                if not frame_path.exists():
                    frame_results.append(
                        {
                            "frame_index": frame_idx,
                            "frame_file": frame_data.get("file", "unknown"),
                            "error": "frame_file_not_found",
                            "edge_pixel_ratio": 0.0,
                            "edge_noise_ratio": 0.0,
                            "soft_edge_ratio": 0.0,
                        }
                    )
                    continue
                img = load_image_from_path(frame_path)

            if img is None:
                frame_results.append(
                    {
                        "frame_index": frame_idx,
                        "frame_file": frame_data.get("file", "unknown"),
                        "error": "image_decode_failed",
                        "edge_pixel_ratio": 0.0,
                        "edge_noise_ratio": 0.0,
                        "soft_edge_ratio": 0.0,
                    }
                )
                continue

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            metrics = _compute_edge_coherence_metrics(gray)

            frame_results.append(
                {
                    "frame_index": frame_idx,
                    "frame_file": frame_data["file"],
                    "edge_pixel_ratio": metrics["edge_pixel_ratio"],
                    "edge_noise_ratio": metrics["edge_noise_ratio"],
                    "soft_edge_ratio": metrics["soft_edge_ratio"],
                }
            )
        except Exception as e:
            frame_results.append(
                {
                    "frame_index": frame_idx,
                    "frame_file": frame_data.get("file", "unknown"),
                    "error": str(e),
                    "edge_pixel_ratio": 0.0,
                    "edge_noise_ratio": 0.0,
                    "soft_edge_ratio": 0.0,
                }
            )

    valid_results = [r for r in frame_results if "error" not in r]

    if valid_results:
        max_edge_noise_ratio = max(r["edge_noise_ratio"] for r in valid_results)
        mean_edge_noise_ratio = float(
            np.mean([r["edge_noise_ratio"] for r in valid_results])
        )
        max_soft_edge_ratio = max(r["soft_edge_ratio"] for r in valid_results)
        mean_soft_edge_ratio = float(
            np.mean([r["soft_edge_ratio"] for r in valid_results])
        )
        # Mark frames with notably soft / noisy edges as candidates
        anomalous_frames = [
            r["frame_index"]
            for r in valid_results
            if (r["soft_edge_ratio"] > 0.6 and r["edge_noise_ratio"] > 0.4)
        ]
    else:
        max_edge_noise_ratio = 0.0
        mean_edge_noise_ratio = 0.0
        max_soft_edge_ratio = 0.0
        mean_soft_edge_ratio = 0.0
        anomalous_frames = []

    return {
        "frame_results": frame_results,
        "summary": {
            "max_edge_noise_ratio": float(max_edge_noise_ratio),
            "mean_edge_noise_ratio": float(mean_edge_noise_ratio),
            "max_soft_edge_ratio": float(max_soft_edge_ratio),
            "mean_soft_edge_ratio": float(mean_soft_edge_ratio),
            "frames_with_soft_noisy_edges": anomalous_frames,
            "frames_analyzed": len(frame_results),
        },
    }

