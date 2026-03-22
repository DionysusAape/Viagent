"""Blur uniformity analysis for spatial agent.

Uses Laplacian variance as a CPU-friendly sharpness proxy to distinguish:
- global/consistent blur (likely real blur / compression / defocus)
- selective semantic blur (e.g., face-only blur) which can be suspicious

This skill is not a face detector. It consumes face bboxes from multi_face_collapse_detection.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from skill.spatial.ela import decode_data_url_to_cv2, load_image_from_path


@dataclass(frozen=True)
class BlurUniformityConfig:
    min_face_pixels: int = 20 * 20
    selective_ratio_threshold: float = 2.0  # bg_var / face_var
    min_frames_for_persistent: int = 2
    eps: float = 1e-6


def _union_face_mask(
    h: int,
    w: int,
    face_bboxes: List[Tuple[int, int, int, int]],
) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    for (x, y, bw, bh) in face_bboxes:
        x0 = max(0, int(x))
        y0 = max(0, int(y))
        x1 = min(w, int(x + bw))
        y1 = min(h, int(y + bh))
        if x1 > x0 and y1 > y0:
            mask[y0:y1, x0:x1] = True
    return mask


def _laplacian_variance(values: np.ndarray) -> float:
    if values.size < 32:
        return 0.0
    v = float(np.var(values))
    if not np.isfinite(v):
        return 0.0
    return v


def analyze_blur_uniformity(
    selected_frames: List[Dict[str, Any]],
    frames_dir: Path,
    frame_inputs: Optional[List[str]] = None,
    multi_face_results: Optional[Dict[str, Any]] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Analyze blur uniformity using Laplacian variance in face vs background regions."""
    cfg = BlurUniformityConfig(
        selective_ratio_threshold=float(
            (config or {}).get("blur_uniformity", {}).get("selective_ratio_threshold", 2.0)
        ),
        min_frames_for_persistent=int(
            (config or {}).get("blur_uniformity", {}).get("min_frames_for_persistent", 2)
        ),
    )

    frame_results: List[Dict[str, Any]] = []
    frames_with_selective_blur: List[int] = []

    mf_frame_results = (multi_face_results or {}).get("frame_results") or []

    for frame_idx, frame_data in enumerate(selected_frames):
        img = None
        if frame_inputs and frame_idx < len(frame_inputs) and frame_inputs[frame_idx]:
            img = decode_data_url_to_cv2(frame_inputs[frame_idx])
        if img is None:
            frame_path = frames_dir / frame_data["file"]
            if frame_path.exists():
                img = load_image_from_path(frame_path)
        if img is None:
            continue

        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        lap = cv2.Laplacian(gray, cv2.CV_64F)

        # Collect face bboxes from multi-face results (same indexing convention: frame_idx)
        face_bboxes: List[Tuple[int, int, int, int]] = []
        if frame_idx < len(mf_frame_results):
            face_details = mf_frame_results[frame_idx].get("face_details") or []
            for fd in face_details:
                bbox = fd.get("bbox")
                if bbox and len(bbox) == 4:
                    face_bboxes.append(tuple(bbox))  # type: ignore[arg-type]

        face_mask = _union_face_mask(h, w, face_bboxes) if face_bboxes else np.zeros((h, w), dtype=bool)
        face_pixels = int(face_mask.sum())
        bg_pixels = int((~face_mask).sum())

        global_var = _laplacian_variance(lap.reshape(-1))
        face_var = 0.0
        bg_var = 0.0

        if face_pixels >= cfg.min_face_pixels:
            face_var = _laplacian_variance(lap[face_mask])
            bg_var = _laplacian_variance(lap[~face_mask]) if bg_pixels > 0 else 0.0

        ratio = float((bg_var + cfg.eps) / (face_var + cfg.eps)) if face_pixels >= cfg.min_face_pixels else 0.0
        selective = bool(face_pixels >= cfg.min_face_pixels and ratio >= cfg.selective_ratio_threshold)
        if selective:
            frames_with_selective_blur.append(frame_idx)

        frame_results.append(
            {
                "frame_index": frame_idx,
                "frame_file": frame_data["file"],
                "faces_pixels": face_pixels,
                "global_lap_var": global_var,
                "face_lap_var": face_var,
                "bg_lap_var": bg_var,
                "bg_to_face_var_ratio": ratio,
                "selective_face_blur_suspected": selective,
            }
        )

    persistent = len(frames_with_selective_blur) >= cfg.min_frames_for_persistent

    return {
        "frame_results": frame_results,
        "summary": {
            "selective_ratio_threshold": cfg.selective_ratio_threshold,
            "frames_with_selective_face_blur": frames_with_selective_blur,
            "selective_face_blur_persistent": persistent,
            "frames_count": len(frame_results),
        },
    }

