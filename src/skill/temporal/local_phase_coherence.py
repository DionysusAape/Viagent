"""
Local Phase Coherence (LPC) temporal skill.

High-level idea:
- For each frame, measure how coherent the edge phase/orientation is
  across multiple spatial scales in a face region (if detected).
- Real-camera footage tends to have stable, sharp edges whose phase
  is consistent across scales and over time.
- AI / heavily synthesized content often shows "melting" or unstable
  edges where the local phase structure becomes inconsistent between
  scales and from frame to frame.

We approximate LPC using multi-scale gradient orientations instead of
full log-Gabor / Riesz filters to keep the implementation lightweight
and CPU-friendly.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from util.logger import logger


@dataclass
class LPCFrameResult:
    frame_index: int  # 1-based frame index from metadata
    timestamp_sec: Optional[float]
    lpc_score: float  # 0.0-1.0, higher = more coherent/sharp edges


@dataclass
class LPCSequenceAnalysis:
    has_face: bool
    frame_results: List[LPCFrameResult]
    std_over_time: float
    max_jump: float
    jump_rate: float
    summary: str


def _load_image(frames_dir: Path, frame_meta: Dict[str, Any]) -> Optional[np.ndarray]:
    file_name = frame_meta.get("file")
    if not file_name:
        return None
    img_path = frames_dir / file_name
    img = cv2.imread(str(img_path))
    if img is None:
        logger.warning(f"LPC skill: failed to load image {img_path}")
    return img


def _detect_all_faces(image_bgr: np.ndarray) -> List[Tuple[int, int, int, int]]:
    """
    Detect ALL faces in the image using Haar Cascade.
    
    Returns:
        List of (x, y, w, h) bounding boxes for all detected faces, sorted by area (largest first)
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
    if len(faces) == 0:
        return []
    # Sort by area (largest first) and return all faces
    faces_sorted = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
    return [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in faces_sorted]


def _compute_lpc_score_for_roi(gray_roi: np.ndarray) -> float:
    """
    Approximate local phase coherence score for a grayscale ROI.

    Implementation:
    - Compute gradients at multiple Gaussian scales.
    - Use gradient orientations as proxy for phase.
    - At each pixel, measure how consistent orientations are across scales.
    - Average coherence over all edge pixels as a scalar score in [0, 1].
    """
    if gray_roi.size == 0:
        return 0.0

    img = gray_roi.astype(np.float32) / 255.0

    # Multi-scale Gaussian sigmas
    sigmas = [1.0, 2.0, 4.0]

    orientations = []
    magnitudes = []

    for sigma in sigmas:
        blurred = cv2.GaussianBlur(img, ksize=(0, 0), sigmaX=sigma, sigmaY=sigma)
        gx = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3)
        mag = cv2.magnitude(gx, gy)
        theta = cv2.phase(gx, gy, angleInDegrees=False)

        orientations.append(theta)
        magnitudes.append(mag)

    mags = np.stack(magnitudes, axis=-1)  # H x W x S
    thetas = np.stack(orientations, axis=-1)  # H x W x S

    # Only consider reasonably strong edges
    edge_mag = mags.mean(axis=-1)
    edge_mask = edge_mag > 0.05  # empirical threshold in [0,1] space

    if not np.any(edge_mask):
        return 0.0

    # Convert orientations to complex unit vectors, weight by magnitude
    complex_vectors = np.exp(1j * thetas) * mags
    sum_vec = complex_vectors.sum(axis=-1)  # H x W
    sum_mag = mags.sum(axis=-1) + 1e-6

    coherence_map = np.abs(sum_vec) / sum_mag  # [0, 1]

    # Average coherence where edges exist
    lpc_score = float(coherence_map[edge_mask].mean())
    # Clamp numerically
    if not np.isfinite(lpc_score):
        return 0.0
    return max(0.0, min(1.0, lpc_score))


def analyze_lpc_sequence(
    frames_dir: str,
    frames_meta: Sequence[Dict[str, Any]],
    max_frames: int = 120,
) -> Optional[LPCSequenceAnalysis]:
    """
    Analyze a sequence of frames for temporal stability of local phase coherence
    around detected face regions.

    - Uses Haar cascade to find all faces per frame (processes all detected faces).
    - Computes an LPC score (0-1) for an expanded ROI around the face.
    - Subsamples to at most `max_frames` frames.
    - Returns a summary describing stability vs. fluctuation over time.

    If no faces / valid LPC scores are found, returns None.
    """
    if not frames_meta:
        return None

    frames_dir_path = Path(frames_dir)

    total = len(frames_meta)
    if total <= max_frames:
        selected_indices = list(range(total))
    else:
        selected_indices = np.linspace(0, total - 1, max_frames, dtype=int).tolist()

    frame_results: List[LPCFrameResult] = []

    for idx in selected_indices:
        meta = frames_meta[idx]
        image = _load_image(frames_dir_path, meta)
        if image is None:
            continue

        h, w = image.shape[:2]
        all_faces = _detect_all_faces(image)

        if len(all_faces) == 0:
            # If no face, skip this frame (we only care about human-like edges here)
            continue

        # Process all faces and take the average LPC score (or minimum for more conservative)
        # Using minimum to catch the most unstable face
        lpc_scores = []
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        for x, y, fw, fh in all_faces:
            # Expand ROI slightly around face to capture hairline/cheeks
            margin = int(0.2 * max(fw, fh))
            x1 = max(0, x - margin)
            y1 = max(0, y - margin)
            x2 = min(w, x + fw + margin)
            y2 = min(h, y + fh + margin)

            roi = gray[y1:y2, x1:x2]
            lpc_score = _compute_lpc_score_for_roi(roi)
            lpc_scores.append(lpc_score)
        
        # Use minimum LPC score (most unstable face) as the frame's score
        # This is more conservative and catches anomalies better
        frame_lpc_score = min(lpc_scores) if lpc_scores else 0.0

        timestamp = meta.get("timestamp_sec")
        frame_index = int(meta.get("index", idx + 1))

        frame_results.append(
            LPCFrameResult(
                frame_index=frame_index,
                timestamp_sec=float(timestamp) if timestamp is not None else None,
                lpc_score=frame_lpc_score,
            )
        )

    if not frame_results:
        return None

    scores = np.array([r.lpc_score for r in frame_results], dtype=np.float32)
    std_over_time = float(scores.std())

    if len(scores) > 1:
        diffs = np.abs(np.diff(scores))
        max_jump = float(diffs.max())
        # Consider jumps larger than 0.08 as "significant" by default
        jump_mask = diffs > 0.08
        jump_rate = float(jump_mask.sum()) / float(len(diffs))
    else:
        max_jump = 0.0
        jump_rate = 0.0

    total_frames = len(frame_results)
    min_lpc = float(scores.min())
    max_lpc = float(scores.max())
    mean_lpc = float(scores.mean())

    summary_parts: List[str] = []
    summary_parts.append(
        f"Local phase coherence (edge sharpness) analysis over {total_frames} face-visible frames: "
        f"LPC range ≈ [{min_lpc:.3f}, {max_lpc:.3f}], mean LPC ≈ {mean_lpc:.3f}, "
        f"std over time ≈ {std_over_time:.3f}."
    )

    if len(scores) > 1:
        summary_parts.append(
            f" Max frame-to-frame LPC change ≈ {max_jump:.3f} with jump rate ≈ {jump_rate * 100:.1f}%."
        )
        if max_jump > 0.15 or (std_over_time > 0.08 and jump_rate > 0.15):
            summary_parts.append(
                " These fluctuations suggest that edge/phase structure around the face is temporally unstable, "
                "which may indicate dynamically synthesized or intermittently re-generated facial regions."
            )
        else:
            summary_parts.append(
                " LPC variation over time is modest; facial-edge phase structure appears reasonably stable for real-camera footage."
            )
    else:
        summary_parts.append(
            " Only a single eligible frame was available; LPC temporal stability cannot be assessed."
        )

    summary = " ".join(summary_parts)

    return LPCSequenceAnalysis(
        has_face=True,
        frame_results=frame_results,
        std_over_time=std_over_time,
        max_jump=max_jump,
        jump_rate=jump_rate,
        summary=summary,
    )

