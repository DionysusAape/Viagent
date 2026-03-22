"""
Feature Stability Check temporal skill.

High-level idea:
- Extract keypoints from the first frame using cv2.goodFeaturesToTrack
- Track these keypoints across subsequent frames using cv2.calcOpticalFlowPyrLK
- Analyze keypoint longevity (how many keypoints survive over 10 frames)
- Analyze trajectory smoothness (compute acceleration/jerk to detect artificial jitter)
- Focus on high-motion regions (faces and hands) if detected

This helps detect pixel-level instability that indicates AI generation or manipulation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from util.logger import logger


@dataclass
class FeatureStabilityAnalysis:
    """Results from feature stability analysis."""
    has_valid_tracking: bool
    initial_keypoints: int
    survival_rate_10_frames: float  # Percentage of keypoints surviving 10 frames
    avg_trajectory_smoothness: float  # Average smoothness score (0-1, higher = smoother)
    high_jerk_ratio: float  # Ratio of keypoints with high jerk (non-physical jumps)
    summary: str


def _load_image(frames_dir: Path, frame_meta: Dict[str, Any]) -> Optional[np.ndarray]:
    """Load an image from frames directory."""
    file_name = frame_meta.get("file")
    if not file_name:
        return None
    img_path = frames_dir / file_name
    img = cv2.imread(str(img_path))
    if img is None:
        logger.warning(f"Feature stability: failed to load image {img_path}")
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


def _extract_keypoints(gray: np.ndarray, roi: Optional[Tuple[int, int, int, int]] = None) -> np.ndarray:
    """
    Extract keypoints using goodFeaturesToTrack.
    
    If roi is provided, extract keypoints only from that region.
    Returns keypoints with shape (N, 2).
    """
    if roi is not None:
        x, y, w, h = roi
        roi_gray = gray[y:y+h, x:x+w]
        if roi_gray.size == 0:
            return np.array([]).reshape(0, 2)
        # Extract keypoints in ROI
        keypoints = cv2.goodFeaturesToTrack(
            roi_gray,
            maxCorners=200,
            qualityLevel=0.01,
            minDistance=10,
            blockSize=3,
            useHarrisDetector=False,
            k=0.04
        )
        if keypoints is None or len(keypoints) == 0:
            return np.array([]).reshape(0, 2)
        # Reshape from (N, 1, 2) to (N, 2)
        if keypoints.ndim == 3:
            keypoints = keypoints.reshape(-1, 2)
        # Convert ROI coordinates to full image coordinates
        keypoints[:, 0] += x
        keypoints[:, 1] += y
        return keypoints
    else:
        # Extract from entire image
        keypoints = cv2.goodFeaturesToTrack(
            gray,
            maxCorners=200,
            qualityLevel=0.01,
            minDistance=10,
            blockSize=3,
            useHarrisDetector=False,
            k=0.04
        )
        if keypoints is None or len(keypoints) == 0:
            return np.array([]).reshape(0, 2)
        # Reshape from (N, 1, 2) to (N, 2)
        if keypoints.ndim == 3:
            keypoints = keypoints.reshape(-1, 2)
        return keypoints


def _track_keypoints(
    prev_gray: np.ndarray,
    curr_gray: np.ndarray,
    prev_points: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Track keypoints from prev_gray to curr_gray using PyrLK.
    
    Returns:
        - next_points: tracked points with shape (N, 2)
        - status: array of 1s (tracked) or 0s (lost)
        - err: tracking error
    """
    if len(prev_points) == 0:
        return prev_points.copy(), np.array([]), np.array([])
    
    # Ensure prev_points has shape (N, 2) for cv2.calcOpticalFlowPyrLK
    if prev_points.ndim == 1:
        prev_points = prev_points.reshape(-1, 2)
    elif prev_points.ndim == 3:
        prev_points = prev_points.reshape(-1, 2)
    
    next_points, status, err = cv2.calcOpticalFlowPyrLK(
        prev_gray,
        curr_gray,
        prev_points,
        None,
        winSize=(15, 15),
        maxLevel=2,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
    )
    
    # Reshape next_points from (N, 1, 2) to (N, 2)
    if next_points is not None and len(next_points) > 0:
        if next_points.ndim == 3:
            next_points = next_points.reshape(-1, 2)
    else:
        next_points = np.array([]).reshape(0, 2)
    
    return next_points, status.flatten(), err.flatten()


def _filter_boundary_losses(
    points: np.ndarray,
    status: np.ndarray,
    img_shape: Tuple[int, int],
    margin: int = 5
) -> np.ndarray:
    """
    Filter out keypoints that were lost due to moving out of frame boundaries.
    
    Returns a boolean mask: True = valid loss (not boundary), False = boundary loss.
    """
    h, w = img_shape
    valid_losses = np.ones_like(status, dtype=bool)
    
    if len(points) == 0:
        return valid_losses
    
    # Ensure points have shape (N, 2)
    if points.ndim == 3:
        points = points.reshape(-1, 2)
    elif points.ndim == 1:
        points = points.reshape(-1, 2)
    
    # Points that are lost (status == 0) but are near boundaries are likely boundary losses
    lost_mask = (status == 0)
    near_left = points[:, 0] < margin
    near_right = points[:, 0] > (w - margin)
    near_top = points[:, 1] < margin
    near_bottom = points[:, 1] > (h - margin)
    near_boundary = near_left | near_right | near_top | near_bottom
    
    # Mark boundary losses as invalid (they shouldn't count as tracking failures)
    valid_losses[lost_mask & near_boundary] = False
    
    return valid_losses


def _compute_trajectory_smoothness(trajectories: List[np.ndarray]) -> Tuple[float, float]:
    """
    Compute trajectory smoothness by analyzing acceleration (second derivative).
    
    Returns:
        - avg_smoothness: average smoothness score (0-1, higher = smoother)
        - high_jerk_ratio: ratio of keypoints with high jerk (non-physical jumps)
    """
    if len(trajectories) < 3:
        return 1.0, 0.0
    
    smoothness_scores = []
    high_jerk_count = 0
    total_keypoints = 0
    
    # Process each keypoint trajectory
    num_keypoints = len(trajectories[0])
    for kp_idx in range(num_keypoints):
        # Extract trajectory for this keypoint
        traj = np.array([t[kp_idx] for t in trajectories])
        
        if len(traj) < 3:
            continue
        
        total_keypoints += 1
        
        # Compute velocity (first derivative)
        velocities = np.diff(traj, axis=0)
        vel_magnitudes = np.linalg.norm(velocities, axis=1)
        
        if len(vel_magnitudes) < 2:
            smoothness_scores.append(1.0)
            continue
        
        # Compute acceleration (second derivative)
        accelerations = np.diff(vel_magnitudes)
        acc_magnitudes = np.abs(accelerations)
        
        # Compute jerk (third derivative) - sudden changes in acceleration
        if len(acc_magnitudes) > 1:
            jerks = np.diff(acc_magnitudes)
            max_jerk = np.max(np.abs(jerks)) if len(jerks) > 0 else 0.0
        else:
            max_jerk = 0.0
        
        # Smoothness: lower acceleration/jerk = smoother
        # Normalize by average velocity to make it scale-invariant
        avg_velocity = np.mean(vel_magnitudes) if len(vel_magnitudes) > 0 else 1.0
        if avg_velocity > 0:
            normalized_acc = np.mean(acc_magnitudes) / (avg_velocity + 1e-6)
            smoothness = 1.0 / (1.0 + normalized_acc * 10.0)  # Map to [0, 1]
        else:
            smoothness = 1.0  # Stationary = perfectly smooth
        
        smoothness_scores.append(smoothness)
        
        # High jerk threshold: sudden non-physical jumps
        # If jerk is very high relative to velocity, it's suspicious
        if avg_velocity > 0 and max_jerk > (avg_velocity * 0.5):
            high_jerk_count += 1
    
    if total_keypoints == 0:
        return 1.0, 0.0
    
    avg_smoothness = np.mean(smoothness_scores) if smoothness_scores else 1.0
    high_jerk_ratio = high_jerk_count / total_keypoints if total_keypoints > 0 else 0.0
    
    return float(avg_smoothness), float(high_jerk_ratio)


def analyze_feature_stability(
    frames_dir: Path,
    frames_meta: Sequence[Dict[str, Any]],
) -> Optional[FeatureStabilityAnalysis]:
    """
    Analyze feature point stability across frames.
    
    Args:
        frames_dir: Directory containing frame images
        frames_meta: List of frame metadata dicts with 'file' key
    
    Returns:
        FeatureStabilityAnalysis object, or None if analysis fails
    """
    if not frames_meta or len(frames_meta) < 2:
        logger.warning("Feature stability: insufficient frames")
        return None
    
    # Load first frame
    first_frame = _load_image(frames_dir, frames_meta[0])
    if first_frame is None:
        logger.warning("Feature stability: failed to load first frame")
        return None
    
    first_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    
    # Detect all faces to focus tracking on high-motion regions
    all_faces = _detect_all_faces(first_frame)
    
    # Extract keypoints from all face regions (merge all ROIs)
    # If multiple faces, we'll extract keypoints from each and combine them
    initial_keypoints_list = []
    if len(all_faces) > 0:
        for face_roi in all_faces:
            kp = _extract_keypoints(first_gray, face_roi)
            if len(kp) > 0:
                initial_keypoints_list.append(kp)
    
    # Also extract keypoints from the whole frame if no faces or to supplement
    if len(initial_keypoints_list) == 0:
        kp = _extract_keypoints(first_gray, None)
        if len(kp) > 0:
            initial_keypoints_list.append(kp)
    
    # Combine all keypoints
    if len(initial_keypoints_list) == 0:
        logger.warning("Feature stability: no keypoints extracted")
        return None
    
    # Concatenate all keypoints
    if len(initial_keypoints_list) > 1:
        initial_keypoints = np.vstack(initial_keypoints_list)
    else:
        initial_keypoints = initial_keypoints_list[0]
    
    initial_count = len(initial_keypoints)
    logger.debug(f"Feature stability: extracted {initial_count} keypoints from first frame")
    
    # Track keypoints across frames
    # We'll maintain a mapping from initial keypoint index to current position
    # and track which ones survive
    prev_gray = first_gray
    # Ensure initial_keypoints has shape (N, 2)
    if initial_keypoints.ndim == 3:
        initial_keypoints = initial_keypoints.reshape(-1, 2)
    prev_points = initial_keypoints.copy()
    keypoint_trajectories = {}  # Map from initial index to list of (x, y) positions
    for i in range(len(initial_keypoints)):
        # Store as (x, y) tuple for easier handling
        keypoint_trajectories[i] = [tuple(initial_keypoints[i])]
    
    window_size = min(10, len(frames_meta) - 1)  # Analyze over 10 frames or available frames
    # Maintain ordered list of survived indices (order matches prev_points)
    survived_indices_list = list(range(len(initial_keypoints)))
    
    for frame_idx in range(1, min(len(frames_meta), window_size + 1)):
        curr_frame = _load_image(frames_dir, frames_meta[frame_idx])
        if curr_frame is None:
            break
        
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        
        # Track keypoints from previous frame
        if len(prev_points) == 0:
            break
        
        next_points, status, err = _track_keypoints(prev_gray, curr_gray, prev_points)
        
        # Filter boundary losses - mark boundary losses as invalid
        valid_losses = _filter_boundary_losses(next_points, status, curr_gray.shape)
        
        # Update trajectories: only for successfully tracked points (status == 1)
        # and exclude boundary losses (valid_losses == False means it's a boundary loss)
        new_prev_points = []
        new_survived_indices_list = []
        
        for idx_in_prev, orig_idx in enumerate(survived_indices_list):
            if idx_in_prev >= len(status):
                break
            if status[idx_in_prev] == 1 and valid_losses[idx_in_prev]:
                # Successfully tracked and not a boundary loss
                point = next_points[idx_in_prev]
                # Ensure point is a 1D array with 2 elements
                if isinstance(point, np.ndarray):
                    if point.ndim > 1:
                        point = point.flatten()
                    elif point.ndim == 0:
                        # Scalar, shouldn't happen but handle it
                        point = np.array([point, point])
                else:
                    # Convert to numpy array
                    point = np.array(point)
                    if point.ndim == 0:
                        point = np.array([point, point])
                keypoint_trajectories[orig_idx].append(tuple(point))
                new_prev_points.append(point.copy())
                new_survived_indices_list.append(orig_idx)
            # If status == 0 (lost) and not a boundary loss, it's a real tracking failure
            # We don't add it to new_prev_points or new_survived_indices_list
        
        if len(new_prev_points) == 0:
            break
        
        # Ensure prev_points has shape (N, 2)
        prev_points = np.array(new_prev_points)
        if prev_points.ndim == 1:
            prev_points = prev_points.reshape(-1, 2)
        elif prev_points.ndim == 3:
            prev_points = prev_points.reshape(-1, 2)
        survived_indices_list = new_survived_indices_list
        prev_gray = curr_gray
    
    if len(survived_indices_list) == 0 or len(keypoint_trajectories) == 0:
        logger.warning("Feature stability: insufficient tracking data")
        return None
    
    # Calculate survival rate (over window_size frames)
    # Count how many initial keypoints survived (excluding boundary losses)
    survived_count = len(survived_indices_list)
    survival_rate = survived_count / initial_count if initial_count > 0 else 0.0
    
    # Calculate trajectory smoothness
    # Convert trajectories dict to list format for smoothness computation
    trajectory_list = []
    for orig_idx in survived_indices_list:
        traj = keypoint_trajectories[orig_idx]
        if len(traj) >= 2:  # Need at least 2 points for smoothness
            # Convert list of tuples to numpy array with shape (N, 2)
            traj_array = np.array(traj)
            if traj_array.ndim == 1:
                traj_array = traj_array.reshape(-1, 2)
            trajectory_list.append(traj_array)
    
    if len(trajectory_list) == 0:
        logger.warning("Feature stability: no valid trajectories for smoothness computation")
        return None
    
    # Compute smoothness: we need to process each trajectory separately
    smoothness_scores = []
    high_jerk_count = 0
    total_keypoints = 0
    
    for traj in trajectory_list:
        if len(traj) < 3:
            continue
        total_keypoints += 1
        
        # Compute velocity (first derivative)
        velocities = np.diff(traj, axis=0)
        vel_magnitudes = np.linalg.norm(velocities, axis=1)
        
        if len(vel_magnitudes) < 2:
            smoothness_scores.append(1.0)
            continue
        
        # Compute acceleration (second derivative)
        accelerations = np.diff(vel_magnitudes)
        acc_magnitudes = np.abs(accelerations)
        
        # Compute jerk (third derivative) - sudden changes in acceleration
        if len(acc_magnitudes) > 1:
            jerks = np.diff(acc_magnitudes)
            max_jerk = np.max(np.abs(jerks)) if len(jerks) > 0 else 0.0
        else:
            max_jerk = 0.0
        
        # Smoothness: lower acceleration/jerk = smoother
        # Normalize by average velocity to make it scale-invariant
        avg_velocity = np.mean(vel_magnitudes) if len(vel_magnitudes) > 0 else 1.0
        if avg_velocity > 0:
            normalized_acc = np.mean(acc_magnitudes) / (avg_velocity + 1e-6)
            smoothness = 1.0 / (1.0 + normalized_acc * 10.0)  # Map to [0, 1]
        else:
            smoothness = 1.0  # Stationary = perfectly smooth
        
        smoothness_scores.append(smoothness)
        
        # High jerk threshold: sudden non-physical jumps
        # If jerk is very high relative to velocity, it's suspicious
        if avg_velocity > 0 and max_jerk > (avg_velocity * 0.5):
            high_jerk_count += 1
    
    if total_keypoints == 0:
        avg_smoothness = 1.0
        high_jerk_ratio = 0.0
    else:
        avg_smoothness = np.mean(smoothness_scores) if smoothness_scores else 1.0
        high_jerk_ratio = high_jerk_count / total_keypoints
    
    # Generate summary
    has_instability = survival_rate < 0.6 or avg_smoothness < 0.7 or high_jerk_ratio > 0.3
    
    summary_parts = []
    summary_parts.append(f"Feature point tracking analysis:")
    summary_parts.append(f"- Initial keypoints: {initial_count}")
    summary_parts.append(f"- Survival rate over {window_size} frames: {survival_rate:.1%}")
    summary_parts.append(f"- Average trajectory smoothness: {avg_smoothness:.2f} (1.0 = perfectly smooth)")
    summary_parts.append(f"- High-jerk ratio (artificial jitter): {high_jerk_ratio:.1%}")
    
    if has_instability:
        summary_parts.append(f"\n⚠️  Instability detected:")
        if survival_rate < 0.6:
            summary_parts.append(f"  - Low keypoint survival ({survival_rate:.1%} < 60%) suggests pixel-level instability")
        if avg_smoothness < 0.7:
            summary_parts.append(f"  - Low trajectory smoothness ({avg_smoothness:.2f} < 0.70) suggests non-physical motion")
        if high_jerk_ratio > 0.3:
            summary_parts.append(f"  - High jerk ratio ({high_jerk_ratio:.1%} > 30%) indicates artificial jitter/jumps")
    else:
        summary_parts.append(f"\n✓ Feature stability appears normal (survival rate >= 60%, smoothness >= 0.70, jerk ratio <= 30%)")
    
    summary = "\n".join(summary_parts)
    
    return FeatureStabilityAnalysis(
        has_valid_tracking=True,
        initial_keypoints=initial_count,
        survival_rate_10_frames=survival_rate,
        avg_trajectory_smoothness=avg_smoothness,
        high_jerk_ratio=high_jerk_ratio,
        summary=summary
    )
