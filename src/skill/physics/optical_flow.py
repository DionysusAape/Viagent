"""Optical flow analysis skill for detecting motion coherence and background coupling anomalies.

Uses Farneback dense optical flow to detect physical motion violations.
"""
import cv2
import numpy as np
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


def analyze_optical_flow(
    prev_frame: np.ndarray,
    curr_frame: np.ndarray,
    config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Analyze optical flow between two consecutive frames.
    
    Detects:
    1. Motion Coherence: Direction consistency of motion vectors
    2. Background Coupling: Whether background is distorted by foreground motion
    
    Args:
        prev_frame: Previous frame (BGR format)
        curr_frame: Current frame (BGR format)
        config: Optional configuration dict
        
    Returns:
        Dict with analysis results
    """
    # Convert to grayscale
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
    
    # Calculate dense optical flow (Farneback method)
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, curr_gray,
        None,  # flow (output)
        0.5,   # pyr_scale
        3,     # levels
        15,    # winsize
        3,     # iterations
        5,     # poly_n
        1.2,   # poly_sigma
        0      # flags
    )
    
    # Calculate flow magnitude and direction
    magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
    angle = np.arctan2(flow[..., 1], flow[..., 0])
    
    # Parameters from config
    motion_threshold = config.get("motion_threshold", 5.0) if config else 5.0
    coherence_threshold = config.get("coherence_threshold", 0.6) if config else 0.6
    
    h, w = magnitude.shape
    
    # ========================================================================
    # 1. Motion Coherence Analysis
    # ========================================================================
    # Sample key regions (5×5 grid + center)
    sample_regions = []
    for y in [h//4, h//2, 3*h//4]:
        for x in [w//4, w//2, 3*w//4]:
            sample_regions.append((y, x))
    
    # Collect directions from regions with significant motion
    directions = []
    magnitudes_at_samples = []
    for y, x in sample_regions:
        mag = magnitude[y, x]
        if mag > motion_threshold:
            directions.append(angle[y, x])
            magnitudes_at_samples.append(mag)
    
    # Calculate direction consistency (coherence)
    if len(directions) > 1:
        directions = np.array(directions)
        # Normalize to [-pi, pi] and handle circular statistics
        directions = np.arctan2(np.sin(directions), np.cos(directions))
        direction_variance = np.var(directions)
        # Coherence score: higher = more consistent
        coherence_score = 1.0 / (1.0 + direction_variance)
    else:
        # No significant motion or too few samples
        coherence_score = 1.0
        direction_variance = 0.0
    
    is_coherence_anomalous = coherence_score < coherence_threshold
    
    # ========================================================================
    # 2. Background Coupling Analysis
    # ========================================================================
    # Identify background regions:
    # - Low motion magnitude (< threshold)
    # - Edge regions (outer 20% of image)
    edge_margin = 0.2
    y_coords, x_coords = np.ogrid[:h, :w]
    is_edge_y = (y_coords < h * edge_margin) | (y_coords > h * (1 - edge_margin))
    is_edge_x = (x_coords < w * edge_margin) | (x_coords > w * (1 - edge_margin))
    is_edge = is_edge_y | is_edge_x
    
    background_mask = (magnitude < motion_threshold) & is_edge
    foreground_mask = magnitude > motion_threshold * 2
    
    # Calculate background direction consistency
    bg_directions = angle[background_mask]
    if len(bg_directions) > 10:  # Need enough background pixels
        bg_directions = np.arctan2(np.sin(bg_directions), np.cos(bg_directions))
        bg_direction_variance = np.var(bg_directions)
        # Lower variance = more consistent background (good)
        # Higher variance = background is "pulled" by foreground (bad)
        background_coupling_score = 1.0 - min(1.0, bg_direction_variance / 1.0)
    else:
        background_coupling_score = 1.0
        bg_direction_variance = 0.0
    
    # Calculate foreground mean direction
    fg_directions = angle[foreground_mask]
    if len(fg_directions) > 0:
        fg_mean_direction = np.mean(np.arctan2(np.sin(fg_directions), np.cos(fg_directions)))
    else:
        fg_mean_direction = 0.0
    
    is_background_coupled = background_coupling_score < 0.5
    
    # ========================================================================
    # 3. Detect sudden motion changes (non-inertial)
    # ========================================================================
    # Check for sudden direction reversals or accelerations
    if len(magnitudes_at_samples) > 0:
        mean_magnitude = np.mean(magnitudes_at_samples)
        max_magnitude = np.max(magnitudes_at_samples)
        if mean_magnitude > 0:
            magnitude_ratio = max_magnitude / mean_magnitude
            # High ratio suggests sudden jump
            sudden_change_score = min(1.0, (magnitude_ratio - 3.0) / 5.0) if magnitude_ratio > 3.0 else 0.0
        else:
            sudden_change_score = 0.0
    else:
        sudden_change_score = 0.0
    
    is_sudden_change = sudden_change_score > 0.3
    
    # Overall anomaly detection
    is_anomalous = is_coherence_anomalous or is_background_coupled or is_sudden_change
    
    return {
        "coherence_score": float(coherence_score),
        "background_coupling_score": float(background_coupling_score),
        "sudden_change_score": float(sudden_change_score),
        "is_anomalous": is_anomalous,
        "is_coherence_anomalous": is_coherence_anomalous,
        "is_background_coupled": is_background_coupled,
        "is_sudden_change": is_sudden_change,
        "direction_variance": float(direction_variance),
        "bg_direction_variance": float(bg_direction_variance),
        "mean_magnitude": float(np.mean(magnitude)) if len(magnitude) > 0 else 0.0,
        "max_magnitude": float(np.max(magnitude)) if len(magnitude) > 0 else 0.0,
    }


def analyze_frames_optical_flow_batch(
    frame_inputs: List[str],
    config: Optional[Dict[str, Any]] = None
) -> List[Optional[Dict[str, Any]]]:
    """
    Batch analyze optical flow for consecutive frame pairs.
    
    Args:
        frame_inputs: List of base64 data URLs
        config: Optional configuration dict
        
    Returns:
        List of analysis results (None for failed frames)
    """
    if len(frame_inputs) < 2:
        return []
    
    results = []
    
    # Decode first frame
    try:
        prev_frame = decode_data_url_to_cv2(frame_inputs[0])
    except Exception as e:
        results.append(None)
        prev_frame = None
    
    # Analyze consecutive pairs
    for i in range(1, len(frame_inputs)):
        if prev_frame is None:
            results.append(None)
            continue
        
        try:
            curr_frame = decode_data_url_to_cv2(frame_inputs[i])
            analysis = analyze_optical_flow(prev_frame, curr_frame, config)
            results.append(analysis)
            prev_frame = curr_frame
        except Exception as e:
            results.append(None)
            # Try to continue with next frame
            try:
                prev_frame = decode_data_url_to_cv2(frame_inputs[i])
            except:
                prev_frame = None
    
    return results


def format_optical_flow_for_prompt(
    optical_flow_results: List[Optional[Dict[str, Any]]],
    frame_labels: List[str]
) -> str:
    """
    Format optical flow analysis results for LLM prompt injection.
    
    Args:
        optical_flow_results: List of analysis results from analyze_frames_optical_flow_batch
        frame_labels: Frame labels (e.g., ["Frame 1", "Frame 2", ...])
        
    Returns:
        Formatted text description
    """
    lines = [
        "**Optical Flow Analysis Evidence (Algorithm Detection - FOR REFERENCE ONLY):**",
        "",
        "This algorithm analyzes pixel motion vectors between consecutive frames to detect physical motion anomalies.",
        ""
    ]
    
    # Collect anomalous segments
    anomalous_segments = []
    all_coherence_scores = []
    all_coupling_scores = []
    
    for i, result in enumerate(optical_flow_results):
        if result is None:
            continue
        
        all_coherence_scores.append(result.get("coherence_score", 1.0))
        all_coupling_scores.append(result.get("background_coupling_score", 1.0))
        
        if result.get("is_anomalous", False):
            # Determine anomaly type
            anomaly_types = []
            if result.get("is_coherence_anomalous", False):
                anomaly_types.append("motion coherence violation")
            if result.get("is_background_coupled", False):
                anomaly_types.append("background coupling distortion")
            if result.get("is_sudden_change", False):
                anomaly_types.append("sudden non-inertial motion")
            
            anomalous_segments.append({
                "frame_pair": (i, i+1),
                "frame_labels": (frame_labels[i] if i < len(frame_labels) else f"Frame {i+1}",
                                frame_labels[i+1] if i+1 < len(frame_labels) else f"Frame {i+2}"),
                "types": anomaly_types,
                "coherence_score": result.get("coherence_score", 1.0),
                "coupling_score": result.get("background_coupling_score", 1.0),
                "sudden_change_score": result.get("sudden_change_score", 0.0)
            })
    
    # Summary statistics
    if all_coherence_scores:
        mean_coherence = np.mean(all_coherence_scores)
        min_coherence = np.min(all_coherence_scores)
        lines.append(f"**Summary Statistics:**")
        lines.append(f"- Average motion coherence score: {mean_coherence:.3f} (0-1, higher = more coherent)")
        lines.append(f"- Minimum coherence score: {min_coherence:.3f} (threshold: 0.6, lower = more anomalous)")
        lines.append("")
    
    if all_coupling_scores:
        mean_coupling = np.mean(all_coupling_scores)
        min_coupling = np.min(all_coupling_scores)
        lines.append(f"- Average background coupling score: {mean_coupling:.3f} (0-1, higher = less distortion)")
        lines.append(f"- Minimum coupling score: {min_coupling:.3f} (threshold: 0.5, lower = more distortion)")
        lines.append("")
    
    # Anomalous segments
    if anomalous_segments:
        lines.append("**⚠️ Anomalous Motion Detected:**")
        lines.append("")
        for seg in anomalous_segments[:10]:  # Limit to first 10
            frame_start, frame_end = seg["frame_pair"]
            label_start, label_end = seg["frame_labels"]
            types_str = ", ".join(seg["types"])
            
            lines.append(f"- **{label_start} → {label_end}**: {types_str}")
            lines.append(f"  - Coherence score: {seg['coherence_score']:.3f}")
            lines.append(f"  - Background coupling score: {seg['coupling_score']:.3f}")
            if seg['sudden_change_score'] > 0:
                lines.append(f"  - Sudden change score: {seg['sudden_change_score']:.3f}")
            lines.append("")
        
        lines.append("**CRITICAL: Algorithm Interpretation Guidelines:**")
        lines.append("- **Motion coherence violation**: Pixel motion vectors show inconsistent directions, suggesting non-inertial motion (sudden reversals, instant accelerations). This may indicate geometric collapse or texture reshaping in AI-generated videos.")
        lines.append("- **Background coupling distortion**: Background regions show motion inconsistent with their expected behavior, suggesting they are being 'pulled' or distorted by foreground objects. This may indicate compositing artifacts or spatial manipulation.")
        lines.append("- **Sudden non-inertial motion**: Large magnitude changes between frames without intermediate motion, suggesting teleportation-like artifacts.")
        lines.append("")
        lines.append("**You MUST visually verify these algorithm findings.** If algorithm reports anomalies but you visually see normal physics → Trust your visual analysis. If algorithm reports anomalies AND you visually confirm → This strengthens your confidence (double confirmation).")
    else:
        lines.append("✅ **No significant motion anomalies detected by algorithm.**")
        lines.append("")
    
    lines.append("**Remember: Algorithm results are FOR REFERENCE ONLY. Your visual analysis is PRIMARY.**")
    
    return "\n".join(lines)
