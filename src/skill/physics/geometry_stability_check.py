"""Geometry stability check skill for detecting background line curvature during camera movement.

Detects if background straight lines (e.g., edges, boundaries, architectural lines) 
bend or curve unnaturally as the camera moves, which may indicate AI generation artifacts.
"""
import cv2
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
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


def detect_lines(
    frame: np.ndarray,
    config: Optional[Dict[str, Any]] = None
) -> List[Tuple[float, float, float, float]]:
    """
    Detect straight lines in the frame using HoughLinesP.
    
    Args:
        frame: Frame image (BGR format)
        config: Optional configuration dict
        
    Returns:
        List of line segments as (x1, y1, x2, y2) tuples
    """
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Edge detection
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    
    # Parameters from config
    min_line_length = config.get("min_line_length", 50) if config else 50
    max_line_gap = config.get("max_line_gap", 10) if config else 10
    threshold = config.get("hough_threshold", 80) if config else 80
    
    # Detect lines using Probabilistic Hough Transform
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=threshold,
        minLineLength=min_line_length,
        maxLineGap=max_line_gap
    )
    
    if lines is None:
        return []
    
    # Convert to list of tuples
    line_segments = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        line_segments.append((float(x1), float(y1), float(x2), float(y2)))
    
    return line_segments


def calculate_line_curvature(
    line: Tuple[float, float, float, float],
    prev_line: Optional[Tuple[float, float, float, float]] = None
) -> float:
    """
    Calculate curvature metric for a line segment.
    
    For a single line, curvature is 0 (straight line).
    For tracking across frames, we check if the line maintains its straightness.
    
    Args:
        line: Current line segment (x1, y1, x2, y2)
        prev_line: Previous frame's corresponding line (if tracking)
        
    Returns:
        Curvature score (0 = perfectly straight, higher = more curved)
    """
    x1, y1, x2, y2 = line
    
    # Calculate line length
    length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    if length < 1e-6:
        return 0.0
    
    # For a single line, we can't measure curvature directly
    # We'll use this in the tracking function
    return 0.0


def track_lines_across_frames(
    prev_lines: List[Tuple[float, float, float, float]],
    curr_lines: List[Tuple[float, float, float, float]],
    config: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """
    Track lines across consecutive frames and detect curvature changes.
    
    Args:
        prev_lines: Lines from previous frame
        curr_lines: Lines from current frame
        config: Optional configuration dict
        
    Returns:
        List of tracked line pairs with curvature metrics
    """
    if not prev_lines or not curr_lines:
        return []
    
    # Matching threshold (distance to consider lines as "same")
    match_threshold = config.get("line_match_threshold", 30.0) if config else 30.0
    
    tracked_pairs = []
    
    for prev_line in prev_lines:
        px1, py1, px2, py2 = prev_line
        prev_center = ((px1 + px2) / 2, (py1 + py2) / 2)
        prev_angle = np.arctan2(py2 - py1, px2 - px1)
        prev_length = np.sqrt((px2 - px1)**2 + (py2 - py1)**2)
        
        best_match = None
        best_distance = float('inf')
        
        for curr_line in curr_lines:
            cx1, cy1, cx2, cy2 = curr_line
            curr_center = ((cx1 + cx2) / 2, (cy1 + cy2) / 2)
            curr_angle = np.arctan2(cy2 - cy1, cx2 - cx1)
            curr_length = np.sqrt((cx2 - cx1)**2 + (cy2 - cy1)**2)
            
            # Calculate distance between line centers
            center_distance = np.sqrt(
                (curr_center[0] - prev_center[0])**2 + 
                (curr_center[1] - prev_center[1])**2
            )
            
            # Calculate angle difference (normalized to [-pi, pi])
            angle_diff = np.abs(np.arctan2(
                np.sin(curr_angle - prev_angle),
                np.cos(curr_angle - prev_angle)
            ))
            
            # Calculate length similarity
            length_ratio = min(prev_length, curr_length) / max(prev_length, curr_length) if max(prev_length, curr_length) > 0 else 0.0
            
            # Combined matching score (lower is better)
            # Prioritize center distance, then angle similarity, then length similarity
            match_score = center_distance + (angle_diff * 50) + ((1.0 - length_ratio) * 100)
            
            if match_score < best_distance:
                best_distance = match_score
                best_match = (curr_line, center_distance, angle_diff, length_ratio)
        
        if best_match and best_distance < match_threshold:
            curr_line, center_dist, angle_diff, length_ratio = best_match
            tracked_pairs.append({
                "prev_line": prev_line,
                "curr_line": curr_line,
                "center_distance": center_dist,
                "angle_difference": angle_diff,
                "length_ratio": length_ratio,
                "match_score": best_distance
            })
    
    return tracked_pairs


def analyze_geometry_stability(
    prev_frame: np.ndarray,
    curr_frame: np.ndarray,
    config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Analyze geometry stability between two consecutive frames.
    
    Detects if background straight lines bend or curve unnaturally during camera movement.
    
    Args:
        prev_frame: Previous frame (BGR format)
        curr_frame: Current frame (BGR format)
        config: Optional configuration dict
        
    Returns:
        Dict with analysis results
    """
    # Detect lines in both frames
    prev_lines = detect_lines(prev_frame, config)
    curr_lines = detect_lines(curr_frame, config)
    
    if len(prev_lines) < 3 or len(curr_lines) < 3:
        # Not enough lines to analyze
        return {
            "prev_line_count": len(prev_lines),
            "curr_line_count": len(curr_lines),
            "tracked_pairs": 0,
            "curvature_anomaly_score": 0.0,
            "is_anomalous": False,
            "anomaly_details": []
        }
    
    # Track lines across frames
    tracked_pairs = track_lines_across_frames(prev_lines, curr_lines, config)
    
    if not tracked_pairs:
        return {
            "prev_line_count": len(prev_lines),
            "curr_line_count": len(curr_lines),
            "tracked_pairs": 0,
            "curvature_anomaly_score": 0.0,
            "is_anomalous": False,
            "anomaly_details": []
        }
    
    # Analyze curvature anomalies
    # In real camera movement, lines should:
    # 1. Move according to perspective transformation (translation + rotation + scale)
    # 2. Maintain their straightness (unless there's lens distortion, which should be global)
    # 3. Have consistent angle changes (similar lines should have similar angle changes)
    
    # Calculate angle differences for all tracked pairs
    angle_diffs = [pair["angle_difference"] for pair in tracked_pairs]
    center_dists = [pair["center_distance"] for pair in tracked_pairs]
    
    if len(angle_diffs) < 2:
        # Need at least 2 tracked pairs to detect anomalies
        return {
            "prev_line_count": len(prev_lines),
            "curr_line_count": len(curr_lines),
            "tracked_pairs": len(tracked_pairs),
            "curvature_anomaly_score": 0.0,
            "is_anomalous": False,
            "anomaly_details": []
        }
    
    # Check for inconsistent angle changes
    # If camera moves, all lines should rotate similarly (unless they're at different depths)
    # High variance in angle differences suggests non-physical deformation
    angle_variance = np.var(angle_diffs)
    mean_angle_diff = np.mean(angle_diffs)
    
    # Check for lines that change angle too much relative to their movement
    # In real camera movement, lines that move more should rotate more (perspective)
    # But if lines rotate without corresponding movement, that's suspicious
    anomalous_pairs = []
    for pair in tracked_pairs:
        angle_diff = pair["angle_difference"]
        center_dist = pair["center_distance"]
        
        # If line rotates significantly but doesn't move much, that's suspicious
        # (unless it's a rotation-only camera movement, but that's less common)
        if angle_diff > 0.1 and center_dist < 20:  # Rotated > 5.7 degrees but moved < 20 pixels
            anomalous_pairs.append({
                "prev_line": pair["prev_line"],
                "curr_line": pair["curr_line"],
                "angle_difference": angle_diff,
                "center_distance": center_dist,
                "anomaly_type": "rotation_without_movement"
            })
        
        # If line moves a lot but angle changes inconsistently with other lines
        if center_dist > 30 and abs(angle_diff - mean_angle_diff) > 0.2:  # Moved > 30 pixels but angle change differs > 11.5 degrees from mean
            anomalous_pairs.append({
                "prev_line": pair["prev_line"],
                "curr_line": pair["curr_line"],
                "angle_difference": angle_diff,
                "center_distance": center_dist,
                "anomaly_type": "inconsistent_angle_change"
            })
    
    # Calculate overall anomaly score
    # Higher variance in angle differences = more suspicious
    # More anomalous pairs = more suspicious
    curvature_anomaly_score = min(1.0, (angle_variance * 10) + (len(anomalous_pairs) / max(len(tracked_pairs), 1) * 0.5))
    
    # Threshold for anomaly detection
    anomaly_threshold = config.get("curvature_anomaly_threshold", 0.3) if config else 0.3
    is_anomalous = curvature_anomaly_score > anomaly_threshold or len(anomalous_pairs) > 0
    
    return {
        "prev_line_count": len(prev_lines),
        "curr_line_count": len(curr_lines),
        "tracked_pairs": len(tracked_pairs),
        "curvature_anomaly_score": float(curvature_anomaly_score),
        "angle_variance": float(angle_variance),
        "mean_angle_diff": float(mean_angle_diff),
        "anomalous_pairs_count": len(anomalous_pairs),
        "is_anomalous": is_anomalous,
        "anomaly_details": anomalous_pairs[:5]  # Limit to first 5 for reporting
    }


def analyze_frames_geometry_stability_batch(
    frame_inputs: List[str],
    config: Optional[Dict[str, Any]] = None
) -> List[Optional[Dict[str, Any]]]:
    """
    Batch analyze geometry stability for consecutive frame pairs.
    
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
            analysis = analyze_geometry_stability(prev_frame, curr_frame, config)
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


def format_geometry_stability_for_prompt(
    geometry_results: List[Optional[Dict[str, Any]]],
    frame_labels: List[str]
) -> str:
    """
    Format geometry stability analysis results for LLM prompt injection.
    
    Args:
        geometry_results: List of analysis results from analyze_frames_geometry_stability_batch
        frame_labels: Frame labels (e.g., ["Frame 1", "Frame 2", ...])
        
    Returns:
        Formatted text description
    """
    lines = [
        "**Geometry Stability Analysis Evidence (Algorithm Detection - FOR REFERENCE ONLY):**",
        "",
        "This algorithm detects if background straight lines (edges, boundaries, architectural lines) "
        "bend or curve unnaturally as the camera moves, which may indicate AI generation artifacts.",
        ""
    ]
    
    # Collect anomalous segments
    anomalous_segments = []
    all_anomaly_scores = []
    all_angle_variances = []
    
    for i, result in enumerate(geometry_results):
        if result is None:
            continue
        
        anomaly_score = result.get("curvature_anomaly_score", 0.0)
        angle_variance = result.get("angle_variance", 0.0)
        
        all_anomaly_scores.append(anomaly_score)
        all_angle_variances.append(angle_variance)
        
        if result.get("is_anomalous", False):
            anomalous_segments.append({
                "frame_pair": (i, i+1),
                "frame_labels": (frame_labels[i] if i < len(frame_labels) else f"Frame {i+1}",
                                frame_labels[i+1] if i+1 < len(frame_labels) else f"Frame {i+2}"),
                "anomaly_score": anomaly_score,
                "angle_variance": angle_variance,
                "anomalous_pairs": result.get("anomalous_pairs_count", 0),
                "tracked_pairs": result.get("tracked_pairs", 0),
                "details": result.get("anomaly_details", [])
            })
    
    # Summary statistics
    if all_anomaly_scores:
        mean_anomaly = np.mean(all_anomaly_scores)
        max_anomaly = np.max(all_anomaly_scores)
        lines.append(f"**Summary Statistics:**")
        lines.append(f"- Average curvature anomaly score: {mean_anomaly:.3f} (0-1, higher = more anomalous)")
        lines.append(f"- Maximum anomaly score: {max_anomaly:.3f} (threshold: 0.3, higher = more anomalous)")
        lines.append("")
    
    if all_angle_variances:
        mean_variance = np.mean(all_angle_variances)
        max_variance = np.max(all_angle_variances)
        lines.append(f"- Average angle variance: {mean_variance:.3f} (higher = more inconsistent line rotations)")
        lines.append(f"- Maximum angle variance: {max_variance:.3f}")
        lines.append("")
    
    # Anomalous segments
    if anomalous_segments:
        lines.append("**Geometry Stability Anomalies Detected:**")
        lines.append("")
        for seg in anomalous_segments[:10]:  # Limit to first 10
            frame_start, frame_end = seg["frame_pair"]
            label_start, label_end = seg["frame_labels"]
            
            lines.append(f"- **{label_start} → {label_end}**: Background line curvature anomaly detected")
            lines.append(f"  - Curvature anomaly score: {seg['anomaly_score']:.3f}")
            lines.append(f"  - Angle variance: {seg['angle_variance']:.3f}")
            lines.append(f"  - Anomalous line pairs: {seg['anomalous_pairs']}/{seg['tracked_pairs']}")
            
            # Add details about specific anomalies
            if seg['details']:
                for detail in seg['details'][:3]:  # Limit to first 3 details
                    anomaly_type = detail.get("anomaly_type", "unknown")
                    angle_diff = detail.get("angle_difference", 0.0)
                    center_dist = detail.get("center_distance", 0.0)
                    lines.append(f"    - {anomaly_type}: angle_diff={angle_diff:.3f}, center_dist={center_dist:.1f}")
            lines.append("")
        
        lines.append("**CRITICAL: Algorithm Interpretation Guidelines:**")
        lines.append("- **Background line curvature anomaly**: Background straight lines (edges, boundaries, architectural lines) bend or curve unnaturally as the camera moves. In real camera footage, lines should maintain their straightness or follow perspective transformation consistently. Unnatural curvature suggests AI generation artifacts or spatial manipulation.")
        lines.append("- **Rotation without movement**: Lines rotate significantly without corresponding movement, suggesting non-physical deformation.")
        lines.append("- **Inconsistent angle changes**: Lines move but their angle changes are inconsistent with other lines, suggesting non-uniform transformation (not consistent with real camera movement).")
        lines.append("")
        lines.append("**CRITICAL: You MUST distinguish transition effects from AI generation artifacts:**")
        lines.append("- **Transition effects (NOT fake evidence)**: Video editing transitions (fade, wipe, dissolve, morph, zoom) can cause line curvature that is localized, directional, structured, and shows deliberate video editing characteristics. These are NOT fake evidence.")
        lines.append("- **AI generation artifacts (fake evidence)**: Line curvature that is global, irregular, inconsistent with camera movement, and shows structural corruption. This is suspicious, but ONLY if content appears intended as photorealistic live-action or AI-generated.")
        lines.append("- **Content type matters**: Only flag geometry stability anomalies as fake when content appears intended as photorealistic live-action or AI-generated. Traditional animation/CG/game/commercial ad content may have artistic line curvature effects that are NOT fake evidence.")
        lines.append("")
        lines.append("**You MUST visually verify these algorithm findings.** Check if background lines (edges, boundaries, architectural features) appear to bend or curve unnaturally as the camera moves. If algorithm reports anomalies AND you visually confirm unnatural line curvature (NOT transition effect) AND content appears intended as photorealistic live-action or AI-generated → This is medium-strong evidence (score_fake >= 0.65). If algorithm reports anomalies but you visually see transition effects, normal line behavior, or content is clearly traditional animation/CG/game/commercial ad → Trust your visual analysis, do NOT flag as fake.")
    else:
        lines.append("**No significant geometry stability anomalies detected by algorithm.**")
        lines.append("")
    
    lines.append("**Remember: Algorithm results are FOR REFERENCE ONLY. Your visual analysis is PRIMARY.**")
    
    return "\n".join(lines)
