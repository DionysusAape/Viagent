"""Formatter for spatial skills analysis results.

Formats algorithm evidence into text description for LLM prompt injection.
"""
from typing import Dict, List, Any


def format_spatial_skills_for_prompt(
    ela_results: Dict[str, Any],
    patch_results: Dict[str, Any],
    boundary_results: Dict[str, Any],
    edge_results: Dict[str, Any],
    multi_face_results: Dict[str, Any],
    blur_results: Dict[str, Any],
    selected_frames: List[Dict[str, Any]],
    frame_labels: List[str],
) -> str:
    """
    Format all spatial skill results into text description for prompt injection.
    
    Args:
        ela_results: ELA analysis results dict
        patch_results: Patch inconsistency analysis results dict
        boundary_results: Boundary anomaly analysis results dict
        edge_results: Edge coherence analysis results dict
        multi_face_results: Multi-face collapse detection results dict
        selected_frames: List of selected frame dicts
        frame_labels: Frame labels (e.g., ["Frame 1", "Frame 2", ...])
        
    Returns:
        Formatted text description
    """
    lines = ["**Algorithm Evidence (Spatial Artifacts Analysis):**"]
    
    # ELA Results
    if ela_results.get("frame_results"):
        lines.append("")
        lines.append("**ELA Boundary Analysis:**")
        summary = ela_results.get("summary", {})
        lines.append(f"- Max halo_score: {summary.get('max_halo_score', 0.0):.2f} (threshold: 1.5)")
        lines.append(f"- Mean boundary_bg_ratio: {summary.get('mean_boundary_bg_ratio', 0.0):.2f}")
        
        anomalous = summary.get("anomalous_frames", [])
        if anomalous:
            frame_indices = [selected_frames[i]["index"] if i < len(selected_frames) else i+1 for i in anomalous[:5]]
            lines.append(f"- Frames with halo detected: {', '.join([f'Frame {idx}' for idx in frame_indices])}")
        
        # Show detailed values for top 3 frames
        frame_results = ela_results["frame_results"]
        valid_results = [r for r in frame_results if r.get("face_detected", False) and "error" not in r]
        if valid_results:
            # Sort by halo_score descending
            sorted_results = sorted(valid_results, key=lambda x: x.get("halo_score", 0.0), reverse=True)
            for fr in sorted_results[:3]:
                idx = fr["frame_index"]
                frame_label = frame_labels[idx] if idx < len(frame_labels) else f"Frame {idx+1}"
                lines.append(f"  - {frame_label}: boundary_ela_mean={fr['boundary_ela_mean']:.1f}, "
                            f"background_ela_mean={fr['background_ela_mean']:.1f}, "
                            f"ratio={fr['boundary_bg_ratio']:.2f}")
    
    # Patch Anomaly Results
    if patch_results.get("frame_results"):
        lines.append("")
        lines.append("**Patch Inconsistency Analysis:**")
        summary = patch_results.get("summary", {})
        lines.append(f"- Max patch_anomaly_score: {summary.get('max_patch_anomaly_score', 0.0):.2f} (threshold: 0.3)")
        
        anomalous = summary.get("frames_with_anomalies", [])
        if anomalous:
            frame_indices = [selected_frames[i]["index"] if i < len(selected_frames) else i+1 for i in anomalous[:5]]
            lines.append(f"- Frames with patch anomalies: {', '.join([f'Frame {idx}' for idx in frame_indices])}")
    
    # Boundary Anomaly Results
    if boundary_results.get("frame_results"):
        lines.append("")
        lines.append("**Boundary Anomaly Analysis:**")
        summary = boundary_results.get("summary", {})
        lines.append(f"- Max edge_melting_score: {summary.get('max_edge_melting', 0.0):.2f} (threshold: 0.4)")
        lines.append(f"- Max hair_boundary_score: {summary.get('max_hair_boundary', 0.0):.2f}")
        lines.append(f"- Max jawline_boundary_score: {summary.get('max_jawline_boundary', 0.0):.2f}")

    # Local Edge Coherence Results
    if edge_results.get("frame_results"):
        lines.append("")
        lines.append("**Local Edge Coherence Analysis (Foreground/Background Boundaries):**")
        summary = edge_results.get("summary", {})
        lines.append(
            f"- Max edge_noise_ratio: {summary.get('max_edge_noise_ratio', 0.0):.2f} "
            "(fraction of edge pixels with very noisy local gradient)"
        )
        lines.append(
            f"- Mean edge_noise_ratio: {summary.get('mean_edge_noise_ratio', 0.0):.2f}"
        )
        lines.append(
            f"- Max soft_edge_ratio: {summary.get('max_soft_edge_ratio', 0.0):.2f} "
            "(neighbor-gradient / edge-gradient; higher = softer/wider edges)"
        )
        lines.append(
            f"- Mean soft_edge_ratio: {summary.get('mean_soft_edge_ratio', 0.0):.2f}"
        )

        anomalous = summary.get("frames_with_soft_noisy_edges", [])
        if anomalous:
            frame_indices = [
                selected_frames[i]["index"] if i < len(selected_frames) else i + 1
                for i in anomalous[:5]
            ]
            lines.append(
                "- Frames with soft/noisy edges (possible boundary bleeding or floating objects): "
                + ", ".join([f"Frame {idx}" for idx in frame_indices])
            )
    
    # Multi-Face Collapse Detection Results
    if multi_face_results.get("frame_results"):
        lines.append("")
        lines.append("**Multi-Face Collapse Detection Analysis:**")
        summary = multi_face_results.get("summary", {})
        
        total_faces = summary.get("total_faces", 0)
        anomalous_faces_count = summary.get("anomalous_faces_count", 0)
        max_anomaly_score = summary.get("max_anomaly_score", 0.0)
        avg_anomaly_score = summary.get("avg_anomaly_score", 0.0)
        
        if total_faces > 0:
            lines.append(f"- Total faces detected across all frames: {total_faces}")
            lines.append(f"- Faces with anomalies detected: {anomalous_faces_count}")
            if total_faces > 0:
                anomalous_ratio = anomalous_faces_count / total_faces
                lines.append(f"- Anomalous faces ratio: {anomalous_ratio:.2%}")
            
            lines.append(f"- Max anomaly_score: {max_anomaly_score:.2f} (threshold: 0.30 for mild, 0.50 for moderate, 0.70 for severe, 0.85 for critical)")
            if anomalous_faces_count > 0:
                lines.append(f"- Avg anomaly_score (for anomalous faces): {avg_anomaly_score:.2f}")
            
            # Count by severity
            critical_count = summary.get("critical_faces_count", 0)
            severe_count = summary.get("severe_faces_count", 0)
            moderate_count = summary.get("moderate_faces_count", 0)
            mild_count = summary.get("mild_faces_count", 0)
            
            if critical_count > 0 or severe_count > 0 or moderate_count > 0 or mild_count > 0:
                lines.append(f"- Severity breakdown: {critical_count} critical, {severe_count} severe, {moderate_count} moderate, {mild_count} mild")
            
            # Show detailed face information for frames with anomalies
            frames_with_anomalies = summary.get("frames_with_anomalies", [])
            if frames_with_anomalies:
                lines.append("")
                lines.append("**Detailed Face Information (for frames with anomalies):**")
                for frame_idx in frames_with_anomalies[:5]:  # Show top 5 frames
                    if frame_idx < len(multi_face_results["frame_results"]):
                        frame_result = multi_face_results["frame_results"][frame_idx]
                        frame_label = frame_labels[frame_idx] if frame_idx < len(frame_labels) else f"Frame {frame_idx+1}"
                        face_details = frame_result.get("face_details", [])
                        
                        anomalous_faces_in_frame = [f for f in face_details if f.get("anomaly_score", 0.0) >= 0.30]
                        if anomalous_faces_in_frame:
                            lines.append(f"- {frame_label}: {len(anomalous_faces_in_frame)} anomalous face(s) detected")
                            for face in anomalous_faces_in_frame[:3]:  # Show top 3 faces per frame
                                face_id = face.get("face_id", 0)
                                bbox = face.get("bbox", (0, 0, 0, 0))
                                anomaly_score = face.get("anomaly_score", 0.0)
                                severity = face.get("severity", "normal")
                                anomaly_types = face.get("anomaly_types", [])
                                details = face.get("details", "")
                                
                                lines.append(f"  * Face {face_id} (bbox: {bbox}): anomaly_score={anomaly_score:.2f}, severity={severity}, types={anomaly_types}, details=\"{details}\"")
        else:
            lines.append("- No faces detected in analyzed frames")

    # Blur Uniformity Results (Laplacian variance)
    if blur_results.get("frame_results"):
        lines.append("")
        lines.append("**Blur Uniformity Analysis (Laplacian Variance; Face vs Background):**")
        summary = blur_results.get("summary", {})
        thr = summary.get("selective_ratio_threshold", 0.0)
        frames_sel = summary.get("frames_with_selective_face_blur", [])
        persistent = summary.get("selective_face_blur_persistent", False)
        lines.append(f"- selective_ratio_threshold (bg_var / face_var): {thr:.2f} (higher = face blur more selective)")
        lines.append(f"- selective_face_blur_persistent: {persistent}")
        if frames_sel:
            frame_indices = [selected_frames[i]["index"] if i < len(selected_frames) else i + 1 for i in frames_sel[:5]]
            lines.append("- Frames with suspected selective face blur: " + ", ".join([f"Frame {idx}" for idx in frame_indices]))

        # Show top 3 ratios
        frs = blur_results.get("frame_results") or []
        valid = [r for r in frs if r.get("faces_pixels", 0) > 0]
        if valid:
            sorted_results = sorted(valid, key=lambda x: x.get("bg_to_face_var_ratio", 0.0), reverse=True)
            for r in sorted_results[:3]:
                idx = r.get("frame_index", 0)
                frame_label = frame_labels[idx] if idx < len(frame_labels) else f"Frame {idx+1}"
                lines.append(
                    f"  - {frame_label}: global_var={r.get('global_lap_var', 0.0):.1f}, "
                    f"face_var={r.get('face_lap_var', 0.0):.1f}, bg_var={r.get('bg_lap_var', 0.0):.1f}, "
                    f"ratio(bg/face)={r.get('bg_to_face_var_ratio', 0.0):.2f}"
                )

    # Add interpretation guidelines
    lines.append("")
    lines.append("**Algorithm Interpretation Guidelines:**")
    lines.append("- halo_score > 1.5 (boundary_bg_ratio) indicates abnormal edge halo, supports face_boundary_halo evidence")
    lines.append("- patch_anomaly_score > 0.3 indicates inconsistent texture/sharpness within face, supports local_patch_inconsistency evidence")
    lines.append("- edge_melting_score > 0.4 indicates abnormal face-to-background transition, supports edge_melting evidence")
    lines.append("- High edge_noise_ratio (> 0.5) and high soft_edge_ratio (> 0.6) indicate soft, noisy edges that may reflect boundary bleeding or floating foreground objects over the background")
    lines.append("- Multi-face collapse detection: anomaly_score >= 0.85 (critical), >= 0.70 (severe), >= 0.50 (moderate), >= 0.30 (mild)")
    lines.append("- Blur uniformity (Laplacian variance): if background is much sharper than face across multiple frames (high bg/face ratio), this suggests selective semantic blur (suspicious). If face and background are similarly blurred (ratio ~ 1), this suggests global blur/compression (often normal).")
    lines.append("- Use these quantitative metrics as SUPPORTING evidence alongside your visual analysis")
    lines.append("- If algorithm reports high scores AND you visually confirm the artifact → STRENGTHEN the evidence")
    lines.append("- If algorithm reports high scores but you visually see normal effects → trust your visual analysis")
    
    return "\n".join(lines)
