from __future__ import annotations
import math
import base64
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np


@dataclass
class NSGLiteConfig:
    num_frames: int = 16
    target_height: int = 256
    target_width: int = 256
    eps: float = 1e-6
    ncs_alpha: float = 5.0
    anomaly_z_threshold: float = 2.0


class NSGLiteVideoAnalyzer:
    def __init__(self, config: Optional[NSGLiteConfig] = None) -> None:
        self.config = config or NSGLiteConfig()

    def analyze_video(self, video_path: str) -> Dict[str, Any]:
        frames = self._sample_and_preprocess_frames(video_path)
        if len(frames) < 2:
            return {
                "ncs_score": 100.0,
                "decision": "Real",
                "physical_metrics": {
                    "spatial_gradient_stability": "high",
                    "temporal_coherence": "high",
                    "flow_conservation_deviation": 0.0,
                },
                "anomaly_hints": [],
                "llm_guidance": (
                    "Video has too few frames for temporal NSG-lite analysis; "
                    "no clear physical inconsistencies detected."
                ),
            }

        mean_abs_residuals, gradient_magnitudes, flow_mag_means = (
            self._compute_flow_and_residuals(frames)
        )

        (
            ncs_score,
            flow_conservation_deviation,
            spatial_grad_level,
            temporal_coherence_level,
            anomaly_indices,
            guidance,
        ) = self._aggregate_metrics(
            mean_abs_residuals, gradient_magnitudes, flow_mag_means
        )

        decision = self._decision_from_ncs_and_temporal(
            ncs_score, temporal_coherence_level, len(anomaly_indices)
        )

        return {
            "ncs_score": float(ncs_score),
            "decision": decision,
            "physical_metrics": {
                "spatial_gradient_stability": spatial_grad_level,
                "temporal_coherence": temporal_coherence_level,
                "flow_conservation_deviation": float(flow_conservation_deviation),
            },
            "anomaly_hints": anomaly_indices,
            "llm_guidance": guidance,
        }

    def analyze_frames_base64(self, frame_inputs: List[str]) -> Dict[str, Any]:
        frames = self._decode_and_preprocess_base64_frames(frame_inputs)
        if len(frames) < 2:
            return {
                "ncs_score": 100.0,
                "decision": "Real",
                "physical_metrics": {
                    "spatial_gradient_stability": "high",
                    "temporal_coherence": "high",
                    "flow_conservation_deviation": 0.0,
                },
                "anomaly_hints": [],
                "llm_guidance": (
                    "NSG-lite analysis skipped or inconclusive due to insufficient frames; "
                    "no clear physical inconsistencies detected."
                ),
            }

        mean_abs_residuals, gradient_magnitudes, flow_mag_means = (
            self._compute_flow_and_residuals(frames)
        )

        (
            ncs_score,
            flow_conservation_deviation,
            spatial_grad_level,
            temporal_coherence_level,
            anomaly_indices,
            guidance,
        ) = self._aggregate_metrics(
            mean_abs_residuals, gradient_magnitudes, flow_mag_means
        )

        decision = self._decision_from_ncs_and_temporal(
            ncs_score, temporal_coherence_level, len(anomaly_indices)
        )

        return {
            "ncs_score": float(ncs_score),
            "decision": decision,
            "physical_metrics": {
                "spatial_gradient_stability": spatial_grad_level,
                "temporal_coherence": temporal_coherence_level,
                "flow_conservation_deviation": float(flow_conservation_deviation),
            },
            "anomaly_hints": anomaly_indices,
            "llm_guidance": guidance,
        }

    def _sample_and_preprocess_frames(self, video_path: str) -> List[np.ndarray]:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return []

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if frame_count <= 0:
            cap.release()
            return []

        num_samples = min(self.config.num_frames, frame_count)
        if num_samples <= 0:
            cap.release()
            return []

        indices = self._uniform_indices(frame_count, num_samples)
        frames: List[np.ndarray] = []
        current_index = 0
        target_set = set(indices)

        success, frame = cap.read()
        while success and current_index <= indices[-1]:
            if current_index in target_set:
                processed = self._preprocess_frame(frame)
                if processed is not None:
                    frames.append(processed)
            success, frame = cap.read()
            current_index += 1

        cap.release()
        return frames

    def _preprocess_frame(self, frame_bgr: np.ndarray) -> Optional[np.ndarray]:
        try:
            h, w = self.config.target_height, self.config.target_width
            resized = cv2.resize(frame_bgr, (w, h), interpolation=cv2.INTER_AREA)
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            gray_f = gray.astype(np.float32) / 255.0
            return gray_f
        except Exception:
            return None

    def _decode_and_preprocess_base64_frames(
        self, frame_inputs: List[str]
    ) -> List[np.ndarray]:
        frames: List[np.ndarray] = []
        total = len(frame_inputs)
        if total == 0:
            return frames

        if total <= self.config.num_frames:
            indices = list(range(total))
        else:
            indices = self._uniform_indices(total, self.config.num_frames)

        for idx in indices:
            data_url = frame_inputs[idx]
            if not data_url:
                continue
            bgr = self._decode_data_url_to_bgr(data_url)
            if bgr is None:
                continue
            gray = self._preprocess_frame(bgr)
            if gray is not None:
                frames.append(gray)
        return frames

    @staticmethod
    def _decode_data_url_to_bgr(data_url: str) -> Optional[np.ndarray]:
        try:
            if "," in data_url:
                _, encoded = data_url.split(",", 1)
            else:
                encoded = data_url
            img_bytes = base64.b64decode(encoded)
            nparr = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None:
                return None
            return img
        except Exception:
            return None

    @staticmethod
    def _uniform_indices(total: int, num: int) -> List[int]:
        if num >= total:
            return list(range(total))
        step = total / float(num)
        idx = [int(i * step) for i in range(num)]
        idx[-1] = total - 1
        # Ensure sorted and unique
        idx_unique = sorted(set(idx))
        return idx_unique

    def _compute_flow_and_residuals(
        self, frames: List[np.ndarray]
    ) -> Tuple[List[float], List[float], List[float]]:
        num = len(frames)
        mean_abs_residuals: List[float] = []
        gradient_magnitudes: List[float] = []
        flow_mag_means: List[float] = []

        # Pre-compute spatial gradients for each frame
        grads_x: List[np.ndarray] = []
        grads_y: List[np.ndarray] = []
        for frame in frames:
            gx = cv2.Sobel(frame, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(frame, cv2.CV_32F, 0, 1, ksize=3)
            grads_x.append(gx)
            grads_y.append(gy)
            grad_mag = np.sqrt(gx * gx + gy * gy)
            gradient_magnitudes.append(float(np.mean(grad_mag)))

        for t in range(num - 1):
            I_t = frames[t]
            I_next = frames[t + 1]
            gx = grads_x[t]
            gy = grads_y[t]

            flow = cv2.calcOpticalFlowFarneback(
                I_t,
                I_next,
                None,
                pyr_scale=0.5,
                levels=3,
                winsize=15,
                iterations=3,
                poly_n=5,
                poly_sigma=1.2,
                flags=0,
            )
            u = flow[..., 0]
            v = flow[..., 1]

            flow_mag = np.sqrt(u * u + v * v)
            flow_mag_means.append(float(np.mean(flow_mag)))

            delta_I = I_next - I_t

            residual = delta_I + gx * u + gy * v
            mean_abs_residuals.append(float(np.mean(np.abs(residual))))

        return mean_abs_residuals, gradient_magnitudes, flow_mag_means

    def _aggregate_metrics(
        self,
        mean_abs_residuals: List[float],
        gradient_magnitudes: List[float],
        flow_mag_means: List[float],
    ) -> Tuple[float, float, str, str, List[int], str]:
        cfg = self.config

        residual_arr = np.array(mean_abs_residuals, dtype=np.float32)
        flow_conservation_deviation = float(np.mean(residual_arr)) if residual_arr.size > 0 else 0.0

        # NCS-lite: higher when residual is small
        # NCS = 100 * exp(-alpha * R_mean^2)
        R_mean = flow_conservation_deviation
        ncs_score = 100.0 * math.exp(-cfg.ncs_alpha * float(R_mean * R_mean))
        ncs_score = max(0.0, min(100.0, ncs_score))

        # Spatial gradient stability
        grad_arr = np.array(gradient_magnitudes, dtype=np.float32)
        if grad_arr.size > 1:
            grad_mean = float(np.mean(grad_arr))
            grad_std = float(np.std(grad_arr))
            grad_ratio = grad_std / (grad_mean + cfg.eps)
        else:
            grad_ratio = 0.0

        if grad_ratio < 0.1:
            spatial_grad_level = "high"
        elif grad_ratio < 0.3:
            spatial_grad_level = "medium"
        else:
            spatial_grad_level = "low"

        # Temporal coherence from flow magnitude variability
        flow_arr = np.array(flow_mag_means, dtype=np.float32)
        if flow_arr.size > 1:
            flow_mean = float(np.mean(flow_arr))
            flow_std = float(np.std(flow_arr))
            cv_mag = flow_std / (flow_mean + cfg.eps)
        else:
            cv_mag = 0.0

        if cv_mag < 0.15:
            temporal_level = "high"
        elif cv_mag < 0.35:
            temporal_level = "medium"
        else:
            temporal_level = "low"

        # Anomaly frames: residual z-score > threshold
        anomaly_indices: List[int] = []
        if residual_arr.size > 1:
            r_mean = float(np.mean(residual_arr))
            r_std = float(np.std(residual_arr)) + cfg.eps
            z_scores = (residual_arr - r_mean) / r_std
            for step_idx, z in enumerate(z_scores):
                if z > cfg.anomaly_z_threshold:
                    anomaly_indices.append(step_idx + 1)

        guidance = self._build_guidance(
            ncs_score,
            spatial_grad_level,
            temporal_level,
            flow_conservation_deviation,
            anomaly_indices,
        )

        return (
            ncs_score,
            flow_conservation_deviation,
            spatial_grad_level,
            temporal_level,
            anomaly_indices,
            guidance,
        )

    @staticmethod
    def _decision_from_ncs_and_temporal(
        ncs_score: float,
        temporal_level: str,
        num_anomalies: int,
    ) -> str:
        if ncs_score >= 80.0 and temporal_level == "high" and num_anomalies == 0:
            return "Real"
        if ncs_score < 50.0 and (temporal_level == "low" or num_anomalies >= 2):
            return "Fake"
        return "Suspicious"

    @staticmethod
    def _build_guidance(
        ncs_score: float,
        spatial_level: str,
        temporal_level: str,
        flow_dev: float,
        anomaly_indices: List[int],
    ) -> str:
        parts: List[str] = []

        if ncs_score >= 80.0:
            parts.append(
                "NSG-lite score is high, suggesting globally consistent physical motion and brightness evolution."
            )
        elif ncs_score >= 50.0:
            parts.append(
                "NSG-lite score is moderate, indicating some deviations from ideal physical consistency."
            )
        else:
            parts.append(
                "NSG-lite score is low, indicating significant deviations from physically consistent motion and brightness evolution."
            )

        if temporal_level == "low":
            parts.append(
                "Temporal coherence of the optical flow field is low, suggesting unstable or non-physical motion patterns."
            )
        elif temporal_level == "medium":
            parts.append(
                "Temporal coherence of the optical flow field is moderate, with some variability in motion magnitude."
            )
        else:
            parts.append(
                "Temporal coherence of the optical flow field is high, with stable motion magnitude across frames."
            )

        if spatial_level == "low":
            parts.append(
                "Spatial gradient stability is low, which may reflect frequent re-rendering or structural instability in the scene."
            )

        if flow_dev > 0.02:
            parts.append(
                f"Average flow-conservation residual is {flow_dev:.3f}, which may indicate violations of brightness constancy."
            )

        if anomaly_indices:
            parts.append(
                f"Frames {sorted(set(anomaly_indices))} show significantly higher residuals and may be physically anomalous."
            )

        if not parts:
            return (
                "NSG-lite analysis did not find clear physical inconsistencies; "
                "treat this as neutral evidence."
            )

        return " ".join(parts)


def format_nsg_lite_for_prompt(
    nsg_result: Dict[str, Any],
    frame_labels: List[str],
) -> str:
    lines: List[str] = []
    lines.append("**NSG-lite Physical Consistency Evidence (Algorithm Detection - FOR REFERENCE ONLY):**")
    lines.append("")

    ncs_score = float(nsg_result.get("ncs_score", 0.0))
    decision = str(nsg_result.get("decision", "Suspicious"))
    physical_metrics = nsg_result.get("physical_metrics", {})
    anomaly_hints = nsg_result.get("anomaly_hints", [])
    guidance = str(nsg_result.get("llm_guidance", "")).strip()

    spatial_level = str(physical_metrics.get("spatial_gradient_stability", "unknown"))
    temporal_level = str(physical_metrics.get("temporal_coherence", "unknown"))
    flow_dev = float(physical_metrics.get("flow_conservation_deviation", 0.0))

    lines.append(f"- NSG-lite NCS score (0-100): **{ncs_score:.1f}** (higher = more physically consistent)")
    lines.append(f"- NSG-lite decision: **{decision}**")
    lines.append(f"- Spatial gradient stability: **{spatial_level}**")
    lines.append(f"- Temporal coherence: **{temporal_level}**")
    lines.append(f"- Flow-conservation deviation (mean residual): **{flow_dev:.4f}**")
    lines.append("")

    if anomaly_hints:
        # Map anomaly indices (0-based) to frame labels if possible
        mapped_labels: List[str] = []
        for idx in anomaly_hints:
            if 0 <= idx < len(frame_labels):
                mapped_labels.append(frame_labels[idx])
            else:
                mapped_labels.append(f"Frame {idx + 1}")
        lines.append(
            "- Frames with unusually high brightness-constancy residuals "
            "(potential physical anomalies): " + ", ".join(mapped_labels)
        )
        lines.append("")

    if guidance:
        lines.append("**Summary for LLM:**")
        lines.append(guidance)
        lines.append("")

    lines.append(
        "You should treat NSG-lite metrics as **supporting physical-consistency signals**. "
        "Low NCS scores, low temporal coherence, and high flow-conservation deviation suggest "
        "non-physical motion evolution, but you MUST visually verify whether these correspond "
        "to real physical violations or are explained by normal motion blur, compression, or editing."
    )

    return "\n".join(lines)


__all__ = ["NSGLiteVideoAnalyzer", "NSGLiteConfig", "format_nsg_lite_for_prompt"]


