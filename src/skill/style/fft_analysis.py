"""2D FFT analysis skill for detecting CG rendering characteristics."""
import base64
import io
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import numpy as np
from PIL import Image


def decode_data_url(data_url: str) -> Image.Image:
    """
    从 base64 data URL 解码图像。
    
    Args:
        data_url: base64 编码的 data URL (格式: "data:image/jpeg;base64,...")
        
    Returns:
        PIL Image 对象
    """
    # 移除 data URL 前缀
    if "," in data_url:
        header, encoded = data_url.split(",", 1)
    else:
        encoded = data_url
    
    # 解码 base64
    image_data = base64.b64decode(encoded)
    image = Image.open(io.BytesIO(image_data))
    return image


def load_image_from_path(image_path: Path) -> Image.Image:
    """
    从文件路径加载图像。
    
    Args:
        image_path: 图像文件路径
        
    Returns:
        PIL Image 对象
    """
    return Image.open(image_path)


def image_to_grayscale_array(image: Image.Image) -> np.ndarray:
    """
    将图像转换为灰度 numpy 数组。
    
    Args:
        image: PIL Image 对象
        
    Returns:
        灰度图像数组 (H, W)
    """
    if image.mode != "L":
        image = image.convert("L")
    return np.array(image, dtype=np.float32)


def compute_fft_features(image_array: np.ndarray) -> Dict[str, float]:
    """
    计算图像的 2D FFT 特征。
    
    Args:
        image_array: 灰度图像数组 (H, W)
        
    Returns:
        包含 FFT 特征的字典
    """
    # 计算 2D FFT
    fft = np.fft.fft2(image_array)
    fft_shifted = np.fft.fftshift(fft)
    magnitude = np.abs(fft_shifted)
    
    # 计算频率坐标
    h, w = image_array.shape
    center_h, center_w = h // 2, w // 2
    
    # 定义频率带（相对于图像中心）
    # 低频：中心区域（0-25%）
    # 中频：中间区域（25-50%）
    # 高频：边缘区域（50-100%）
    low_freq_radius = min(h, w) * 0.25
    mid_freq_radius = min(h, w) * 0.50
    
    # 创建频率带掩码
    y, x = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((x - center_w)**2 + (y - center_h)**2)
    
    low_freq_mask = dist_from_center <= low_freq_radius
    mid_freq_mask = (dist_from_center > low_freq_radius) & (dist_from_center <= mid_freq_radius)
    high_freq_mask = dist_from_center > mid_freq_radius
    
    # 计算各频率带的能量
    total_energy = np.sum(magnitude**2)
    low_freq_energy = np.sum(magnitude[low_freq_mask]**2)
    mid_freq_energy = np.sum(magnitude[mid_freq_mask]**2)
    high_freq_energy = np.sum(magnitude[high_freq_mask]**2)
    
    # 归一化能量比例
    if total_energy > 0:
        low_freq_ratio = low_freq_energy / total_energy
        mid_freq_ratio = mid_freq_energy / total_energy
        high_freq_ratio = high_freq_energy / total_energy
    else:
        low_freq_ratio = mid_freq_ratio = high_freq_ratio = 0.0
    
    # 检测周期性模式（通过查找频率域中的峰值）
    # 计算径向平均功率谱
    radial_profile = []
    max_radius = min(h, w) // 2
    for r in range(1, max_radius):
        mask = (dist_from_center >= r - 0.5) & (dist_from_center < r + 0.5)
        if np.any(mask):
            radial_profile.append(np.mean(magnitude[mask]))
        else:
            radial_profile.append(0.0)
    
    radial_profile = np.array(radial_profile)
    
    # 计算周期性指标（峰值与平均值的比值）
    if len(radial_profile) > 0 and np.mean(radial_profile) > 0:
        peak_value = np.max(radial_profile)
        mean_value = np.mean(radial_profile)
        periodicity_score = peak_value / mean_value if mean_value > 0 else 0.0
    else:
        periodicity_score = 0.0
    
    # 计算边缘频率特征（高频能量集中度）
    # CG 渲染通常有更规则的高频分布
    high_freq_std = np.std(magnitude[high_freq_mask]) if np.any(high_freq_mask) else 0.0
    high_freq_mean = np.mean(magnitude[high_freq_mask]) if np.any(high_freq_mask) else 0.0
    edge_regularity = high_freq_std / high_freq_mean if high_freq_mean > 0 else 0.0
    
    return {
        "low_freq_ratio": float(low_freq_ratio),
        "mid_freq_ratio": float(mid_freq_ratio),
        "high_freq_ratio": float(high_freq_ratio),
        "periodicity_score": float(periodicity_score),
        "edge_regularity": float(edge_regularity),
    }


def analyze_frame_fft(data_url: str) -> Optional[Dict[str, float]]:
    """
    分析单个帧的 FFT 特征。
    
    Args:
        data_url: base64 编码的 data URL
        
    Returns:
        FFT 特征字典，如果失败则返回 None
    """
    try:
        image = decode_data_url(data_url)
        image_array = image_to_grayscale_array(image)
        features = compute_fft_features(image_array)
        return features
    except Exception:
        return None


def analyze_frames_fft_batch(frame_inputs: List[str]) -> List[Optional[Dict[str, float]]]:
    """
    批量分析多个帧的 FFT 特征。
    
    Args:
        frame_inputs: base64 编码的 data URL 列表
        
    Returns:
        FFT 特征字典列表（失败时对应位置为 None）
    """
    results = []
    for frame_input in frame_inputs:
        if frame_input is None:
            results.append(None)
        else:
            features = analyze_frame_fft(frame_input)
            results.append(features)
    return results


def format_fft_features_for_prompt(
    fft_features_list: List[Optional[Dict[str, float]]],
    frame_labels: List[str]
) -> str:
    """
    将 FFT 特征格式化为文本描述，用于添加到 prompt。
    
    Args:
        fft_features_list: FFT 特征字典列表
        frame_labels: 帧标签列表（如 ["Frame 1", "Frame 2", ...]）
        
    Returns:
        格式化的文本描述
    """
    if not fft_features_list or all(f is None for f in fft_features_list):
        return "FFT Analysis: No frequency domain features available."
    
    # 计算平均特征
    valid_features = [f for f in fft_features_list if f is not None]
    if not valid_features:
        return "FFT Analysis: No valid frequency domain features available."
    
    avg_features = {
        "low_freq_ratio": np.mean([f["low_freq_ratio"] for f in valid_features]),
        "mid_freq_ratio": np.mean([f["mid_freq_ratio"] for f in valid_features]),
        "high_freq_ratio": np.mean([f["high_freq_ratio"] for f in valid_features]),
        "periodicity_score": np.mean([f["periodicity_score"] for f in valid_features]),
        "edge_regularity": np.mean([f["edge_regularity"] for f in valid_features]),
    }
    
    # 检测异常帧（可能有周期性模式）
    periodicity_scores = [f["periodicity_score"] for f in valid_features]
    if periodicity_scores:
        max_periodicity = max(periodicity_scores)
        avg_periodicity = np.mean(periodicity_scores)
    else:
        max_periodicity = avg_periodicity = 0.0
    
    # 格式化描述
    lines = [
        "**FFT Frequency Domain Analysis:**",
        f"- Low-frequency energy ratio: {avg_features['low_freq_ratio']:.3f}",
        f"- Mid-frequency energy ratio: {avg_features['mid_freq_ratio']:.3f}",
        f"- High-frequency energy ratio: {avg_features['high_freq_ratio']:.3f}",
        f"- Periodicity score (texture repetition indicator): {avg_features['periodicity_score']:.3f} (avg), {max_periodicity:.3f} (max)",
        f"- Edge regularity (high-frequency distribution): {avg_features['edge_regularity']:.3f}",
    ]
    
    # 添加解释性说明
    lines.append("")
    lines.append("**FFT Interpretation Guidelines:**")
    lines.append("- High periodicity score (>2.0) may indicate texture repetition patterns (common in CG texture mapping)")
    lines.append("- Very low high-frequency ratio (<0.1) may indicate overly smooth rendering (CG characteristic)")
    lines.append("- Very high edge regularity (>1.5) may indicate uniform edge sharpening (CG anti-aliasing)")
    lines.append("- Real camera footage typically shows more random frequency distribution")
    
    return "\n".join(lines)
