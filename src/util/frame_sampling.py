"""Frame sampling utilities for LLM analysis agents."""
from typing import List


def sample_frames_for_llm(images: List[str], max_images: int) -> List[str]:
    """
    从所有帧中均匀采样，选出不超过 max_images 张代表性帧（覆盖前中后）。
    
    采样策略：
    - 如果帧数 <= max_images，返回所有帧
    - 否则，均匀采样 max_images 张帧，确保最后一帧被包含
    
    Args:
        images: 所有有效帧的列表（base64 编码的图片路径）
        max_images: 最大允许的图片数量（从配置的 llm.max_images 读取）
        
    Returns:
        采样后的帧列表
    """
    if not images:
        return images
    total = len(images)
    if max_images <= 0 or total <= max_images:
        return images

    step = total / float(max_images)
    indices = [int(i * step) for i in range(max_images)]
    # 确保最后一帧一定被包含
    indices[-1] = total - 1
    # 去重并排序
    indices = sorted(set(indices))
    return [images[i] for i in indices]


def sample_frame_indices_for_llm(total_frames: int, max_images: int) -> List[int]:
    """
    返回被选中的帧索引列表，用于同时子采样帧和时间戳（主要用于 temporal agent）。
    
    Args:
        total_frames: 总帧数
        max_images: 最大允许的图片数量
        
    Returns:
        被选中的帧索引列表
    """
    if total_frames <= 0:
        return []
    if max_images <= 0 or total_frames <= max_images:
        return list(range(total_frames))

    step = total_frames / float(max_images)
    indices = [int(i * step) for i in range(max_images)]
    # 确保最后一帧一定被包含
    indices[-1] = total_frames - 1
    # 去重并排序
    indices = sorted(set(indices))
    return indices
