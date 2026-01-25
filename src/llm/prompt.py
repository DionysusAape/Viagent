import json
from typing import Any, Dict, List


def build_video_forensics_messages(meta: Dict[str, Any], frame_inputs: List[str]) -> List[Dict[str, Any]]:
    """
    OpenAI-style multimodal messages (text + multiple images).
    frame_inputs can be http(s) urls OR data urls.
    """
    prompt = (
        "You are a forensic video analyst.\n"
        "Given multiple frames sampled from a short video, decide whether the video is REAL or AI-GENERATED (FAKE).\n"
        "Use the frames as visual evidence and metadata as context.\n\n"
        "Return STRICT JSON ONLY:\n"
        "{\n"
        '  "prediction": "real" | "fake",\n'
        '  "confidence": 0-1,\n'
        '  "signals": [string, ...],\n'
        '  "reasons": [string, ...]\n'
        "}\n\n"
        f"Metadata:\n{json.dumps(meta, ensure_ascii=False)}\n"
        f"Frame_count: {len(frame_inputs)}\n"
    )

    content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
    for u in frame_inputs:
        content.append({"type": "image_url", "image_url": {"url": u}})

    return [{"role": "user", "content": content}]


# 以后你要加新的任务 prompt，就继续在这个文件里加函数，比如：
def build_summarize_messages(text: str) -> List[Dict[str, Any]]:
    return [{"role": "user", "content": [{"type": "text", "text": f"Summarize:\n{text}"}]}]
