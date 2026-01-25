"""Convert video evidence to LLM ready format"""

from typing import Any, Dict, Iterable, Optional
from ..apis.client import VideoEvidenceClient


def iter_video_items(
    client: VideoEvidenceClient,
    label: str = "all",
    generator: Optional[str] = None,
    limit_per_group: int = 500,
) -> Iterable[Dict[str, Any]]:
    """
    Your /videos has no pagination. We iterate:
      - real videos
      - fake videos by each generator from /ping (if generator=None)
    Each group is limited to <=500 by API.
    """
    info = client.ping()
    fake_generators = info.get("fake_generators_hint", [])

    if label in ("all", "real"):
        data = client.list_videos(label="real", limit=limit_per_group)
        for it in data.get("items", []):
            yield it

    if label in ("all", "fake"):
        if generator:
            data = client.list_videos(label="fake", generator=generator, limit=limit_per_group)
            for it in data.get("items", []):
                yield it
        else:
            for g in fake_generators:
                data = client.list_videos(label="fake", generator=g, limit=limit_per_group)
                for it in data.get("items", []):
                    yield it


def get_llm_ready_evidence(
    client: VideoEvidenceClient,
    video_id: str,
    use_data_urls: bool = False,
) -> Dict[str, Any]:
    """
    Returns:
      {
        "meta": {...},
        "frame_inputs": [url_or_data_url, ...],
        "frames": [{index,file,timestamp_sec,url,(data_url)}...]
      }
    """
    ev = client.get_evidence(video_id=video_id, include_data_urls=use_data_urls)
    meta = ev.get("meta", {})
    frames = ev.get("frames", [])

    if use_data_urls:
        frame_inputs = [f.get("data_url") for f in frames if f.get("data_url")]
    else:
        frame_inputs = [f.get("url") for f in frames if f.get("url")]

    return {"meta": meta, "frames": frames, "frame_inputs": frame_inputs}
