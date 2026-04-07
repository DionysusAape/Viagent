"""Shared video metadata JSON for sub-skill LLM routers."""
from __future__ import annotations
from typing import Any, Dict
from graph.schema import VideoCase

def build_video_routing_context(case: VideoCase, artifacts: Dict[str, Any]) -> Dict[str, Any]:
    meta = artifacts.get("meta") or {}
    frames = artifacts.get("frames") or []
    return {
        "case_id": case.case_id,
        "label": case.label,
        "rel_path": meta.get("rel_path"),
        "file_name": meta.get("file_name"),
        "duration_sec": meta.get("duration_sec"),
        "width": meta.get("width"),
        "height": meta.get("height"),
        "codec": meta.get("codec"),
        "sampled_frame_count_for_workflow": len(frames),
    }
