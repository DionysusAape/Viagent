"""Config helpers for CV sub-skills (per-analyst routing + OpenCV pipelines)."""
from __future__ import annotations

from typing import Any, Dict


def cv_skills_enabled(config: Dict[str, Any]) -> bool:
    """
    When False, style/physics/spatial/temporal skip skill routing and CV,
    and only run their vision LLM pass (prompts get empty / disabled evidence blocks).

    Default True if omitted (backward compatible).
    """
    if "enable_skills" not in config:
        return True
    return bool(config["enable_skills"])
