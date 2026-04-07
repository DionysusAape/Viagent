"""Load Agent-Skills-style SKILL.md packs from repo src/skill/<agent>/SKILL.md."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml

from util.config import get_repo_root


def split_skill_frontmatter(text: str) -> Tuple[Dict[str, Any], str]:
    """Split YAML frontmatter and Markdown body (Agent Skills style)."""
    if not text.strip().startswith("---"):
        return {}, text
    parts = text.split("---", 2)
    if len(parts) < 3:
        return {}, text
    front = yaml.safe_load(parts[1]) or {}
    body = parts[2].lstrip("\n")
    return front, body


def load_agent_skill_pack(agent_name: str) -> Tuple[Dict[str, Any], str]:
    """
    Load src/skill/<agent_name>/SKILL.md. Returns (frontmatter_dict, body_markdown).
    Missing file -> ({}, "").
    """
    path = get_repo_root() / "src" / "skill" / agent_name / "SKILL.md"
    if not path.is_file():
        return {}, ""
    raw = path.read_text(encoding="utf-8")
    return split_skill_frontmatter(raw)


def extract_viagent_routing(body: str) -> Dict[str, Any]:
    """Parse first ```yaml``` block in body that contains top-level key viagent_routing."""
    for match in re.finditer(r"```yaml\n(.*?)```", body, re.DOTALL):
        chunk = match.group(1).strip()
        try:
            data = yaml.safe_load(chunk)
        except yaml.YAMLError:
            continue
        if isinstance(data, dict) and "viagent_routing" in data:
            return data["viagent_routing"] if isinstance(data["viagent_routing"], dict) else {}
    return {}


def routing_subskills(routing: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return list of {id, when} from viagent_routing dict."""
    raw = routing.get("subskills")
    if not isinstance(raw, list):
        return []
    out: List[Dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        sid = item.get("id")
        if isinstance(sid, str) and sid.strip():
            out.append({"id": sid.strip(), "when": str(item.get("when", "")).strip()})
    return out
