"""Constants for agent registry and workflow"""
from enum import Enum


class AgentKey(str, Enum):
    """Agent identifier keys"""
    HUMAN_EYES = "human_eyes"
    STYLE = "style"
    PHYSICS = "physics"
    SPATIAL = "spatial"
    TEMPORAL = "temporal"
    WATERMARK = "watermark"
    PLANNER = "planner"
    JUDGE = "judge"


class VerdictLabel(str, Enum):
    """Final verdict labels"""
    REAL = "real"
    FAKE = "fake"
    UNCERTAIN = "uncertain"
