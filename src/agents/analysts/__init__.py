"""Video analysis agents"""
from .spatial import spatial_agent
from .temporal import temporal_agent
from .watermark import watermark_agent
from .style import style_agent
from .physics import physics_agent

__all__ = [
    "spatial_agent",
    "temporal_agent",
    "watermark_agent",
    "style_agent",
    "physics_agent",
]