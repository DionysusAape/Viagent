"""Agent registry for dynamic agent management"""
from typing import Dict, Callable, List
from graph.constants import AgentKey
from agents.analysts import *


class AgentRegistry:
    """Registry for all agents"""

    agent_func_mapping: Dict[str, Callable] = {}
    agent_description_mapping: Dict[str, str] = {}

    ANALYSIS_KEYS: List[AgentKey] = [
        AgentKey.STYLE,
        AgentKey.PHYSICS,
        AgentKey.SPATIAL,
        AgentKey.TEMPORAL,
        AgentKey.WATERMARK,
    ]

    @classmethod
    def get_agent_func_by_key(cls, key: str) -> Callable:
        """Get the agent function by key (string or AgentKey)"""
        return cls.agent_func_mapping.get(key)

    @classmethod
    def get_agent_description_by_key(cls, key: str) -> str:
        """Get the agent description (string or AgentKey)"""
        return cls.agent_description_mapping.get(key)

    @classmethod
    def get_analysis_agents_keys(cls) -> list:
        """Get all analysis agents keys"""
        return cls.ANALYSIS_KEYS

    @classmethod
    def check_agent_key(cls, key: str) -> bool:
        """Check if the agent key is valid"""
        return key in cls.ANALYSIS_KEYS

    @classmethod
    def register_agent(cls, key: AgentKey, func: Callable, description: str) -> None:
        """Register an agent"""
        cls.agent_func_mapping[key] = func
        cls.agent_description_mapping[key] = description

    @classmethod
    def run_registry(cls):
        """Run the registry"""
        cls.register_agent(
            AgentKey.STYLE,
            style_agent,
            (
                "Analyze visual style to distinguish AI-generated synthetic content from traditional CG/animation/game/commercial content. "
                "Uses FFT (Fast Fourier Transform) analysis for frequency domain characteristics. "
                "Detects: AI generation indicators (unstable rendering quality, unnatural mixed features, ai rendering artifacts, "
                "overly smooth surface textures, weak natural microtexture, synthetic material uniformity, unidentifiable source), "
                "traditional CG/animation/game indicators (consistent game/animation/commercial ad style, professional CG pipeline). "
                "Analyzes material/texture details: surface smoothness, natural microtexture, material uniformity. "
                "Useful when: content type is ambiguous (could be AI-generated or traditional CG/animation/game), "
                "frames show stylized/synthetic appearance, source type is uncertain, or material/texture details need analysis."
            )
        )
        cls.register_agent(
            AgentKey.PHYSICS,
            physics_agent,
            (
                "Analyze physical plausibility and objective law consistency across frames. "
                "Uses optical flow analysis to detect motion coherence, background coupling, and sudden non-inertial changes. "
                "Detects: objects floating without support, impossible trajectories, object interpenetration, "
                "gravity violations, anomalous visual artifacts (rendering/compositing artifacts, NOT real visual effects). "
                "Distinguishes AI generation errors (e.g., legs twisted during normal walking, objects clipping through body) "
                "from artistic choices (animation/CG/game physics). "
                "Only flags physics violations as fake when content appears AI-generated/photorealistic AND violation is clearly an AI error. "
                "Useful when: people/objects are interacting physically, there is motion/movement "
                "(jumping, falling, throwing, collisions), unusual poses/trajectories, potential physics violations, "
                "or anomalous visual artifacts appear. Should be selected for videos with visible motion, physical interactions, or objects/subjects that move."
            )
        )
        cls.register_agent(
            AgentKey.SPATIAL,
            spatial_agent,
            (
                "Analyze spatial artifacts within individual frames. "
                "Uses multiple CV skills: ELA (Error Level Analysis) for boundary detection, "
                "patch inconsistency analysis, boundary anomaly detection (halo scores, boundary-to-background ratios), "
                "local edge coherence analysis (edge noise, soft edge ratios), "
                "and multi-face collapse detection (detects all faces, analyzes each for collapse). "
                "Detects: edge melting/halos, texture inconsistencies, anatomy errors (hands/faces), "
                "lighting inconsistencies, object boundary problems, face collapse (structural distortion, feature misalignment, texture inconsistency). "
                "Applies 'One Drop of Blood' rule: any face with AI-characteristic collapse leads to high fake score, regardless of proportion. "
                "Useful when: faces/people/animals or complex 3D objects/scenes are present, realism is uncertain, "
                "or boundary/edge anomalies are suspected."
            )
        )
        cls.register_agent(
            AgentKey.TEMPORAL,
            temporal_agent,
            (
                "Analyze temporal consistency of the same subject across consecutive frames. "
                "Uses Local Phase Coherence (LPC) analysis for facial edge stability and "
                "feature point tracking stability (survival rate, trajectory smoothness, jerk ratio) "
                "to detect pixel-level inconsistencies. "
                "Detects: identity drift, geometry popping, texture flicker, within-shot warping, "
                "unnatural motion interpolation, shot-boundary changes, photo slideshow sequences. "
                "Distinguishes photo slideshows (normal content, high jerk/low survival is expected) "
                "from video footage (high jerk within same scene indicates AI generation). "
                "Considers scene changes: reduces weight of feature stability signals when multiple shot-boundary changes detected. "
                "Useful when: animate subjects present, frame-to-frame changes detected, motion scenes, "
                "or temporal consistency needs verification."
            )
        )
        cls.register_agent(
            AgentKey.WATERMARK,
            watermark_agent,
            (
                "Analyze watermarks, logos, and branding elements. "
                "Detects: generator/editor tool marks, platform watermarks, news agency logos, trial version overlays. "
                "Useful when: visible branding, tool marks, text overlays, or watermarks are present in frames."
            )
        )
