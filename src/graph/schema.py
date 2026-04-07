"""Pydantic schemas for video analysis workflow"""
from typing import Optional, List, Dict, Any, TypedDict, NotRequired, Annotated
from dataclasses import dataclass, field
from pydantic import BaseModel, Field


def merge_dicts(left: Dict[str, Any], right: Dict[str, Any]) -> Dict[str, Any]:
    """Merge two dictionaries, with right taking precedence"""
    return {**left, **right}


@dataclass
class VideoCase:
    """Video case information"""
    case_id: str
    video_path: str
    label: Optional[str] = None  # "real" or "fake" if known
    source: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvidenceItem:
    """Individual piece of evidence"""
    agent: str
    type: str  # e.g., "metadata", "frame_anomaly", "audio_artifact"
    detail: str
    score: Optional[float] = None  # 0-1, higher = more fake


@dataclass
class AgentResult:
    """Result from a single agent"""
    agent: str
    status: str  # "ok", "skipped", "error"
    score_fake: Optional[float] = None  # 0-1, higher = more fake
    confidence: Optional[float] = None  # 0-1
    evidence: List[EvidenceItem] = field(default_factory=list)
    error: Optional[str] = None


@dataclass
class Verdict:
    """Final verdict from judge"""
    label: str  # "real", "fake", "uncertain"
    score_fake: Optional[float] = None  # 0-1
    confidence: Optional[float] = None  # 0-1
    rationale: str = ""
    evidence: List[EvidenceItem] = field(default_factory=list)


class AnalystLLMOutput(BaseModel):
    """Common Pydantic model for analyst agent LLM outputs"""
    score_fake: float = Field(
        description="Fake score (0.0-1.0), higher = more fake",
        ge=0.0,
        le=1.0
    )
    confidence: float = Field(
        description="Confidence score (0.0-1.0)",
        ge=0.0,
        le=1.0
    )
    reasoning: str = Field(
        description="Brief explanation of the analysis decision and key findings"
    )
    evidence: list = Field(
        description="List of evidence items with type, detail, and score",
        default_factory=list
    )


class SubskillRouterOutput(BaseModel):
    """LLM output for selecting which CV sub-skills to run for an analyst."""

    selected_skill_ids: List[str] = Field(
        description="Subset of analyst sub-skill ids to enable for this video (exact id strings)",
    )
    rationale: str = Field(
        default="",
        description="Short justification for which skills were included or omitted",
    )


class JudgeLLMOutput(BaseModel):
    """Pydantic model for judge agent LLM output"""
    label: str = Field(
        description="Verdict label: 'real', 'fake', or 'uncertain'"
    )
    score_fake: float = Field(
        description="Fake score (0.0-1.0), higher = more fake",
        ge=0.0,
        le=1.0
    )
    confidence: float = Field(
        description="Confidence score (0.0-1.0)",
        ge=0.0,
        le=1.0
    )
    rationale: str = Field(
        description="Explanation of the decision"
    )


class GraphState(TypedDict):
    """State passed through the LangGraph workflow"""
    case: VideoCase
    artifacts: Dict[str, Any]
    # Use merge_dicts reducer to allow multiple nodes to update results concurrently
    results: Annotated[Dict[str, AgentResult], merge_dicts]
    verdict: NotRequired[Optional[Verdict]]  # Optional, set by judge node
    run_id: str
    config: Dict[str, Any]
    analysts: List[str]  # List of agent keys to run
    current_agent_index: int  # Current agent being processed
    human_eyes_result: Optional[Any]  # HumanEyesOutput result