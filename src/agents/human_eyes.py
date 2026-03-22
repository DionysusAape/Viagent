"""Human eyes agent for detecting obvious violations of objective laws"""
from typing import Dict, Any, List
from pydantic import BaseModel, Field
from graph.constants import AgentKey
from graph.schema import VideoCase, Verdict, EvidenceItem
from llm.inference import call_llm
from llm.prompt import HUMAN_EYES_PROMPT
from util.logger import logger


class HumanEyesOutput(BaseModel):
    """Pydantic model for human eyes agent output."""
    is_obviously_fake: bool = Field(
        description="Whether the video is obviously fake based on objective law violations"
    )
    confidence: float = Field(
        description="Confidence score (0.0-1.0) for the detection",
        ge=0.0,
        le=1.0
    )
    reasoning: str = Field(
        description="Explanation of what was observed and why it indicates fake or real"
    )
    violations: List[str] = Field(
        description="List of detected violations (empty if is_obviously_fake=false)",
        default_factory=list
    )


def human_eyes_agent(case: VideoCase, artifacts: Dict[str, Any], config: Dict[str, Any]) -> Verdict:
    """Analyze key frames (first, middle, last) of a video to detect obvious violations of objective laws. (Physical/Biological/Common sense/Watermarks)"""
    logger.log_agent_status(AgentKey.HUMAN_EYES, case.case_id, "Human eyes check started")
    
    # Extract key frames (first, middle, last) - same strategy as planner
    frame_inputs = artifacts.get("frame_inputs", [])
    if not frame_inputs:
        logger.warning(f"No frames available for {case.case_id}, skipping human eyes check")
        return Verdict(label="uncertain", score_fake=None, confidence=0.0, rationale="No frames available for analysis", evidence=[])
    
    # Extract three key frames: first, middle, last
    key_frames = []
    key_frames.append(frame_inputs[0])  # First frame
    if len(frame_inputs) > 1:
        key_frames.append(frame_inputs[len(frame_inputs) // 2])  # Middle frame
    if len(frame_inputs) > 2:
        key_frames.append(frame_inputs[-1])  # Last frame
    
    # Remove None frames
    key_frames = [f for f in key_frames if f is not None]
    
    if not key_frames:
        logger.warning(f"No valid frames available for {case.case_id}, skipping human eyes check")
        return Verdict(label="uncertain", score_fake=None, confidence=0.0, rationale="No valid frames available for analysis", evidence=[])
    
    prompt = HUMAN_EYES_PROMPT.format(case_id=case.case_id)
    llm_response = call_llm(prompt, config["llm"], HumanEyesOutput, images=key_frames)
    
    evidence = []
    if llm_response.is_obviously_fake and llm_response.confidence > 0.8:
        violations_str = ", ".join(llm_response.violations) if llm_response.violations else "None"
        rationale = f"Human eyes detected obvious violation: {llm_response.reasoning}. Violations: {violations_str}."
        evidence.append(EvidenceItem(agent=AgentKey.HUMAN_EYES, type="obvious_violation", detail=llm_response.reasoning, score=1.0))
        for violation in llm_response.violations:
            evidence.append(EvidenceItem(agent=AgentKey.HUMAN_EYES, type="violation_detail", detail=violation, score=1.0))
        logger.info(f"Human eyes detected fake for {case.case_id} (confidence: {llm_response.confidence:.2f}): {llm_response.reasoning}")
        verdict = Verdict(
            label="fake",
            score_fake=1.0,
            confidence=llm_response.confidence,
            rationale=rationale,
            evidence=evidence
        )
    else:
        rationale = f"Human eyes check passed: {llm_response.reasoning}"
        evidence.append(EvidenceItem(agent=AgentKey.HUMAN_EYES, type="no_obvious_violation", detail=llm_response.reasoning, score=0.0))
        logger.info(f"Human eyes check passed for {case.case_id} (confidence: {llm_response.confidence:.2f}): {llm_response.reasoning}")
        verdict = Verdict(
            label="real",
            score_fake=0.0,
            confidence=llm_response.confidence,
            rationale=rationale,
            evidence=evidence
        )
    
    logger.log_agent_status(AgentKey.HUMAN_EYES, case.case_id, "completed")
    return verdict
