"""Meta-planner for selecting optimal analyst combination"""
from typing import List, Dict, Any
from pydantic import BaseModel, Field

from agents.registry import AgentRegistry
from graph.constants import AgentKey
from graph.schema import VideoCase
from llm.inference import call_llm
from llm.prompt import PLANNER_PROMPT
from util.logger import logger
from util.frame_sampling import sample_frames_for_llm

class PlannerOutput(BaseModel):
    """Pydantic model for planner agent output."""
    agents: List[str] = Field(
        description="Name list of selected agents"
    )
    reasoning: str = Field(
        description="Brief explanation of why these agents were selected based on content and visible clues"
    )

def planner_agent(case: VideoCase, config: Dict[str, Any], workflow_analysts: List[str], artifacts: Dict[str, Any]) -> List[str]:
    """
    Selects which analysts to run for a video case using LLM.
    """

    logger.log_agent_status(AgentKey.PLANNER, case.case_id, "Planner started")

    # Build agent info string from registry
    agent_info = "\n".join([f"- {agent}: {AgentRegistry.get_agent_description_by_key(agent)}" for agent in workflow_analysts])

    # Extract all available frames from artifacts (already sampled by evidence pipeline)
    frame_inputs = artifacts.get("frame_inputs") or []
    # Filter out None frames
    all_valid_frames = [frame for frame in frame_inputs if frame is not None]

    # 根据当前 LLM 配置的 max_images（如果未配置，则默认为 50）
    llm_conf = config.get("llm", {})
    max_images = int(llm_conf.get("max_images") or 50)
    # 使用与 analyst agents 相同的采样策略，覆盖视频前中后
    frame_images = sample_frames_for_llm(all_valid_frames, max_images)

    # Format prompt template with case and agent information
    prompt = PLANNER_PROMPT.format(case_id=case.case_id, agent_info=agent_info)

    # Call LLM with structured output (with images if available)
    response = call_llm(prompt, config["llm"], PlannerOutput, images=frame_images)

    logger.info(f"Planner agent selected {response.agents} for case {case.case_id} with reasoning: {response.reasoning}")
    return response.agents
