"""Watermark analysis agent for detecting watermarks and logos"""
from graph.constants import AgentKey
from graph.schema import AgentResult, GraphState, EvidenceItem, AnalystLLMOutput
from llm.inference import call_llm
from llm.prompt import WATERMARK_PROMPT
from util.logger import logger
from util.frame_sampling import sample_frames_for_llm


def watermark_agent(state: GraphState) -> GraphState:
    """Watermark analysis agent analyzing video frames for watermarks and logos"""
    agent_name = AgentKey.WATERMARK
    case = state["case"]
    artifacts = state["artifacts"]
    config = state["config"]

    logger.log_agent_status(agent_name, case.case_id, "Analyzing watermarks and logos")

    # Extract frame information from artifacts
    # Note: Frame count check is done during conversion phase
    # If insufficient frames, cache won't be created and workflow won't run
    frame_inputs = artifacts.get("frame_inputs")

    # Filter out None frames, 再根据上限采样
    all_valid_frames = [f for f in frame_inputs if f is not None]
    llm_conf = config.get("llm", {})
    max_images = int(llm_conf.get("max_images") or 50)
    sampled_frames = sample_frames_for_llm(all_valid_frames, max_images)

    # Prepare frame information
    frame_count = len(sampled_frames)

    # Format frame URLs for prompt
    frame_urls_str = ", ".join([f"Frame {i+1}" for i in range(frame_count)])

    # Format prompt with frame information
    prompt = WATERMARK_PROMPT.format(
        frame_count=frame_count,
        frame_urls=frame_urls_str
    )

    # Call LLM with structured output (with frame images)
    llm_response = call_llm(
        prompt=prompt,
        config=config["llm"],
        pydantic_model=AnalystLLMOutput,
        images=sampled_frames
    )

    # Convert LLM output directly to AgentResult
    evidence_items = []
    for item in llm_response.evidence or []:
        if not isinstance(item, dict):
            continue
        evidence_items.append(
            EvidenceItem(
                agent=agent_name,
                type=item.get("type", "unknown"),
                detail=item.get("detail", ""),
                score=item.get("score", 0.0)
            )
        )

    result = AgentResult(
        agent=agent_name,
        status="ok",
        score_fake=llm_response.score_fake,
        confidence=llm_response.confidence,
        evidence=evidence_items,
        error=None
    )

    logger.info(
        f"Watermark analysis for {case.case_id}: "
        f"score_fake={result.score_fake:.3f}, confidence={result.confidence:.3f}, "
        f"reasoning: {llm_response.reasoning}"
    )
    logger.log_agent_status(agent_name, case.case_id, "completed")

    # Return only the new result entry (LangGraph will merge it with existing results)
    return {"results": {agent_name: result}}
