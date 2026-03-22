"""Style/source analysis agent for detecting global CG/animation rendering style."""
from graph.constants import AgentKey
from graph.schema import AgentResult, GraphState, EvidenceItem, AnalystLLMOutput
from llm.inference import call_llm
from llm.prompt import STYLE_PROMPT
from util.logger import logger
from util.frame_sampling import sample_frames_for_llm

# FFT 分析功能（必需依赖：numpy 和 Pillow）
from skill.style.fft_analysis import analyze_frames_fft_batch, format_fft_features_for_prompt


def style_agent(state: GraphState) -> GraphState:
    """Style analysis agent analyzing global visual style and source type."""
    agent_name = AgentKey.STYLE
    case = state["case"]
    artifacts = state["artifacts"]
    config = state["config"]

    logger.log_agent_status(agent_name, case.case_id, "Analyzing global style and source type")

    frame_inputs = artifacts.get("frame_inputs")

    # 先过滤掉 None
    all_valid_frames = [frame for frame in frame_inputs if frame is not None]

    # 根据当前 LLM 配置的 max_images（如果未配置，则默认为 50）
    llm_conf = config.get("llm", {})
    max_images = int(llm_conf.get("max_images") or 50)
    sampled_frames = sample_frames_for_llm(all_valid_frames, max_images)

    # Prepare frame information（使用实际发送给 LLM 的帧数）
    frame_count = len(sampled_frames)

    # Format frame labels for prompt
    frame_labels = [f"Frame {index + 1}" for index in range(frame_count)]
    frame_labels_str = ", ".join(frame_labels)

    # 进行 FFT 分析（必需依赖：numpy 和 Pillow）
    try:
        fft_features_list = analyze_frames_fft_batch(sampled_frames)
        fft_description = format_fft_features_for_prompt(fft_features_list, frame_labels)
    except (ValueError, IOError) as error:
        logger.error(f"FFT analysis failed for {case.case_id}: {error}")
        raise RuntimeError(f"FFT analysis failed: {error}") from error

    # Format prompt with frame information and FFT features
    prompt = STYLE_PROMPT.format(
        frame_count=frame_count,
        frame_labels=frame_labels_str,
        fft_analysis=fft_description,
    )

    # Call LLM with structured output (with frame images)
    llm_response = call_llm(
        prompt=prompt,
        config=config["llm"],
        pydantic_model=AnalystLLMOutput,
        images=sampled_frames,
    )

    # Convert LLM output directly to AgentResult
    result = AgentResult(
        agent=agent_name,
        status="ok",
        score_fake=llm_response.score_fake,
        confidence=llm_response.confidence,
        evidence=[
            EvidenceItem(
                agent=agent_name,
                type=item.get("type", "unknown"),
                detail=item.get("detail", ""),
                score=item.get("score", 0.0),
            )
            for item in llm_response.evidence
        ],
        error=None,
    )

    logger.info(
        f"Style analysis for {case.case_id}: "
        f"score_fake={result.score_fake:.3f}, confidence={result.confidence:.3f}, "
        f"reasoning: {llm_response.reasoning}",
    )
    logger.log_agent_status(agent_name, case.case_id, "completed")

    # Return only the new result entry (LangGraph will merge it with existing results)
    return {"results": {agent_name: result}}
