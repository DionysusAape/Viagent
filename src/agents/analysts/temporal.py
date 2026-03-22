"""Temporal analysis agent for detecting temporal inconsistencies"""
from graph.constants import AgentKey
from graph.schema import AgentResult, GraphState, EvidenceItem, AnalystLLMOutput
from llm.inference import call_llm
from llm.prompt import TEMPORAL_PROMPT
from pipeline.evidence import get_evidence_paths
from skill.temporal.local_phase_coherence import analyze_lpc_sequence
from skill.temporal.feature_stability_check import analyze_feature_stability
from util.logger import logger
from util.frame_sampling import sample_frame_indices_for_llm


def temporal_agent(state: GraphState) -> GraphState:
    """Temporal analysis agent analyzing video frames for temporal inconsistencies"""
    agent_name = AgentKey.TEMPORAL
    case = state["case"]
    artifacts = state["artifacts"]
    config = state["config"]

    logger.log_agent_status(agent_name, case.case_id, "Analyzing temporal consistency")

    # Extract frame information from artifacts
    frame_inputs = artifacts.get("frame_inputs")
    meta = artifacts.get("meta")

    # Note: Frame count check is done during conversion phase
    # If insufficient frames, cache won't be created and workflow won't run
    
    # Filter out None frames, 再根据上限选代表性帧
    all_valid_frames = [f for f in frame_inputs if f is not None]
    llm_conf = config.get("llm", {})
    max_images = int(llm_conf.get("max_images") or 50)
    selected_indices = sample_frame_indices_for_llm(len(all_valid_frames), max_images)
    sampled_frames = [all_valid_frames[i] for i in selected_indices]

    # Prepare frame information
    frame_count = len(sampled_frames)

    frame_timestamps = meta.get("frame_timestamps_sec") or []
    # 按相同索引子采样时间戳，长度与 sampled_frames 对齐
    sampled_timestamps = [
        frame_timestamps[i] if i < len(frame_timestamps) else None
        for i in selected_indices
    ]

    # Optional temporal CV evidence (local phase coherence on faces, feature stability)
    temporal_algo_evidence_parts = []
    try:
        frames_meta = artifacts.get("frames") or []
        if frames_meta:
            paths = get_evidence_paths(case.case_id)
            frames_dir = paths["frames"]
            
            # Local Phase Coherence analysis
            lpc_analysis = analyze_lpc_sequence(
                frames_dir=frames_dir,
                frames_meta=frames_meta,
            )
            if lpc_analysis is not None:
                if lpc_analysis.has_face:
                    logger.debug(f"Temporal LPC: detected face, adding evidence for {case.case_id}")
                    temporal_algo_evidence_parts.append(
                        "Temporal CV Evidence (local phase coherence around faces):\n"
                        f"{lpc_analysis.summary}"
                    )
                else:
                    logger.debug(f"Temporal LPC: no face detected for {case.case_id}")
            else:
                logger.debug(f"Temporal LPC: analysis returned None for {case.case_id}")
            
            # Feature Stability Check
            feature_stability_analysis = analyze_feature_stability(
                frames_dir=frames_dir,
                frames_meta=frames_meta,
            )
            if feature_stability_analysis is not None:
                if feature_stability_analysis.has_valid_tracking:
                    logger.debug(f"Temporal feature stability: valid tracking (survival={feature_stability_analysis.survival_rate_10_frames:.1%}, smoothness={feature_stability_analysis.avg_trajectory_smoothness:.2f}, jerk={feature_stability_analysis.high_jerk_ratio:.1%}), adding evidence for {case.case_id}")
                    temporal_algo_evidence_parts.append(
                        "Temporal CV Evidence (feature point tracking stability):\n"
                        f"{feature_stability_analysis.summary}"
                    )
                else:
                    logger.debug(f"Temporal feature stability: no valid tracking for {case.case_id}")
            else:
                logger.debug(f"Temporal feature stability: analysis returned None for {case.case_id}")
    except Exception as error:  # noqa: BLE001 - log and continue
        logger.error(
            f"Temporal CV skills failed for {case.case_id}: {error}"
        )
    
    temporal_algo_evidence = ""
    if temporal_algo_evidence_parts:
        temporal_algo_evidence = "\n\n" + "\n\n".join(temporal_algo_evidence_parts) + "\n"

    # Format frame URLs and timestamps for prompt
    frame_urls_str = ", ".join([f"Frame {i+1}" for i in range(frame_count)])
    timestamps_str = ", ".join(
        [f"{ts:.3f}" if ts is not None else "N/A" for ts in sampled_timestamps]
    )

    # Format prompt with frame information and optional algorithm evidence
    prompt = TEMPORAL_PROMPT.format(
        frame_count=frame_count,
        frame_urls=frame_urls_str,
        frame_timestamps=timestamps_str,
        temporal_algo_evidence=temporal_algo_evidence,
    )

    # Call LLM with structured output (with frame images)
    llm_response = call_llm(
        prompt=prompt,
        config=config["llm"],
        pydantic_model=AnalystLLMOutput,
        images=sampled_frames
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
                score=item.get("score", 0.0)
            )
            for item in llm_response.evidence
        ],
        error=None
    )

    logger.info(f"Temporal analysis for {case.case_id}: score_fake={result.score_fake:.3f}, confidence={result.confidence:.3f}, reasoning: {llm_response.reasoning}")
    logger.log_agent_status(agent_name, case.case_id, "completed")

    # Return only the new result entry (LangGraph will merge it with existing results)
    return {"results": {agent_name: result}}
