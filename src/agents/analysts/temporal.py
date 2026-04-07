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
from agents.routing.router_llm import pick_router_preview_frames, ROUTER_PREVIEW_MAX_FRAMES
from agents.routing.temporal_skill_router import select_temporal_skill_ids


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

    # Filter out None frames, 再根据上限选代表性帧
    all_valid_frames = [f for f in frame_inputs if f is not None]
    llm_conf = config.get("llm", {})
    max_images = int(llm_conf.get("max_images") or 50)
    selected_indices = sample_frame_indices_for_llm(len(all_valid_frames), max_images)
    sampled_frames = [all_valid_frames[i] for i in selected_indices]

    # Prepare frame information
    frame_count = len(sampled_frames)

    frame_timestamps = meta.get("frame_timestamps_sec") or []
    sampled_timestamps = [
        frame_timestamps[i] if i < len(frame_timestamps) else None
        for i in selected_indices
    ]

    temporal_algo_evidence_parts = []
    frames_meta = artifacts.get("frames") or []
    if frames_meta:
        max_router_preview = int(
            llm_conf.get("skill_router_max_preview_frames", ROUTER_PREVIEW_MAX_FRAMES)
        )
        router_preview = pick_router_preview_frames(sampled_frames, max_router_preview)
        selected_skill_ids = select_temporal_skill_ids(
            case, artifacts, config, router_preview
        )
        active = set(selected_skill_ids)
        logger.info(f"{case.case_id} temporal sub-skills active: {selected_skill_ids}")
        paths = get_evidence_paths(case.case_id)
        frames_dir = paths["frames"]

        if "lpc" in active:
            try:
                lpc_analysis = analyze_lpc_sequence(
                    frames_dir=frames_dir,
                    frames_meta=frames_meta,
                )
                if lpc_analysis is not None:
                    if lpc_analysis.has_face:
                        logger.debug(
                            f"Temporal LPC: detected face, adding evidence for {case.case_id}"
                        )
                        temporal_algo_evidence_parts.append(
                            "Temporal CV Evidence (local phase coherence around faces):\n"
                            f"{lpc_analysis.summary}"
                        )
                    else:
                        logger.debug(f"Temporal LPC: no face detected for {case.case_id}")
                else:
                    logger.debug(f"Temporal LPC: analysis returned None for {case.case_id}")
            except Exception as error:  # noqa: BLE001
                logger.error(f"Temporal LPC skill failed for {case.case_id}: {error}")

        if "feature_stability" in active:
            try:
                feature_stability_analysis = analyze_feature_stability(
                    frames_dir=frames_dir,
                    frames_meta=frames_meta,
                )
                if feature_stability_analysis is not None:
                    if feature_stability_analysis.has_valid_tracking:
                        logger.debug(
                            f"Temporal feature stability: valid tracking "
                            f"(survival={feature_stability_analysis.survival_rate_10_frames:.1%}, "
                            f"smoothness={feature_stability_analysis.avg_trajectory_smoothness:.2f}, "
                            f"jerk={feature_stability_analysis.high_jerk_ratio:.1%}), "
                            f"adding evidence for {case.case_id}"
                        )
                        temporal_algo_evidence_parts.append(
                            "Temporal CV Evidence (feature point tracking stability):\n"
                            f"{feature_stability_analysis.summary}"
                        )
                    else:
                        logger.debug(
                            f"Temporal feature stability: no valid tracking for {case.case_id}"
                        )
                else:
                    logger.debug(
                        f"Temporal feature stability: analysis returned None for {case.case_id}"
                    )
            except Exception as error:  # noqa: BLE001
                logger.error(
                    f"Temporal feature_stability skill failed for {case.case_id}: {error}"
                )

    temporal_algo_evidence = ""
    if temporal_algo_evidence_parts:
        temporal_algo_evidence = "\n\n" + "\n\n".join(temporal_algo_evidence_parts) + "\n"

    frame_urls_str = ", ".join([f"Frame {i+1}" for i in range(frame_count)])
    timestamps_str = ", ".join(
        [f"{ts:.3f}" if ts is not None else "N/A" for ts in sampled_timestamps]
    )

    prompt = TEMPORAL_PROMPT.format(
        frame_count=frame_count,
        frame_urls=frame_urls_str,
        frame_timestamps=timestamps_str,
        temporal_algo_evidence=temporal_algo_evidence,
    )

    llm_response = call_llm(
        prompt=prompt,
        config=config["llm"],
        pydantic_model=AnalystLLMOutput,
        images=sampled_frames
    )

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

    logger.info(
        f"Temporal analysis for {case.case_id}: score_fake={result.score_fake:.3f}, "
        f"confidence={result.confidence:.3f}, reasoning: {llm_response.reasoning}"
    )
    logger.log_agent_status(agent_name, case.case_id, "completed")

    return {"results": {agent_name: result}}
