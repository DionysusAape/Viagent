"""Spatial analysis agent for detecting spatial artifacts"""
from graph.constants import AgentKey
from graph.schema import AgentResult, GraphState, EvidenceItem, AnalystLLMOutput
from llm.inference import call_llm
from llm.prompt import SPATIAL_PROMPT
from util.logger import logger
from util.frame_sampling import sample_frame_indices_for_llm
from pipeline.evidence import get_evidence_paths
from agents.routing.router_llm import pick_router_preview_frames, ROUTER_PREVIEW_MAX_FRAMES
from agents.routing.spatial_skill_router import select_spatial_skill_ids


def spatial_agent(state: GraphState) -> GraphState:
    """Spatial analysis agent with frame selector and CV skills"""
    agent_name = AgentKey.SPATIAL
    case = state["case"]
    artifacts = state["artifacts"]
    config = state["config"]

    logger.log_agent_status(agent_name, case.case_id, "Analyzing spatial artifacts")

    # 1. Get data from artifacts
    frames = artifacts.get("frames", [])  # List of {"index": 1, "file": "frame_001.jpg", "timestamp_sec": 0.5}
    frame_inputs = artifacts.get("frame_inputs", [])  # List of base64 data URLs
    
    # Get frames directory path
    paths = get_evidence_paths(case.case_id)
    frames_dir = paths["frames"]

    # 2. First, do LLM sampling (consistent with existing logic)
    all_valid_frames = [f for f in frame_inputs if f is not None]
    llm_conf = config.get("llm", {})
    max_images = int(llm_conf.get("max_images") or 50)
    sampled_indices = sample_frame_indices_for_llm(len(all_valid_frames), max_images)
    
    # Get sampled frames data and inputs
    sampled_frames_data = [frames[i] for i in sampled_indices if i < len(frames)]
    sampled_frame_inputs = [all_valid_frames[i] for i in sampled_indices]

    # 3. Use all sampled frames for analysis (no frame selector)
    # This ensures we don't miss frames with useful information
    selected_frames = sampled_frames_data
    spatial_config = config.get("spatial", {})

    max_router_preview = int(
        llm_conf.get("skill_router_max_preview_frames", ROUTER_PREVIEW_MAX_FRAMES)
    )
    router_preview = pick_router_preview_frames(sampled_frame_inputs, max_router_preview)

    # 4. Spatial Skills (vision LLM picks subset using preview frames + skills/spatial/SKILL.md)
    from skill.spatial.ela import analyze_ela_boundary
    from skill.spatial.patch_anomaly import analyze_patch_inconsistency
    from skill.spatial.boundary_anomaly import analyze_boundary_anomaly
    from skill.spatial.edge_coherence import analyze_edge_coherence
    from skill.spatial.multi_face_collapse_detection import analyze_multi_face_collapse
    from skill.spatial.blur_uniformity import analyze_blur_uniformity

    selected_skill_ids = select_spatial_skill_ids(case, artifacts, config, router_preview)
    active = set(selected_skill_ids)
    logger.info(f"{case.case_id} spatial sub-skills active: {selected_skill_ids}")

    ela_results: dict = {}
    patch_results: dict = {}
    boundary_results: dict = {}
    edge_results: dict = {}
    multi_face_results: dict = {}
    blur_results: dict = {}

    try:
        if "ela" in active:
            ela_results = analyze_ela_boundary(
                selected_frames,
                frames_dir,
                frame_inputs=sampled_frame_inputs,
                config=spatial_config,
            )
        if "patch" in active:
            patch_results = analyze_patch_inconsistency(
                selected_frames,
                frames_dir,
                frame_inputs=sampled_frame_inputs,
                config=spatial_config,
            )
        if "boundary" in active:
            boundary_results = analyze_boundary_anomaly(
                selected_frames,
                frames_dir,
                frame_inputs=sampled_frame_inputs,
                config=spatial_config,
            )
        if "edge" in active:
            edge_results = analyze_edge_coherence(
                selected_frames,
                frames_dir,
                frame_inputs=sampled_frame_inputs,
                config=spatial_config,
            )
        if "multi_face" in active:
            multi_face_results = analyze_multi_face_collapse(
                selected_frames,
                frames_dir,
                frame_inputs=sampled_frame_inputs,
                config=spatial_config,
            )
        if "blur" in active:
            blur_results = analyze_blur_uniformity(
                selected_frames,
                frames_dir,
                frame_inputs=sampled_frame_inputs,
                multi_face_results=multi_face_results or None,
                config=spatial_config,
            )
    except (ValueError, IOError) as error:
        logger.error(f"Spatial skills analysis failed for {case.case_id}: {error}")
        raise RuntimeError(f"Spatial skills analysis failed: {error}") from error

    # 5. Format results
    frame_labels = [f"Frame {f['index']}" for f in selected_frames]
    from skill.spatial.formatter import format_spatial_skills_for_prompt
    spatial_skills_description = format_spatial_skills_for_prompt(
        ela_results,
        patch_results,
        boundary_results,
        edge_results,
        multi_face_results,
        blur_results,
        selected_frames,
        frame_labels,
    )

    # 6. Prepare images for LLM (all sampled frames)
    # Since selected_frames = sampled_frames_data, we can use sampled_frame_inputs directly
    selected_frame_inputs = sampled_frame_inputs

    # 7. Format prompt
    prompt = SPATIAL_PROMPT.format(
        frame_count=len(selected_frames),
        frame_urls=", ".join(frame_labels),
        spatial_skills_analysis=spatial_skills_description
    )

    # 8. Call LLM (only send selected frames)
    llm_response = call_llm(
        prompt=prompt,
        config=config["llm"],
        pydantic_model=AnalystLLMOutput,
        images=selected_frame_inputs
    )

    # 9. Build result (completely compatible with existing format)
    result = AgentResult(
        agent=agent_name,
        status="ok",
        score_fake=llm_response.score_fake,
        confidence=llm_response.confidence,
        evidence=[
            EvidenceItem(
                agent=agent_name,
                type=item.get("type", "unknown"),
                detail=item.get("detail", ""),  # Already contains algorithm metrics
                score=item.get("score", 0.0)
            )
            for item in llm_response.evidence
        ],
        error=None
    )

    logger.info(
        f"Spatial analysis for {case.case_id}: "
        f"score_fake={result.score_fake:.3f}, confidence={result.confidence:.3f}, "
        f"reasoning: {llm_response.reasoning}"
    )
    logger.log_agent_status(agent_name, case.case_id, "completed")

    return {"results": {agent_name: result}}

