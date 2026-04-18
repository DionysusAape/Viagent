"""Judge agent that fuses results from multiple analysts into a verdict"""
import math
from typing import Dict, Any, List
from graph.constants import AgentKey
from graph.schema import Verdict, GraphState, JudgeLLMOutput
from llm.inference import call_llm
from llm.prompt import JUDGE_PROMPT, JUDGE_VISUAL_PROMPT
from util.logger import logger
from util.frame_sampling import sample_frames_for_llm


def clamp(value: float, min_val: float, max_val: float) -> float:
    """Clamp value to [min_val, max_val]"""
    return max(min_val, min(max_val, value))


def std_dev(values: List[float]) -> float:
    """Calculate standard deviation of values"""
    if not values or len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((x - mean) ** 2 for x in values) / len(values)
    return math.sqrt(variance)


def decide_dynamic(
    analysis_results: Dict[str, Dict[str, Any]],
    cfg: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Three-stage decision making: Strong Override -> Majority Vote -> Aggressive Fusion.
    
    Args:
        analysis_results: Dict of agent results, e.g. {
            "spatial": {"score_fake": 0.2, "confidence": 0.7, "evidence": [...]},
            ...
        }
        cfg: Configuration dict with judge_dynamic settings
        
    Returns:
        Dict with label, score_fake, confidence, rationale
    """
    # Extract config parameters
    w_min = cfg.get("w_min", 0.05)
    p = cfg.get("p", 2.0)
    m0 = cfg.get("m0", 0.05)
    m1 = cfg.get("m1", 0.15)
    k_disagree = cfg.get("k_disagree", 0.4)
    vote_s_high = cfg.get("vote_s_high", 0.8)
    vote_s_low = cfg.get("vote_s_low", 0.25)
    vote_c = cfg.get("vote_c", 0.6)
    strong_s = cfg.get("strong_s", 0.85)
    strong_c = cfg.get("strong_c", 0.6)
    
    # Parse results with evidence information
    agent_data = []
    for agent_name, result in analysis_results.items():
        if not isinstance(result, dict):
            continue
        score = result.get("score_fake")
        conf = result.get("confidence")
        evidence = result.get("evidence", [])
        
        if score is None or conf is None:
            continue
        
        score_i = clamp(float(score), 0.0, 1.0)
        conf_i = clamp(float(conf), 0.0, 1.0)
        evidence_types = [ev.get("type", "") for ev in evidence if isinstance(ev, dict)]
        
        agent_data.append({
            "name": agent_name,
            "score": score_i,
            "conf": conf_i,
            "evidence": evidence,
            "evidence_types": evidence_types
        })
    
    # D) 防呆：如果没有有效结果，返回 uncertain
    if not agent_data:
        logger.warning("[judge] No valid analysis results, returning uncertain")
        return {
            "label": "uncertain",
            "score_fake": 0.5,
            "confidence": 0.0,
            "rationale": "No valid analysis results available"
        }
    
    # ============================================================
    # Stage 1: Strong Evidence Override (一滴血原则)
    # ============================================================
    for agent in agent_data:
        # 一滴血原则：只要 score >= 0.85 且 conf >= 0.6，直接判定为 fake
        # 不需要检查 evidence.type 或 verifiable
        if agent["score"] >= strong_s and agent["conf"] >= strong_c:
            # Override: return fake immediately
            label = "fake"
            score_fake = agent["score"]
            confidence = min(0.85, agent["conf"] + 0.1)
            
            rationale = (
                f"Strong evidence override: {agent['name']} high fake score "
                f"(score={score_fake:.3f}, conf={agent['conf']:.3f})"
            )
            
            logger.info(
                f"[judge] STRONG OVERRIDE triggered by {agent['name']}: "
                f"score={score_fake:.3f}, conf={confidence:.3f}"
            )
            
            return {
                "label": label,
                "score_fake": score_fake,
                "confidence": confidence,
                "rationale": rationale
            }
    
    # ============================================================
    # Stage 2: Majority Vote
    # ============================================================
    if len(agent_data) >= 2:
        high_fake_votes = 0
        low_real_votes = 0
        total_votes = 0
        
        for agent in agent_data:
            # Check if high fake vote (一滴血原则：不需要 verifiable)
            if agent["score"] >= vote_s_high and agent["conf"] >= vote_c:
                high_fake_votes += 1
                total_votes += 1
            # Check if low real vote
            elif agent["score"] <= vote_s_low and agent["conf"] >= vote_c:
                low_real_votes += 1
                total_votes += 1
            else:
                total_votes += 1
        
        if total_votes > 0:
            fake_ratio = high_fake_votes / total_votes
            real_ratio = low_real_votes / total_votes
            
            if fake_ratio > 0.5:
                # Majority vote: fake
                label = "fake"
                # Use weighted average of high fake votes (一滴血原则：不需要 verifiable)
                fake_scores = [
                    a["score"] for a in agent_data
                    if a["score"] >= vote_s_high and a["conf"] >= vote_c
                ]
                score_fake = sum(fake_scores) / len(fake_scores) if fake_scores else 0.8
                confidence = sum(a["conf"] for a in agent_data if a["score"] >= vote_s_high) / len([a for a in agent_data if a["score"] >= vote_s_high]) if high_fake_votes > 0 else 0.7
                confidence = min(0.85, confidence)
                rationale = f"Majority vote: {high_fake_votes}/{total_votes} agents indicate fake (score={score_fake:.3f}, conf={confidence:.3f})"
                
                logger.info(
                    f"[judge] MAJORITY VOTE: fake ({high_fake_votes}/{total_votes}), "
                    f"score={score_fake:.3f}, conf={confidence:.3f}"
                )
                
                return {
                    "label": label,
                    "score_fake": score_fake,
                    "confidence": confidence,
                    "rationale": rationale
                }
            
            if real_ratio > 0.5:
                # Majority vote: real
                label = "real"
                real_scores = [
                    a["score"] for a in agent_data
                    if a["score"] <= vote_s_low and a["conf"] >= vote_c
                ]
                score_fake = sum(real_scores) / len(real_scores) if real_scores else 0.2
                confidence = sum(a["conf"] for a in agent_data if a["score"] <= vote_s_low) / len([a for a in agent_data if a["score"] <= vote_s_low]) if low_real_votes > 0 else 0.7
                confidence = min(0.85, confidence)
                rationale = f"Majority vote: {low_real_votes}/{total_votes} agents indicate real (score={score_fake:.3f}, conf={confidence:.3f})"
                
                logger.info(
                    f"[judge] MAJORITY VOTE: real ({low_real_votes}/{total_votes}), "
                    f"score={score_fake:.3f}, conf={confidence:.3f}"
                )
                
                return {
                    "label": label,
                    "score_fake": score_fake,
                    "confidence": confidence,
                    "rationale": rationale
                }
    
    # ============================================================
    # Stage 3: Aggressive Dynamic Fusion
    # ============================================================
    # Calculate weights with neutral penalty
    valid_results = []
    per_agent_info = []
    
    for agent in agent_data:
        w_i = (max(w_min, agent["conf"]) ** p)
        
        valid_results.append((
            agent["name"],
            agent["score"],
            agent["conf"],
            w_i
        ))
        per_agent_info.append((
            agent["name"],
            agent["score"],
            agent["conf"],
            w_i,
            agent["evidence_types"]
        ))
    
    # Weighted fusion score
    weighted_sum = sum(w_i * score_i for _, score_i, _, w_i in valid_results)
    weight_sum = sum(w_i for _, _, _, w_i in valid_results)
    
    if weight_sum == 0:
        S = 0.5
    else:
        S = weighted_sum / weight_sum
    
    # Fusion confidence
    weighted_conf_sum = sum(w_i * conf_i for _, _, conf_i, w_i in valid_results)
    C_base = weighted_conf_sum / weight_sum if weight_sum > 0 else 0.0
    
    # Calculate disagree on all agents
    all_scores = [score_i for _, score_i, _, _ in valid_results]
    if len(all_scores) >= 2:
        disagree = std_dev(all_scores)
    else:
        disagree = 0.0
    
    C_adj = clamp(C_base * (1 - k_disagree * disagree), 0.0, 1.0)
    
    # Decision: simple binary threshold (no uncertain zone)
    # S > 0.5 → fake, S <= 0.5 → real
    if S > 0.5:
        label = "fake"
    else:
        label = "real"
    
    # Generate rationale
    rationale = f"S={S:.3f}, C={C_adj:.3f}"
    
    # Debug log
    per_agent_str = ", ".join([
        f"{name}(s={s:.3f},c={c:.3f},w={w:.3f},types={types})"
        for name, s, c, w, types in per_agent_info
    ])
    logger.info(
        f"[judge] FUSION: S={S:.3f}, C_adj={C_adj:.3f}, "
        f"disagree={disagree:.3f}, label={label}, per_agent=[{per_agent_str}]"
    )
    
    return {
        "label": label,
        "score_fake": S,
        "confidence": C_adj,
        "rationale": rationale
    }


def _should_trigger_visual_review(
    results: Dict[str, Any],
    visual_cfg: Dict[str, Any] = None,
    judge_cfg: Dict[str, Any] = None
) -> bool:
    """Decide whether to trigger visual review (default judge functionality).

    Trigger when:
    - At least one agent has a relatively high fake score (>= high_th but < override_threshold) with decent confidence
    - AND at least one agent has a low fake score (<= low_th) with decent confidence
    This indicates strong conflict between agents, but no strong override.
    """
    # Default parameters (visual review is always enabled)
    high_th = 0.75
    low_th = 0.25
    min_conf = 0.6
    
    # Override with config if provided
    if visual_cfg:
        high_th = float(visual_cfg.get("trigger_high_score", 0.75))
        low_th = float(visual_cfg.get("trigger_low_score", 0.25))
        min_conf = float(visual_cfg.get("min_confidence", 0.6))
    
    # Get override threshold to exclude agents that triggered override
    override_s = 0.85
    if judge_cfg:
        override_s = judge_cfg.get("strong_s", 0.85)

    high = 0
    low = 0

    for result in results.values():
        if getattr(result, "status", None) != "ok":
            continue
        score = getattr(result, "score_fake", None)
        conf = getattr(result, "confidence", None)
        if score is None or conf is None:
            continue
        if conf < min_conf:
            continue
        # High score: >= high_th but < override_threshold (exclude override cases)
        if score >= high_th and score < override_s:
            high += 1
        elif score <= low_th:
            low += 1

    return high >= 1 and low >= 1


def _run_visual_review(
    case,
    results: Dict[str, Any],
    artifacts: Dict[str, Any],
    config: Dict[str, Any],
    dynamic_decision: Dict[str, Any]
) -> Dict[str, Any] | None:
    """Second-stage visual review using a small set of frames (default judge functionality).

    This uses JUDGE_VISUAL_PROMPT and passes selected frames as images.
    If anything goes wrong, returns None and caller should fall back to dynamic_decision.
    """
    frame_inputs = None
    if artifacts is not None:
        frame_inputs = artifacts.get("frame_inputs")
    if not frame_inputs:
        # No frames available, cannot do visual review
        logger.warning(
            f"[judge] Visual review requested for {case.case_id} but no frame_inputs available"
        )
        return None

    # Filter out None frames
    all_valid_frames = [f for f in frame_inputs if f is not None]
    if not all_valid_frames:
        logger.warning(
            f"[judge] Visual review requested for {case.case_id} but no valid frames available"
        )
        return None

    # Use same sampling strategy as other agents, capped by llm.max_images
    llm_conf = config.get("llm", {})
    max_images = int(llm_conf.get("max_images") or 50)
    sampled_frames = sample_frames_for_llm(all_valid_frames, max_images)

    # Format analysis results for visual prompt
    analysis_parts: List[str] = []
    for agent_key, result in results.items():
        if getattr(result, "status", None) != "ok":
            continue
        score = getattr(result, "score_fake", None)
        conf = getattr(result, "confidence", None)
        if score is None or conf is None:
            continue
        evidence_types = [ev.type for ev in result.evidence]
        reasoning = getattr(result, "reasoning", "") or ""
        # Truncate reasoning to avoid overly long prompts
        if len(reasoning) > 400:
            reasoning = reasoning[:400] + "..."
        part = (
            f"- {agent_key}: score_fake={score:.3f}, "
            f"confidence={conf:.3f}, "
            f"evidence_types={evidence_types}, "
            f"reasoning={reasoning}"
        )
        analysis_parts.append(part)

    if not analysis_parts:
        logger.warning(
            f"[judge] Visual review requested for {case.case_id} but no valid analysis parts"
        )
        return None

    analysis_results_str = "\n".join(analysis_parts)

    # Build visual review prompt
    prompt = JUDGE_VISUAL_PROMPT.format(
        analysis_results=analysis_results_str,
    )

    try:
        llm_response = call_llm(
            prompt=prompt,
            config=config["llm"],
            pydantic_model=JudgeLLMOutput,
            images=sampled_frames,
        )
    except Exception as error:  # noqa: BLE001
        logger.error(
            f"[judge] Visual review LLM call failed for {case.case_id}: {error}"
        )
        return None

    if (
        llm_response is None
        or llm_response.score_fake is None
        or llm_response.confidence is None
        or not llm_response.label
    ):
        logger.error(
            f"[judge] Visual review LLM returned invalid response for {case.case_id}"
        )
        return None

    # Map label strictly to real/fake (prompt already restricts this)
    label = llm_response.label
    if label not in ("real", "fake"):
        # Fallback: map anything else using dynamic decision
        logger.warning(
            f"[judge] Visual review label '{label}' unexpected, falling back to dynamic decision"
        )
        return None

    logger.info(
        f"[judge] VISUAL REVIEW: label={label}, "
        f"score={llm_response.score_fake:.3f}, "
        f"conf={llm_response.confidence:.3f}"
    )

    return {
        "label": label,
        "score_fake": float(llm_response.score_fake),
        "confidence": float(llm_response.confidence),
        "rationale": llm_response.rationale,
    }


def _run_direct_visual_judge(
    case,
    artifacts: Dict[str, Any],
    config: Dict[str, Any],
) -> Dict[str, Any] | None:
    """Direct visual judge path when no analyst results are available."""
    frame_inputs = None
    if artifacts is not None:
        frame_inputs = artifacts.get("frame_inputs")
    if not frame_inputs:
        logger.warning(
            f"[judge] Direct visual judge requested for {case.case_id} but no frame_inputs available"
        )
        return None

    all_valid_frames = [f for f in frame_inputs if f is not None]
    if not all_valid_frames:
        logger.warning(
            f"[judge] Direct visual judge requested for {case.case_id} but no valid frames available"
        )
        return None

    llm_conf = config.get("llm", {})
    max_images = int(llm_conf.get("max_images") or 50)
    sampled_frames = sample_frames_for_llm(all_valid_frames, max_images)

    prompt = JUDGE_VISUAL_PROMPT.format(
        analysis_results=(
            "No analyst results are available for this case. "
            "You must judge directly from the provided video frames only."
        ),
    )

    try:
        llm_response = call_llm(
            prompt=prompt,
            config=config["llm"],
            pydantic_model=JudgeLLMOutput,
            images=sampled_frames,
        )
    except Exception as error:  # noqa: BLE001
        logger.error(
            f"[judge] Direct visual judge LLM call failed for {case.case_id}: {error}"
        )
        return None

    if (
        llm_response is None
        or llm_response.score_fake is None
        or llm_response.confidence is None
        or not llm_response.label
    ):
        logger.error(
            f"[judge] Direct visual judge returned invalid response for {case.case_id}"
        )
        return None

    label = llm_response.label
    if label not in ("real", "fake"):
        logger.warning(
            f"[judge] Direct visual judge label '{label}' unexpected for {case.case_id}"
        )
        return None

    logger.info(
        f"[judge] DIRECT VISUAL JUDGE: label={label}, "
        f"score={llm_response.score_fake:.3f}, "
        f"conf={llm_response.confidence:.3f}"
    )

    return {
        "label": label,
        "score_fake": float(llm_response.score_fake),
        "confidence": float(llm_response.confidence),
        "rationale": llm_response.rationale,
    }


def judge_agent(state: GraphState) -> GraphState:
    """Judge agent that fuses results from multiple analysts into a verdict"""
    agent_name = AgentKey.JUDGE
    case = state["case"]
    results = state["results"]
    config = state["config"]
    artifacts = state.get("artifacts")

    logger.log_agent_status(agent_name, case.case_id, "Fusing analyst results")

    # Check if dynamic decision is enabled
    judge_cfg = config.get("judge_dynamic", {})
    use_dynamic = judge_cfg.get("enabled", False)

    if not results:
        logger.info(
            f"[judge] No analyst results for {case.case_id}; using direct visual judge"
        )
        direct_visual_decision = _run_direct_visual_judge(case, artifacts, config)
        if direct_visual_decision is not None:
            verdict = Verdict(
                label=direct_visual_decision["label"],
                score_fake=direct_visual_decision["score_fake"],
                confidence=direct_visual_decision["confidence"],
                rationale=direct_visual_decision["rationale"],
                evidence=[],
            )
        else:
            verdict = Verdict(
                label="uncertain",
                score_fake=0.5,
                confidence=0.0,
                rationale="No analyst results available and direct visual judge failed",
                evidence=[],
            )
    elif use_dynamic:
        # Use dynamic decision logic
        # Convert results to dict format for decide_dynamic
        analysis_results = {}
        for agent_key, result in results.items():
            if result.status == "ok" and result.score_fake is not None:
                analysis_results[agent_key] = {
                    "score_fake": result.score_fake,
                    "confidence": result.confidence,
                    "evidence": result.evidence,
                }

        # Call dynamic decision function
        decision = decide_dynamic(analysis_results, judge_cfg)

        # Visual review (default judge functionality, always enabled)
        # Visual review is triggered only if no override was triggered and conflict exists
        visual_cfg = config.get("judge_visual_review", {})
        if _should_trigger_visual_review(results, visual_cfg, judge_cfg):
            logger.info(
                f"[judge] Visual review triggered for {case.case_id} "
                f"(dynamic S={decision['score_fake']:.3f}, "
                f"C={decision['confidence']:.3f})"
            )
            visual_decision = _run_visual_review(
                case=case,
                results=results,
                artifacts=artifacts,
                config=config,
                dynamic_decision=decision,
            )
            if visual_decision is not None:
                decision = visual_decision

        # Collect all evidence from analyst results
        all_evidence = []
        for result in results.values():
            all_evidence.extend(result.evidence)

        # Create Verdict from decision
        verdict = Verdict(
            label=decision["label"],
            score_fake=decision["score_fake"],
            confidence=decision["confidence"],
            rationale=decision["rationale"],
            evidence=all_evidence,
        )
    else:
        # Use old LLM-based decision logic
        # Format analysis results for prompt (include evidence types)
        analysis_results_parts = []
        for agent_key, result in results.items():
            if result.status == "ok" and result.score_fake is not None:
                evidence_types = [ev.type for ev in result.evidence]
                parts = (
                    f"- {agent_key}: score_fake={result.score_fake:.3f}, "
                    f"confidence={result.confidence:.3f}, "
                    f"evidence_types={evidence_types}"
                )
                analysis_results_parts.append(parts)

        analysis_results_str = (
            "\n".join(analysis_results_parts)
            if analysis_results_parts
            else "No valid analysis results"
        )

        # Get policy thresholds
        policy = config.get("decision_policy", {})
        threshold_fake = policy.get("threshold_fake", 0.7)
        threshold_real = policy.get("threshold_real", 0.3)

        # Format prompt with analysis results and policy
        prompt = JUDGE_PROMPT.format(
            analysis_results=analysis_results_str,
            threshold_fake=threshold_fake,
            threshold_real=threshold_real
        )

        # Call LLM with structured output
        llm_response = call_llm(
            prompt=prompt,
            config=config["llm"],
            pydantic_model=JudgeLLMOutput,
            images=None
        )

        # Collect all evidence from analyst results
        all_evidence = []
        for result in results.values():
            all_evidence.extend(result.evidence)

        # Convert LLM output to Verdict
        verdict = Verdict(
            label=llm_response.label,
            score_fake=llm_response.score_fake,
            confidence=llm_response.confidence,
            rationale=llm_response.rationale,
            evidence=all_evidence
        )

    logger.info(
        f"Verdict for {case.case_id}: {verdict.label} "
        f"(score_fake={verdict.score_fake:.3f}, "
        f"confidence={verdict.confidence:.3f})"
    )
    logger.log_agent_status(agent_name, case.case_id, "completed")

    # Update state with verdict (only return fields that need updating)
    return {"verdict": verdict}
