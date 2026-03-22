"""Agent-level workflow for video analysis decision making"""
import uuid
from pathlib import Path
from typing import Dict, Any, Optional, List
from langgraph.graph import StateGraph, START, END
from agents.planner import planner_agent
from agents.judge import judge_agent
from agents.human_eyes import human_eyes_agent
from agents.registry import AgentRegistry
from database.db_helper import ViagentDB
from graph.constants import AgentKey
from graph.schema import VideoCase, GraphState
from pipeline.evidence import load_evidence, make_video_id
from util.logger import logger
from util.paths import get_data_root


class AgentWorkflow:
    """
    Decision Workflow for Viagent.
    
    This workflow orchestrates planner agent, analyst agents, and judge agent
    to make decisions on individual videos. It analyzes different frames of a
    video and produces a final verdict.
    
    Workflow structure:
    1. Human eyes agent checks first frame for obvious violations (if enabled, outside graph)
    2. If obviously fake, directly return fake verdict (skip graph)
    3. Otherwise, planner agent selects which analysts to run (if planner_mode=True, outside graph)
    4. Analyst agents analyze the video frames (in graph)
    5. Judge agent makes final decision based on all analyst results (in graph)
    """

    def __init__(self, config: Dict[str, Any], config_path: Optional[str] = None):
        self.llm_config = config["llm"]
        self._config_path = config_path
        self._db = ViagentDB()

        # Register all agents
        AgentRegistry.run_registry()

        # Initialize workflow configuration
        self.planner_mode = config.get("planner_mode")
        self.workflow_analysts = config.get("workflow_analysts", [])
        self.current_analysts: Optional[List[str]] = None
        self.human_eyes_enabled = config.get("human_eyes_enabled")

        # Validate analysts and remove invalid ones
        if self.workflow_analysts:
            available_agents = [
                agent.value for agent in AgentRegistry.get_analysis_agents_keys()
            ]
            invalid_analysts = [
                a for a in self.workflow_analysts if a not in available_agents
            ]
            if invalid_analysts:
                logger.warning(f"Invalid analyst keys removed: {invalid_analysts}")
                self.workflow_analysts = [
                    a for a in self.workflow_analysts if a in available_agents
                ]

        if not self.workflow_analysts:
            logger.info("No analysts configured - using direct judge mode")

    def load_analysts(self, case: VideoCase, config: Dict[str, Any], artifacts: Dict[str, Any]):
        """
        Load the analysts for processing:
        - If planner_mode is True: use planner to select from verified workflow_analysts
        - If planner_mode is False: use all verified workflow_analysts
        - If no workflow_analysts: use direct judge mode (no analysts)
        """
        if not self.workflow_analysts:
            logger.info(f"Direct judge mode for {case.case_id} - no analysts")
            self.current_analysts = []
        elif self.planner_mode:
            logger.info("Using planner agent to select analysts from verified list")
            # Planner needs frame_inputs with data URLs (base64 images)
            # analyze.py should have loaded with include_data_urls=True if planner_mode=True
            # But reload if needed (defensive check)
            if any(f is not None for f in artifacts["frame_inputs"]):
                planner_artifacts = artifacts
            else:
                video_path_obj = Path(case.video_path)
                rel_path = video_path_obj.relative_to(get_data_root()) if video_path_obj.is_absolute() else video_path_obj
                planner_artifacts = load_evidence(make_video_id(rel_path), include_data_urls=True)

            self.current_analysts = planner_agent(case, config, self.workflow_analysts, planner_artifacts)
            if not self.current_analysts:
                raise ValueError("No analysts selected by planner")
        else:
            logger.info("Using all verified analysts")
            self.current_analysts = self.workflow_analysts.copy()

        logger.info(f"Active analysts for {case.case_id}: {self.current_analysts}")

    def check_human_eyes(self, case: VideoCase, artifacts: Dict[str, Any], config: Dict[str, Any], run_id: str) -> Optional[Dict[str, Any]]:
        """Check human_eyes agent for obvious violations"""
        if not self.human_eyes_enabled:
            return None

        human_eyes_verdict = human_eyes_agent(case, artifacts, config)

        # If human_eyes detected obvious fake, directly return verdict (skip graph)
        if human_eyes_verdict.label == "fake":
            logger.info(f"Human eyes detected obvious fake, skipping graph for {case.case_id}")
            analysis_data = {"config": config, "analysts": [], "results": {}, "verdict": human_eyes_verdict}

            if not self._db.save_complete_analysis(run_id, case, analysis_data):
                raise RuntimeError(f"Failed to save decision to database: {run_id}")
            logger.info(f"Decision results saved to database: {run_id}")

            return {
                "run_id": run_id,
                "case_id": case.case_id,
                "results": {},
                "verdict": {
                    "label": human_eyes_verdict.label,
                    "score_fake": human_eyes_verdict.score_fake,
                    "confidence": human_eyes_verdict.confidence,
                    "rationale": human_eyes_verdict.rationale,
                    "evidence_count": len(human_eyes_verdict.evidence),
                },
            }

        return None

    def build(self) -> StateGraph:
        """Build the workflow graph"""
        graph = StateGraph(GraphState)

        # Create node for judge
        graph.add_node(AgentKey.JUDGE, judge_agent)

        # Create node for each analyst
        if self.current_analysts:
            for analyst_key in self.current_analysts:
                agent_func = AgentRegistry.get_agent_func_by_key(analyst_key)
                if agent_func is None:
                    raise ValueError(f"Agent function not found for key: {analyst_key}. Make sure AgentRegistry.run_registry() was called.")
                graph.add_node(analyst_key, agent_func)
                graph.add_edge(START, analyst_key)
                graph.add_edge(analyst_key, AgentKey.JUDGE)
        else:
            # No analysts configured, go directly to judge
            graph.add_edge(START, AgentKey.JUDGE)

        # Route judge to end
        graph.add_edge(AgentKey.JUDGE, END)
        workflow = graph.compile()

        return workflow

    def run_decision(self, case: VideoCase, artifacts: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run decision workflow for a single video case.
        
        This method:
        1. Runs human_eyes check first (if enabled) - outside graph
        2. If obviously fake, directly returns fake verdict (skips graph)
        3. Otherwise, loads analysts, builds the workflow graph, then
           runs the selected analyst agents, and finally the judge agent
        """
        run_id = str(uuid.uuid4())[:8]
        logger.info(f"Running decision workflow for case: {case.case_id} (run_id: {run_id})")

        # Step 1: Run human_eyes check first (if enabled) - outside graph
        human_eyes_result = self.check_human_eyes(case, artifacts, config, run_id)
        if human_eyes_result:
            return human_eyes_result

        # Step 2: Normal flow - load analysts and run graph
        self.load_analysts(case, config, artifacts)

        workflow = self.build()
        logger.info(f"{case.case_id} workflow compiled successfully")

        initial_state: GraphState = {
            "case": case,
            "artifacts": artifacts,
            "results": {},
            "verdict": None,
            "run_id": run_id,
            "config": config,
            "analysts": self.current_analysts or [],
            "current_agent_index": 0,
            "human_eyes_result": None,  # No longer used, kept for backward compatibility
        }

        try:
            final_state = workflow.invoke(initial_state)
        except Exception as exc:
            logger.error(f"Error running workflow: {exc}")
            raise RuntimeError(f"Failed to run workflow for case {case.case_id}") from exc

        # Save to database
        analysis_data = {
            "config": config,
            "analysts": final_state["analysts"],
            "results": final_state["results"],
            "verdict": final_state["verdict"],
        }
        if not self._db.save_complete_analysis(run_id, case, analysis_data):
            raise RuntimeError(f"Failed to save decision to database: {run_id}")
        logger.info(f"Decision results saved to database: {run_id}")

        return {
            "run_id": final_state["run_id"],
            "case_id": final_state["case"].case_id,
            "results": {
                k: {
                    "agent": r.agent,
                    "status": r.status,
                    "score_fake": r.score_fake,
                    "confidence": r.confidence,
                    "evidence_count": len(r.evidence),
                    "error": r.error,
                }
                for k, r in final_state["results"].items()
            },
            "verdict": {
                "label": final_state["verdict"].label,
                "score_fake": final_state["verdict"].score_fake,
                "confidence": final_state["verdict"].confidence,
                "rationale": final_state["verdict"].rationale,
                "evidence_count": len(final_state["verdict"].evidence),
            },
        }
