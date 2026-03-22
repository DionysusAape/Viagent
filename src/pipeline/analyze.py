"""Unified entry point for video analysis"""
from pathlib import Path
from typing import Dict, Any, Optional

from graph.workflow import AgentWorkflow
from graph.schema import VideoCase
from util.config import load_config
from util.paths import DATA_ROOT, get_data_root
from .evidence import make_video_id, load_evidence


def analyze_video(video_path: str, label: Optional[str] = None, config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Unified entry point for video analysis.
    
    This function:
    1. Loads configuration from YAML/env/defaults
    2. Prepares VideoCase and artifacts
    3. Runs the complete multi-agent workflow
    4. Returns structured analysis results
    
    Args:
        video_path: Path to video file (relative to data root or absolute)
        label: Optional ground truth label ("real" or "fake")
        config_path: Optional path to config file (default: src/config/config.yaml)
        
    Returns:
        Dict with run_id, case_id, results, verdict
    """
    # Load configuration
    config = load_config(config_path)
    
    # Resolve video path
    video_path_obj = Path(video_path)
    if not video_path_obj.is_absolute():
        video_path_obj = get_data_root() / video_path_obj
    
    if not video_path_obj.exists():
        raise FileNotFoundError(f"Video file not found: {video_path_obj}")
    
    # Get relative path from data root (raises ValueError if not within data root)
    rel_path = video_path_obj.relative_to(get_data_root())
    
    # Create video_id and VideoCase
    video_id = make_video_id(rel_path)
    case = VideoCase(
        case_id=video_id,
        video_path=str(video_path_obj),
        label=label,
        source=None,
    )
    
    # Load evidence (raises FileNotFoundError if cache is missing)
    # Always include data URLs since all agents (spatial, temporal, watermark) need images
    artifacts = load_evidence(video_id, include_data_urls=True)
    
    # Create workflow and run decision
    workflow = AgentWorkflow(config, config_path)
    result = workflow.run_decision(case, artifacts, config)
    
    # Add case info to result for backward compatibility
    result["case"] = {
        "case_id": case.case_id,
        "video_path": case.video_path,
        "label": case.label,
    }
    
    return result


# Keep backward compatibility functions
def list_videos(
    label: str = "all",
    generator: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
    dataset_structure: Optional[Dict[str, str]] = None
) -> Dict[str, Any]:
    """
    List videos from data directory.
    
    Args:
        label: Category name (directory name under data/), or 'all' to list all videos.
        generator: Generator name (only for nested category structures)
        limit: Maximum number of videos to return
        offset: Number of videos to skip
        dataset_structure: Optional dict with category names as keys.
                          If provided, uses these category names instead of scanning data/.
                          Example: {"real_dir": "Real", "fake_dir": "Fake"}
        
    Returns:
        Dict with 'count' and 'items' keys
    """
    items = []
    
    # If label="all", list all videos recursively
    if label == "all":
        return _list_all_videos_generic(limit, offset)
    
    # For specific category, determine directory path
    # If dataset_structure is provided, try to map label to directory name
    # Otherwise, use label directly as directory name
    if dataset_structure:
        category_dir_name = dataset_structure.get(label)
        category_dir = DATA_ROOT / (category_dir_name or label)
    else:
        category_dir = DATA_ROOT / label
    
    # List videos in the category directory
    if not category_dir.exists():
        return {"count": 0, "items": []}
    
    count = 0
    for video_path in sorted(category_dir.rglob("*.mp4")):
        if count < offset:
            count += 1
            continue
        
        try:
            rel_path = video_path.relative_to(DATA_ROOT)
            parts = rel_path.parts
            
            # Generator is the second part if category has subdirectories
            gen = parts[1] if len(parts) >= 3 and parts[0].lower() == label.lower() else None
            if generator and gen != generator:
                continue
            
            items.append({
                "video_id": make_video_id(rel_path),
                "label": label,
                "generator": gen,
                "file": video_path.name,
                "rel_path": rel_path.as_posix(),
            })
            
            if len(items) >= limit:
                break
        except ValueError:
            continue
    
    return {"count": len(items), "items": items}


def _list_all_videos_generic(limit: int, offset: int) -> Dict[str, Any]:
    """
    Recursively list all videos in data root (for arbitrary dataset structures).
    
    This function scans the entire data root directory recursively to find all .mp4 files,
    regardless of directory structure.
    """
    items = []
    count = 0
    
    # Recursively find all .mp4 files
    for video_path in sorted(DATA_ROOT.rglob("*.mp4")):
        if count < offset:
            count += 1
            continue
        
        try:
            rel_path = video_path.relative_to(DATA_ROOT)
            parts = rel_path.parts
            label = parts[0].lower() if parts else None
            generator = parts[1] if len(parts) >= 3 else None
            items.append({
                "video_id": make_video_id(rel_path),
                "label": label,
                "generator": generator,
                "file": video_path.name,
                "rel_path": rel_path.as_posix(),
            })
            
            if len(items) >= limit:
                break
        except ValueError:
            # Skip if path is not relative to DATA_ROOT
            continue
    
    return {"count": len(items), "items": items}
