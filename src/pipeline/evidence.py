"""Core evidence generation and caching for video analysis"""
import os
import json
import base64
import hashlib
import shutil
import time
from pathlib import Path
from typing import Dict, Any, List
from util.paths import EVID_DIR, get_data_root
from util.media_extensions import is_video_suffix
from apis.video_io.probe import get_video_metadata, ensure_ffmpeg
from apis.video_io.ffmpeg import extract_frames, extract_all_frames, uniform_timestamps
from util.config import get_frame_ext, get_frame_quality, get_lock_wait_sec, get_lock_poll_sec, get_frame_mime


def sha256_hash(p: Path, chunk=1024 * 1024) -> str:
    """Calculate SHA256 hash of a file"""
    h = hashlib.sha256()
    with p.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def b64url_encode(s: str) -> str:
    """Base64 URL-safe encode"""
    return base64.urlsafe_b64encode(s.encode("utf-8")).decode("ascii").rstrip("=")


def b64url_decode(s: str) -> str:
    """Base64 URL-safe decode"""
    pad = "=" * (-len(s) % 4)
    return base64.urlsafe_b64decode((s + pad).encode("ascii")).decode("utf-8")


def make_video_id(rel_path: Path) -> str:
    """Create video_id from relative path"""
    return b64url_encode(rel_path.as_posix())


def parse_video_id(video_id: str) -> Path:
    """Parse video_id to relative path"""
    rel = Path(b64url_decode(video_id))
    if ".." in rel.parts or rel.is_absolute():
        raise ValueError("invalid video_id")
    return rel


def resolve_video_from_id(video_id: str) -> tuple[Path, Path]:
    """Resolve video path from video_id"""
    rel = parse_video_id(video_id)
    data_root = get_data_root()
    p = data_root / rel
    if not p.exists() or not is_video_suffix(p.suffix):
        raise FileNotFoundError(video_id)
    return p, rel


def get_evidence_paths(video_id: str) -> Dict[str, Path]:
    """Get paths for evidence cache"""
    root = EVID_DIR / video_id
    return {
        "root": root,
        "meta": root / "meta.json",
        "frames": root / "frames",
        "lock": root / ".lock"
    }


def list_frame_files(frames_dir: Path) -> List[str]:
    """List frame files in directory"""
    if not frames_dir.exists():
        return []
    frame_ext = get_frame_ext()
    return sorted([p.name for p in frames_dir.glob(f"*.{frame_ext}")])


def is_evidence_ready(video_id: str, min_frames: int = 1) -> bool:
    """
    Check if evidence cache is ready for a video.
    
    Args:
        video_id: Base64-encoded relative path from data root
        min_frames: Minimum number of frames required
        
    Returns:
        True if evidence cache exists and has at least min_frames frames
    """
    paths = get_evidence_paths(video_id)
    if not paths["meta"].exists():
        return False
    if not paths["frames"].exists():
        return False
    frame_count = len(list_frame_files(paths["frames"]))
    return frame_count >= min_frames


def acquire_lock(lock_path: Path) -> bool:
    """Acquire file lock atomically"""
    try:
        fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.close(fd)
        return True
    except FileExistsError:
        return False


def release_lock(lock_path: Path):
    """Release file lock"""
    try:
        lock_path.unlink(missing_ok=True)
    except Exception:
        pass


# Constants
MIN_REQUIRED_FRAMES = 2
# Sampling rates for adaptive frame selection (frames per second)
_SAMPLE_FPS_SHORT = 4.0   # For videos < 5 seconds (not used for new cache; we抽全帧)
_SAMPLE_FPS_MEDIUM = 4.0  # For videos between 5 seconds and 60 seconds
_SAMPLE_FPS_LONG = 2.0    # For videos > 60 seconds
_SHORT_VIDEO_THRESHOLD = 5.0    # Short video threshold in seconds
_LONG_VIDEO_THRESHOLD = 60.0    # Long video threshold in seconds


def _compute_target_frames(duration_sec: float) -> int:
    """
    Compute adaptive target frame count based on video duration.

    - For videos < 5 seconds: 4 frames per second（主要用于旧 cache 更新，新的短视频直接抽全帧）
    - For videos between 5 and 60 seconds (inclusive): 4 frames per second
    - For videos > 60 seconds: 2 frames per second
    - Minimum: MIN_REQUIRED_FRAMES frames
    - No maximum limit
    """
    # Fallback if duration is missing or invalid
    if duration_sec <= 0:
        return MIN_REQUIRED_FRAMES

    # Choose sampling rate based on video duration
    if duration_sec < _SHORT_VIDEO_THRESHOLD:
        sample_fps = _SAMPLE_FPS_SHORT
    elif duration_sec <= _LONG_VIDEO_THRESHOLD:
        sample_fps = _SAMPLE_FPS_MEDIUM
    else:
        sample_fps = _SAMPLE_FPS_LONG

    approx = int(duration_sec * sample_fps)
    
    # Ensure minimum frame count
    if approx < MIN_REQUIRED_FRAMES:
        return MIN_REQUIRED_FRAMES
    
    # No maximum limit - return calculated value
    return approx


def _build_frames_list(files: List[str], ts_list: List[float]) -> List[Dict[str, Any]]:
    """Build frames list from file names and timestamps"""
    return [
        {
            "index": i + 1,
            "file": fn,
            "timestamp_sec": float(ts_list[i]) if i < len(ts_list) else None,
        }
        for i, fn in enumerate(files)
    ]


def _load_meta(meta_path: Path) -> Dict[str, Any]:
    """Load metadata from JSON file"""
    return json.loads(meta_path.read_text(encoding="utf-8"))


def _save_meta(meta_path: Path, meta: Dict[str, Any]):
    """Save metadata to JSON file"""
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def _cleanup_cache(frames_dir: Path, meta_path: Path, root_dir: Path = None):
    """Clean up cache files and directories"""
    if frames_dir.exists():
        shutil.rmtree(frames_dir)
    if meta_path.exists():
        meta_path.unlink()
    if root_dir and root_dir.exists() and not meta_path.exists():
        try:
            root_dir.rmdir()
        except OSError:
            pass


def _check_frame_count(actual: int, min_required: int = MIN_REQUIRED_FRAMES, nb_frames: int = 0) -> None:
    """Check if frame count is sufficient, raise ValueError if not"""
    if actual < min_required:
        nb_info = f" (video has {nb_frames} total frames)" if nb_frames > 0 else ""
        raise ValueError(
            f"Insufficient frames for video analysis: {actual} < {min_required}. "
            f"Video is too short or frame extraction failed.{nb_info}"
        )


def wait_for_unlock_and_frames(frames_dir: Path, lock_path: Path, want: int) -> bool:
    """Wait for lock to be released and frames to be ready"""
    lock_wait_sec = get_lock_wait_sec()
    lock_poll_sec = get_lock_poll_sec()
    deadline = time.time() + lock_wait_sec
    while time.time() < deadline:
        have = len(list_frame_files(frames_dir))
        if have >= want and not lock_path.exists():
            return True
        time.sleep(lock_poll_sec)
    return False


def build_evidence(video_id: str) -> Dict[str, Any]:
    """
    Build LLM-ready evidence for a video.
    
    This is the core conversion function that:
    1. Extracts video metadata using ffprobe
    2. Extracts frames at uniform timestamps
    3. Caches results for future use
    4. Returns metadata and frame information
    
    Args:
        video_id: Base64-encoded relative path from data root
        
    Returns:
        Dict with 'meta' and 'frames' keys
    """
    ensure_ffmpeg()
    
    video_path, rel = resolve_video_from_id(video_id)
    paths = get_evidence_paths(video_id)
    paths["root"].mkdir(parents=True, exist_ok=True)
    
    frames_dir = paths["frames"]
    meta_path = paths["meta"]
    lock_path = paths["lock"]
    
    # Load basic metadata once to determine duration and adaptive frame target
    meta: Dict[str, Any] = {}
    try:
        meta = get_video_metadata(video_path)
    except Exception:
        meta = {}
    duration = float(meta.get("duration_sec") or 0.0)

    target = _compute_target_frames(duration)
    
    # Wait if another request is generating
    if lock_path.exists():
        wait_for_unlock_and_frames(frames_dir, lock_path, want=target)
    
    got_lock = acquire_lock(lock_path)
    try:
        # Fast path: ready cache
        files = list_frame_files(frames_dir)
        if meta_path.exists() and len(files) >= MIN_REQUIRED_FRAMES:
            cached_meta = _load_meta(meta_path)
            if len(files) >= MIN_REQUIRED_FRAMES:
                frames = _build_frames_list(files, cached_meta.get("frame_timestamps_sec", []))
                return {"meta": cached_meta, "frames": frames}
        
        if not got_lock:
            ready = wait_for_unlock_and_frames(frames_dir, lock_path, want=target)
            if ready and meta_path.exists():
                meta = _load_meta(meta_path)
                files = list_frame_files(frames_dir)
                if len(files) >= MIN_REQUIRED_FRAMES:
                    frames = _build_frames_list(files, meta.get("frame_timestamps_sec", []))
                    return {"meta": meta, "frames": frames}
            raise RuntimeError("Timeout waiting for evidence generation. Please retry.")
        
        # Have lock: generate meta + frames once (fixed cache)
        if not meta_path.exists():
            # New conversion: extract frames first, then check before creating meta.json
            # meta and duration were already loaded above

            # Extract frames first (adaptive target based on duration)
            # Get actual frame count from metadata if available
            nb_frames = meta.get("nb_frames", 0)

            # Special handling for 0-duration videos
            if duration <= 0:
                # duration 信息异常时，直接抽取整段视频的所有帧
                extract_all_frames(video_path, frames_dir, get_frame_ext(), get_frame_quality())
                files_after = list_frame_files(frames_dir)
                actual = len(files_after)
                ts = [float(i) * 0.033 for i in range(actual)]  # Assume ~30fps spacing
            # For short videos (< 5 seconds), extract all frames
            elif duration < _SHORT_VIDEO_THRESHOLD:
                # 对于短视频，不设置上限，直接让 ffmpeg 把整段视频的所有帧都导出
                extract_all_frames(video_path, frames_dir, get_frame_ext(), get_frame_quality())
                files_after = list_frame_files(frames_dir)
                actual = len(files_after)
                # Use uniform timestamps across the actual duration
                ts = uniform_timestamps(duration, actual) if duration > 0 else [float(i) * 0.033 for i in range(actual)]
            else:
                ts = uniform_timestamps(duration, target)
                ts = [min(t, duration - 0.01) for t in ts]  # 0.01s buffer
                extract_frames(video_path, frames_dir, ts, get_frame_ext(), get_frame_quality())
                files_after = list_frame_files(frames_dir)
                actual = len(files_after)
            
            # If we didn't get enough frames and video is very short, try extracting all available frames
            if actual < MIN_REQUIRED_FRAMES and duration > 0 and duration < 2.0:
                extract_all_frames(video_path, frames_dir, get_frame_ext(), get_frame_quality())
                files_after = list_frame_files(frames_dir)
                actual = len(files_after)
                # Regenerate timestamps for the newly extracted frames
                if duration > 0:
                    ts = uniform_timestamps(duration, actual)
                else:
                    ts = [float(i) * 0.033 for i in range(actual)]
            
            # Check frame count immediately - fail before creating meta.json
            try:
                _check_frame_count(actual, nb_frames=nb_frames)
            except ValueError:
                _cleanup_cache(frames_dir, meta_path, paths["root"])
                raise
            
            # Only create meta.json if we have enough frames
            parts = rel.parts
            label = None
            generator = None
            if parts:
                label = parts[0].lower()
                generator = parts[1] if len(parts) >= 3 else None
            
            meta.update({
                "video_id": video_id,
                "rel_path": rel.as_posix(),
                "file_name": video_path.name,
                "label": label,
                "generator": generator,
                "sha256": sha256_hash(video_path),
                "frame_timestamps_sec": ts[:actual],
            })
            _save_meta(meta_path, meta)
        else:
            # Existing cache: load and update if needed
            meta = _load_meta(meta_path)
            duration = float(meta.get("duration_sec") or 0.0)
            files_before = list_frame_files(frames_dir)

            # Recompute adaptive target in case duration changed
            target = _compute_target_frames(duration)

            if len(files_before) < target:
                # Special handling for 0-duration videos
                if duration <= 0:
                    extract_all_frames(video_path, frames_dir, target, get_frame_ext(), get_frame_quality())
                else:
                    ts = uniform_timestamps(duration, target)
                    ts = [min(t, duration - 0.01) for t in ts]
                    extract_frames(video_path, frames_dir, ts, get_frame_ext(), get_frame_quality())

                files_after = list_frame_files(frames_dir)
                actual = len(files_after)

                # Check frame count for cache update
                try:
                    _check_frame_count(actual)
                except ValueError:
                    _cleanup_cache(frames_dir, meta_path)
                    raise

                meta["frame_timestamps_sec"] = ts[:actual]
                _save_meta(meta_path, meta)
            else:
                # Ensure timestamps exist and align
                ts_list = meta.get("frame_timestamps_sec")
                if not isinstance(ts_list, list) or len(ts_list) < len(files_before):
                    meta["frame_timestamps_sec"] = uniform_timestamps(duration, len(files_before))
                    _save_meta(meta_path, meta)
        
        # Reload meta to get latest timestamps
        meta = _load_meta(meta_path)
        duration = float(meta.get("duration_sec") or 0.0)
        files = list_frame_files(frames_dir)
        ts_list = meta.get("frame_timestamps_sec")
        if not isinstance(ts_list, list) or len(ts_list) < len(files):
            meta["frame_timestamps_sec"] = uniform_timestamps(duration, len(files))
            _save_meta(meta_path, meta)
            ts_list = meta["frame_timestamps_sec"]
        
        # Build frames list
        frames = _build_frames_list(files, ts_list)
        
        # Final check: ensure we have enough frames (defensive check for existing cache)
        try:
            _check_frame_count(len(files))
        except ValueError:
            _cleanup_cache(frames_dir, meta_path)
            raise
        
        return {"meta": meta, "frames": frames}
    
    finally:
        if got_lock:
            release_lock(lock_path)


def _build_frame_inputs(frames: List[Dict[str, Any]], frames_dir: Path, include_data_urls: bool) -> List[Any]:
    """Helper function to build frame_inputs from frames list"""
    if include_data_urls:
        frame_inputs = []
        for fr in frames:
            fn = fr["file"]
            frame_path = frames_dir / fn
            if frame_path.exists():
                enc = base64.b64encode(frame_path.read_bytes()).decode("ascii")
                frame_mime = get_frame_mime()
                frame_inputs.append(f"data:{frame_mime};base64,{enc}")
            else:
                frame_inputs.append(None)
        return frame_inputs
    else:
        return [None] * len(frames)


def load_evidence(video_id: str, include_data_urls: bool = False) -> Dict[str, Any]:
    """
    Load evidence from cache (read-only, does not convert).
    
    This function only reads existing cache. For conversion, use build_evidence().
    
    Args:
        video_id: Base64-encoded relative path from data root
        include_data_urls: If True, include base64-encoded images
        
    Returns:
        Dict with 'meta', 'frames', and 'frame_inputs' keys
        
    Raises:
        FileNotFoundError: If evidence cache does not exist
    """
    paths = get_evidence_paths(video_id)
    meta_path = paths["meta"]
    frames_dir = paths["frames"]
    
    # Load files list first (safe even if directory doesn't exist)
    files = list_frame_files(frames_dir)
    
    # Check if cache exists and is valid (all three conditions in one if)
    if not meta_path.exists() or not frames_dir.exists() or len(files) < MIN_REQUIRED_FRAMES:
        raise FileNotFoundError(f"Evidence cache not found for video_id: {video_id}")
    
    # Load from cache
    meta = _load_meta(meta_path)
    frames = _build_frames_list(files, meta.get("frame_timestamps_sec", []))
    
    # Build frame_inputs
    frame_inputs = _build_frame_inputs(frames, frames_dir, include_data_urls)
    
    return {
        "meta": meta,
        "frames": frames,
        "frame_inputs": frame_inputs
    }


def build_llm_inputs(video_id: str, include_data_urls: bool = False) -> Dict[str, Any]:
    """
    Build LLM-ready inputs by converting video (for conversion phase).
    
    This function converts the video and returns LLM-ready inputs.
    For analysis phase (read-only), use load_evidence() instead.
    
    Args:
        video_id: Base64-encoded relative path from data root
        include_data_urls: If True, include base64-encoded images
        
    Returns:
        Dict with 'meta', 'frames', and 'frame_inputs' keys
    """
    # Convert video (build evidence cache)
    evidence = build_evidence(video_id)
    meta = evidence["meta"]
    frames = evidence["frames"]
    
    # Build frame_inputs
    frames_dir = get_evidence_paths(video_id)["frames"]
    frame_inputs = _build_frame_inputs(frames, frames_dir, include_data_urls)
    
    return {
        "meta": meta,
        "frames": frames,
        "frame_inputs": frame_inputs
    }
