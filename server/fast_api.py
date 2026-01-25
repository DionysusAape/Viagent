import os
import json
import base64
import hashlib
import subprocess
import time
from pathlib import Path
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles

# 导入统一配置
from src.config import (
    FRAME_EXT,
    FRAME_QUALITY,
    FRAME_MIME,
    CANONICAL_MAX_FRAMES,
    LOCK_WAIT_SEC,
    LOCK_POLL_SEC,
    ROOT_DIR,
    SOURCE_DIR,
    REAL_DIR,
    FAKE_DIR,
    CACHE_DIR,
    EVID_DIR,
)

CACHE_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Video Evidence API", version="1.0")
app.mount("/asset", StaticFiles(directory=str(CACHE_DIR)), name="asset")


# =========================
# Helpers
# =========================
def _run(cmd: List[str]) -> str:
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT)
        return out.decode("utf-8", errors="replace")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"Command failed: {' '.join(cmd)}\n{e.output.decode('utf-8', errors='replace')}"
        )

def _ensure_ffmpeg():
    for tool in ("ffprobe", "ffmpeg"):
        _run([tool, "-version"])

def _sha256(p: Path, chunk=1024 * 1024) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()

def _ffprobe_meta(video_path: Path) -> Dict[str, Any]:
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=codec_name,width,height,r_frame_rate,avg_frame_rate",
        "-show_entries", "format=duration,bit_rate,format_name",
        "-of", "json",
        str(video_path)
    ]
    j = json.loads(_run(cmd))
    stream = (j.get("streams") or [{}])[0]
    fmt = j.get("format") or {}

    def to_float(x, default=0.0):
        try:
            return float(x)
        except Exception:
            return default

    return {
        "codec": stream.get("codec_name"),
        "width": stream.get("width"),
        "height": stream.get("height"),
        "duration_sec": to_float(fmt.get("duration")),
        "bit_rate": fmt.get("bit_rate"),
        "format": fmt.get("format_name"),
        "r_frame_rate": stream.get("r_frame_rate"),
        "avg_frame_rate": stream.get("avg_frame_rate"),
    }

# video_id = base64url(relative_path_from_Source)
def _b64url_encode(s: str) -> str:
    return base64.urlsafe_b64encode(s.encode("utf-8")).decode("ascii").rstrip("=")

def _b64url_decode(s: str) -> str:
    pad = "=" * (-len(s) % 4)
    return base64.urlsafe_b64decode((s + pad).encode("ascii")).decode("utf-8")

def _make_video_id(rel_path: Path) -> str:
    return _b64url_encode(rel_path.as_posix())

def _parse_video_id(video_id: str) -> Path:
    rel = Path(_b64url_decode(video_id))
    if ".." in rel.parts or rel.is_absolute():
        raise ValueError("invalid video_id")
    return rel

def _resolve_video_from_id(video_id: str):
    rel = _parse_video_id(video_id)
    p = SOURCE_DIR / rel
    if not p.exists() or p.suffix.lower() != ".mp4":
        raise FileNotFoundError(video_id)
    return p, rel

def _ev_paths(video_id: str) -> Dict[str, Path]:
    root = EVID_DIR / video_id
    return {
        "root": root,
        "meta": root / "meta.json",
        "frames": root / "frames",
        "lock": root / ".lock"
    }

def _uniform_timestamps(duration: float, n: int) -> List[float]:
    if duration <= 0 or n <= 1:
        return [0.0]
    start = duration * 0.05
    end = max(start, duration * 0.95)
    if end <= start:
        return [duration * 0.5]
    step = (end - start) / (n - 1)
    return [start + i * step for i in range(n)]

def _extract_frames(video_path: Path, out_dir: Path, timestamps: List[float]):
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, t in enumerate(timestamps, start=1):
        out_path = out_dir / f"{i:06d}.{FRAME_EXT}"
        if out_path.exists():
            continue
        # 使用更兼容的 ffmpeg 参数，支持各种视频格式
        # 先解码为 RGB，再编码为 JPEG，避免像素格式兼容性问题
        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-ss", f"{t:.3f}",
            "-i", str(video_path),
            "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2,format=yuvj420p",  # 确保宽高是偶数并转换为 full-range YUV
            "-frames:v", "1",
            "-q:v", str(FRAME_QUALITY),
            "-y",  # 覆盖输出文件
            str(out_path),
        ]
        try:
            _run(cmd)
            # 验证文件是否成功生成
            if not out_path.exists() or out_path.stat().st_size == 0:
                # 如果文件不存在或为空，尝试使用更宽松的参数
                cmd_fallback = [
                    "ffmpeg", "-hide_banner", "-loglevel", "error",
                    "-ss", f"{t:.3f}",
                    "-i", str(video_path),
                    "-vframes", "1",
                    "-q:v", str(FRAME_QUALITY),
                    "-y",
                    str(out_path),
                ]
                _run(cmd_fallback)
        except RuntimeError:
            # 如果提取失败，跳过这一帧（不中断整个过程）
            # 这样即使某些帧失败，其他帧仍能成功提取
            continue

def _list_frame_files(frames_dir: Path) -> List[str]:
    if not frames_dir.exists():
        return []
    return sorted([p.name for p in frames_dir.glob(f"*.{FRAME_EXT}")])

def _read_as_data_url(p: Path) -> str:
    enc = base64.b64encode(p.read_bytes()).decode("ascii")
    return f"data:{FRAME_MIME};base64,{enc}"

# atomic lock
def _acquire_lock(lock_path: Path) -> bool:
    try:
        fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.close(fd)
        return True
    except FileExistsError:
        return False

def _release_lock(lock_path: Path):
    try:
        lock_path.unlink(missing_ok=True)
    except Exception:
        pass

def _wait_for_unlock_and_frames(frames_dir: Path, lock_path: Path, want: int) -> bool:
    deadline = time.time() + LOCK_WAIT_SEC
    while time.time() < deadline:
        have = len(_list_frame_files(frames_dir))
        if have >= want and not lock_path.exists():
            return True
        time.sleep(LOCK_POLL_SEC)
    return False

def _build_evidence(video_id: str) -> Dict[str, Any]:
    _ensure_ffmpeg()

    video_path, rel = _resolve_video_from_id(video_id)
    paths = _ev_paths(video_id)
    paths["root"].mkdir(parents=True, exist_ok=True)

    frames_dir = paths["frames"]
    meta_path = paths["meta"]
    lock_path = paths["lock"]

    target = CANONICAL_MAX_FRAMES

    # wait if another request is generating
    if lock_path.exists():
        _wait_for_unlock_and_frames(frames_dir, lock_path, want=target)

    got_lock = _acquire_lock(lock_path)
    try:
        # fast path: ready cache
        if meta_path.exists() and len(_list_frame_files(frames_dir)) >= 1:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            files = _list_frame_files(frames_dir)
            # Return available frames even if less than requested (video might be too short)
            if len(files) >= 1:
                ts_list = meta.get("frame_timestamps_sec", [])
                # Use all actual extracted frames from meta.json
                out = files
                frames = [
                    {
                        "index": i + 1,
                        "file": fn,
                        "timestamp_sec": float(ts_list[i]) if i < len(ts_list) else None,
                    }
                    for i, fn in enumerate(out)
                ]
                return {"meta": meta, "frames": frames}

        if not got_lock:
            ready = _wait_for_unlock_and_frames(frames_dir, lock_path, want=target)
            if ready and meta_path.exists():
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                files = _list_frame_files(frames_dir)
                if len(files) >= 1:
                    ts_list = meta.get("frame_timestamps_sec", [])
                    # Use all actual extracted frames from meta.json
                    out = files
                    frames = [
                        {
                            "index": i + 1,
                            "file": fn,
                            "timestamp_sec": float(ts_list[i]) if i < len(ts_list) else None,
                        }
                        for i, fn in enumerate(out)
                    ]
                    return {"meta": meta, "frames": frames}
            raise RuntimeError("Timeout waiting for evidence generation. Please retry.")

        # have lock: generate meta + frames once (fixed cache)
        if not meta_path.exists():
            meta = _ffprobe_meta(video_path)
            parts = rel.parts
            label = "real" if parts and parts[0].lower() == "real" else "fake"
            generator = parts[1] if label == "fake" and len(parts) >= 3 else None
            meta.update({
                "video_id": video_id,
                "rel_path": rel.as_posix(),
                "file_name": video_path.name,
                "label": label,
                "generator": generator,
                "sha256": _sha256(video_path),
            })
            meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        else:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))

        duration = float(meta.get("duration_sec") or 0.0)
        files_before = _list_frame_files(frames_dir)

        if len(files_before) < target:
            ts = _uniform_timestamps(duration, target)
            # 确保时间戳不超过视频长度（避免最后一帧提取失败）
            ts = [min(t, duration - 0.01) for t in ts]  # 留0.01秒缓冲
            _extract_frames(video_path, frames_dir, ts)

            files_after = _list_frame_files(frames_dir)
            actual = len(files_after)

            # IMPORTANT: timestamps must match actual extracted frames (Fix #1)
            # 只保存成功提取的帧的时间戳
            meta["frame_timestamps_sec"] = ts[:actual]
            meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        else:
            # ensure timestamps exist and align
            ts_list = meta.get("frame_timestamps_sec")
            if not isinstance(ts_list, list) or len(ts_list) < len(files_before):
                meta["frame_timestamps_sec"] = _uniform_timestamps(duration, len(files_before))
                meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

        # Reload meta to get latest timestamps (in case we just updated them)
        # Handle case where meta.json might not exist (incomplete conversion)
        if not meta_path.exists():
            # This shouldn't happen, but if it does, regenerate meta
            meta = _ffprobe_meta(video_path)
            parts = rel.parts
            label = "real" if parts and parts[0].lower() == "real" else "fake"
            generator = parts[1] if label == "fake" and len(parts) >= 3 else None
            meta.update({
                "video_id": video_id,
                "rel_path": rel.as_posix(),
                "file_name": video_path.name,
                "label": label,
                "generator": generator,
                "sha256": _sha256(video_path),
            })
            duration = float(meta.get("duration_sec") or 0.0)
            files = _list_frame_files(frames_dir)
            if files:
                meta["frame_timestamps_sec"] = _uniform_timestamps(duration, len(files))
            meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        else:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        
        # Ensure timestamps exist if meta.json exists but lacks them
        duration = float(meta.get("duration_sec") or 0.0)
        files = _list_frame_files(frames_dir)
        ts_list = meta.get("frame_timestamps_sec")
        if not isinstance(ts_list, list) or len(ts_list) < len(files):
            # Regenerate timestamps if missing or incomplete
            meta["frame_timestamps_sec"] = _uniform_timestamps(duration, len(files))
            meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
            ts_list = meta["frame_timestamps_sec"]
        
        # Use all actual extracted frames from meta.json
        out = files
        frames = [
            {
                "index": i + 1,
                "file": fn,
                "timestamp_sec": float(ts_list[i]) if i < len(ts_list) else None,
            }
            for i, fn in enumerate(out)
        ]
        return {"meta": meta, "frames": frames}

    finally:
        if got_lock:
            _release_lock(lock_path)

def _iter_real(limit: int, offset: int = 0):
    if not REAL_DIR.exists():
        return
    count = 0
    for p in sorted(REAL_DIR.glob("*.mp4")):
        if count < offset:
            count += 1
            continue
        yield p.relative_to(SOURCE_DIR)
        limit -= 1
        if limit <= 0:
            break

def _iter_fake(generator: Optional[str], limit: int, offset: int = 0):
    if not FAKE_DIR.exists():
        return
    gens = [generator] if generator else [d.name for d in sorted(FAKE_DIR.iterdir()) if d.is_dir()]
    count = 0
    for g in gens:
        d = FAKE_DIR / g
        if not d.exists():
            continue
        for p in sorted(d.glob("*.mp4")):
            if count < offset:
                count += 1
                continue
            yield p.relative_to(SOURCE_DIR)
            limit -= 1
            if limit <= 0:
                return


# =========================
# API
# =========================
@app.get("/ping")
def ping():
    return {
        "ok": True,
        "root": str(ROOT_DIR),
        "source_dir_exists": SOURCE_DIR.exists(),
        "canonical_max_frames": CANONICAL_MAX_FRAMES,
        "real_count_hint": len(list(REAL_DIR.glob("*.mp4"))) if REAL_DIR.exists() else 0,
        "fake_generators_hint": [d.name for d in sorted(FAKE_DIR.iterdir()) if d.is_dir()] if FAKE_DIR.exists() else [],
    }

@app.get("/videos")
def videos(label: str = "all", generator: Optional[str] = None, limit: int = 50, offset: int = 0):
    """
    获取视频列表
    - label: all | real | fake
    - generator: 假视频生成器名称（仅用于 fake）
    - limit: 返回数量（1-500）
    - offset: 跳过数量（用于分页）
    """
    if label not in ("all", "real", "fake"):
        raise HTTPException(status_code=400, detail="label must be all|real|fake")
    if limit < 1 or limit > 500:
        raise HTTPException(status_code=400, detail="limit must be 1..500")
    if offset < 0:
        raise HTTPException(status_code=400, detail="offset must be >= 0")

    items = []

    if label in ("all", "real"):
        # 对于 "all"，需要先计算 real 部分的数量来确定 fake 的 offset
        if label == "all":
            # 先获取所有 real 视频来计算数量
            real_count = len(list(_iter_real(999999, offset=0)))
            if offset < real_count:
                # offset 在 real 范围内
                real_offset = offset
                real_limit = min(limit, real_count - offset)
                for rel in _iter_real(real_limit, offset=real_offset):
                    items.append({
                        "video_id": _make_video_id(rel),
                        "label": "real",
                        "generator": None,
                        "file": rel.name,
                        "rel_path": rel.as_posix(),
                    })
                    if len(items) >= limit:
                        return {"count": len(items), "items": items}
            # 如果还需要更多，从 fake 获取
            if len(items) < limit:
                fake_offset = max(0, offset - real_count)
                remaining = limit - len(items)
                for rel in _iter_fake(generator, remaining, offset=fake_offset):
                    parts = rel.parts
                    gen = parts[1] if len(parts) >= 3 else None
                    items.append({
                        "video_id": _make_video_id(rel),
                        "label": "fake",
                        "generator": gen,
                        "file": rel.name,
                        "rel_path": rel.as_posix(),
                    })
                    if len(items) >= limit:
                        break
        else:
            # label == "real"
            for rel in _iter_real(limit, offset=offset):
                items.append({
                    "video_id": _make_video_id(rel),
                    "label": "real",
                    "generator": None,
                    "file": rel.name,
                    "rel_path": rel.as_posix(),
                })
                if len(items) >= limit:
                    break

    elif label == "fake":
        for rel in _iter_fake(generator, limit, offset=offset):
            parts = rel.parts
            gen = parts[1] if len(parts) >= 3 else None
            items.append({
                "video_id": _make_video_id(rel),
                "label": "fake",
                "generator": gen,
                "file": rel.name,
                "rel_path": rel.as_posix(),
            })
            if len(items) >= limit:
                break

    return {"count": len(items), "items": items}

@app.get("/videos/{video_id}/evidence")
def evidence(request: Request, video_id: str, include_data_urls: bool = False):
    try:
        ev = _build_evidence(video_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"invalid video_id: {e}")
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=f"video_id not found: {e}")
    except RuntimeError as e:
        # 提供更详细的错误信息
        error_msg = str(e)
        raise HTTPException(status_code=500, detail=f"Internal server error: {error_msg}")
    except Exception as e:
        # 捕获所有其他异常，避免暴露内部错误
        raise HTTPException(status_code=500, detail=f"Unexpected error: {type(e).__name__}: {str(e)}")

    base = str(request.base_url).rstrip("/")
    frames_dir = _ev_paths(video_id)["frames"]

    frames = []
    for fr in ev["frames"]:
        fn = fr["file"]
        url = f"{base}/asset/evidence/{video_id}/frames/{fn}"
        item = {**fr, "url": url}
        if include_data_urls:
            item["data_url"] = _read_as_data_url(frames_dir / fn)
        frames.append(item)

    return {"video_id": video_id, "meta": ev["meta"], "frames": frames}
