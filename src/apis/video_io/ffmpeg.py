"""ffmpeg wrapper for frame extraction"""
from pathlib import Path
from typing import List
from .probe import run_command


def extract_frames(
    video_path: Path,
    out_dir: Path,
    timestamps: List[float],
    frame_ext: str = "jpg",
    frame_quality: int = 2
):
    """Extract frames from video at specified timestamps"""
    out_dir.mkdir(parents=True, exist_ok=True)
    
    for i, t in enumerate(timestamps, start=1):
        out_path = out_dir / f"{i:06d}.{frame_ext}"
        if out_path.exists():
            continue
        
        # Primary command with better compatibility
        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-ss", f"{t:.3f}",
            "-i", str(video_path),
            "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2,format=yuvj420p",
            "-frames:v", "1",
            "-q:v", str(frame_quality),
            "-y",
            str(out_path),
        ]
        
        try:
            run_command(cmd)
            # Verify file was created
            if not out_path.exists() or out_path.stat().st_size == 0:
                # Fallback with simpler parameters
                cmd_fallback = [
                    "ffmpeg", "-hide_banner", "-loglevel", "error",
                    "-ss", f"{t:.3f}",
                    "-i", str(video_path),
                    "-vframes", "1",
                    "-q:v", str(frame_quality),
                    "-y",
                    str(out_path),
                ]
                run_command(cmd_fallback)
        except RuntimeError:
            # Skip failed frames, continue with others
            continue


def extract_all_frames(
    video_path: Path,
    out_dir: Path,
    frame_ext: str = "jpg",
    frame_quality: int = 2,
):
    """Extract all frames directly from video without timestamps."""
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(video_path),
        "-vf",
        "scale=trunc(iw/2)*2:trunc(ih/2)*2,format=yuvj420p",
        "-q:v",
        str(frame_quality),
        "-y",
        str(out_dir / f"%06d.{frame_ext}"),
    ]

    try:
        run_command(cmd)
    except RuntimeError:
        # Fallback with simpler parameters
        cmd_fallback = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(video_path),
            "-q:v",
            str(frame_quality),
            "-y",
            str(out_dir / f"%06d.{frame_ext}"),
        ]
        run_command(cmd_fallback)


def uniform_timestamps(duration: float, n: int) -> List[float]:
    """Generate n uniformly distributed timestamps across video duration (5%-95%)"""
    if duration <= 0 or n <= 1:
        return [0.0]
    start = duration * 0.05
    end = max(start, duration * 0.95)
    if end <= start:
        return [duration * 0.5]
    step = (end - start) / (n - 1)
    return [start + i * step for i in range(n)]
