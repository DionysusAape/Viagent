"""ffprobe wrapper for video metadata extraction"""
import json
import subprocess
from pathlib import Path
from typing import Dict, Any


def run_command(cmd: list[str]) -> str:
    """Run shell command and return stdout"""
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT)
        return out.decode("utf-8", errors="replace")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"Command failed: {' '.join(cmd)}\n{e.output.decode('utf-8', errors='replace')}"
        )


def ensure_ffmpeg():
    """Check if ffmpeg and ffprobe are available"""
    for tool in ("ffprobe", "ffmpeg"):
        run_command([tool, "-version"])


def get_video_metadata(video_path: Path) -> Dict[str, Any]:
    """Extract video metadata using ffprobe"""
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=codec_name,width,height,r_frame_rate,avg_frame_rate,nb_frames",
        "-show_entries", "format=duration,bit_rate,format_name",
        "-of", "json",
        str(video_path)
    ]
    j = json.loads(run_command(cmd))
    stream = (j.get("streams") or [{}])[0]
    fmt = j.get("format") or {}

    def to_float(x, default=0.0):
        try:
            return float(x)
        except Exception:
            return default
    
    def to_int(x, default=0):
        try:
            return int(x)
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
        "nb_frames": to_int(stream.get("nb_frames")),
    }
