"""render tools"""

import shutil
import subprocess
from pathlib import Path

from langchain_core.tools import tool

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger("tools.render")


@tool
def render_final_video(
    job_id: str,
    input_path: str,
    output_path: str,
    keep_ranges: list,
    jobs_dir: str,
) -> dict:
    """Render the final trimmed video. Re-encodes segments for frame-accurate cuts."""
    out = Path(output_path)

    if not keep_ranges:
        shutil.copy2(input_path, output_path)
        size = out.stat().st_size
        return {
            "output_path": output_path,
            "output_size_bytes": size,
            "output_duration": round(_get_duration(output_path), 3),
            "segments_rendered": 0,
            "note": "No edits needed — original video returned as-is",
        }

    if not Path(input_path).exists():
        raise FileNotFoundError(f"Input video not found: {input_path}")

    if len(keep_ranges) == 1:
        r = keep_ranges[0]
        _trim_single(input_path, output_path, r["start"], r["end"])
    else:
        _render_segments(input_path, output_path, keep_ranges, jobs_dir, job_id)

    if not out.exists():
        raise RuntimeError("Render failed — no output file created")

    size = out.stat().st_size
    if size < 1000:
        raise RuntimeError(f"Output suspiciously small: {size} bytes")

    return {
        "output_path": output_path,
        "output_size_bytes": size,
        "output_duration": round(_get_duration(output_path), 3),
        "segments_rendered": len(keep_ranges),
    }


def _trim_single(input_path: str, output_path: str, start: float, end: float) -> None:
    cmd = [
        "ffmpeg", "-y", "-i", input_path,
        "-ss", str(start), "-t", str(end - start),
        "-c:v", "libx264", "-crf", "18", "-preset", "fast",
        "-c:a", "aac", "-b:a", "128k", "-strict", "-2",
        "-movflags", "+faststart",
        output_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(f"FFmpeg trim failed: {result.stderr[-500:]}")
        raise RuntimeError(f"FFmpeg trim failed: {result.stderr[-200:]}")


def _render_segments(
    input_path: str, output_path: str,
    keep_ranges: list, jobs_dir: str, job_id: str,
) -> None:
    """Extract each segment, then concat them together."""
    temp_dir = Path(jobs_dir) / job_id / "temp_segments"
    temp_dir.mkdir(parents=True, exist_ok=True)
    try:
        segment_files = []
        for i, r in enumerate(keep_ranges):
            seg_path = temp_dir / f"seg_{i:04d}.mp4"
            cmd = [
                "ffmpeg", "-y", "-i", input_path,
                "-ss", str(r["start"]), "-t", str(r["end"] - r["start"]),
                "-c:v", "libx264", "-crf", "18", "-preset", "fast",
                "-c:a", "aac", "-b:a", "128k", "-strict", "-2",
                str(seg_path),
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                segment_files.append(seg_path)
            else:
                logger.error(f"Segment {i} failed: {result.stderr[-300:]}")

        if not segment_files:
            raise RuntimeError("No segments extracted successfully")

        list_file = temp_dir / "concat_list.txt"
        with open(list_file, "w") as f:
            for seg in segment_files:
                f.write(f"file '{str(seg).replace(chr(92), '/')}'\n")

        cmd = [
            "ffmpeg", "-y", "-f", "concat", "-safe", "0",
            "-i", str(list_file), "-c", "copy",
            "-movflags", "+faststart", output_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"Concat failed: {result.stderr[-200:]}")
    finally:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)


def _get_duration(path: str) -> float:
    cmd = ["ffprobe", "-v", "quiet", "-show_entries", "format=duration", "-of", "csv=p=0", path]
    result = subprocess.run(cmd, capture_output=True, text=True)
    try:
        return float(result.stdout.strip())
    except ValueError:
        return 0.0


RENDER_TOOLS = [render_final_video]
