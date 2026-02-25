"""
Ingest Tools — registered exclusively to the Ingest Agent.

Tools: extract_audio, get_video_metadata
"""

import json
import subprocess
from pathlib import Path

from langchain_core.tools import tool


@tool
def extract_audio(video_path: str, output_path: str) -> dict:
    """Extract 16kHz mono WAV audio from a video file using FFmpeg.

    Args:
        video_path: Absolute path to the input video file.
        output_path: Absolute path where the WAV file should be written.

    Returns:
        dict with keys: success, output_path, error
    """
    try:
        cmd = [
            "ffmpeg", "-i", video_path,
            "-vn", "-acodec", "pcm_s16le",
            "-ar", "16000", "-ac", "1",
            "-y", output_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            return {"success": False, "output_path": None, "error": result.stderr[-300:]}
        return {"success": True, "output_path": output_path, "error": None}
    except Exception as e:
        return {"success": False, "output_path": None, "error": str(e)}


@tool
def get_video_metadata(video_path: str) -> dict:
    """Probe a video file and return its technical metadata.

    Args:
        video_path: Absolute path to the video file.

    Returns:
        dict with fps, duration, width, height, video_codec, audio_codec, file_size_bytes
    """
    probe_cmd = [
        "ffprobe", "-v", "quiet", "-print_format", "json",
        "-show_format", "-show_streams", video_path,
    ]
    result = subprocess.run(probe_cmd, capture_output=True, text=True, check=True)
    probe_data = json.loads(result.stdout)

    video_stream = None
    audio_stream = None
    for stream in probe_data.get("streams", []):
        if stream["codec_type"] == "video" and video_stream is None:
            video_stream = stream
        elif stream["codec_type"] == "audio" and audio_stream is None:
            audio_stream = stream

    if video_stream is None:
        raise ValueError("No video stream found in file")

    fps_str = video_stream.get("r_frame_rate", "30/1")
    if "/" in fps_str:
        num, den = fps_str.split("/")
        fps = float(num) / float(den) if float(den) != 0 else 30.0
    else:
        fps = float(fps_str)

    return {
        "fps": round(fps, 3),
        "duration": round(float(probe_data.get("format", {}).get("duration", 0)), 3),
        "width": int(video_stream.get("width", 0)),
        "height": int(video_stream.get("height", 0)),
        "video_codec": video_stream.get("codec_name", "unknown"),
        "audio_codec": audio_stream.get("codec_name", "none") if audio_stream else "none",
        "file_size_bytes": int(probe_data.get("format", {}).get("size", 0)),
    }


# expose for agent registration
INGEST_TOOLS = [extract_audio, get_video_metadata]
