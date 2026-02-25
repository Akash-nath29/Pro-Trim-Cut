import json
import subprocess
from pathlib import Path

from app.agents.base import BaseAgent
from app.core.workspace import JobWorkspace


class IngestAgent(BaseAgent):
    """Pulls apart the uploaded video — extracts audio and reads metadata via FFmpeg."""

    @property
    def name(self) -> str:
        return "ingest"

    @property
    def output_artifact(self) -> str:
        return "metadata.json"

    def process(self, job_id: str, workspace: JobWorkspace) -> dict:
        input_path = workspace.input_video
        audio_path = workspace.audio_path

        if not input_path.exists():
            raise FileNotFoundError(f"Input video not found: {input_path}")

        # probe metadata
        probe_cmd = [
            "ffprobe", "-v", "quiet", "-print_format", "json",
            "-show_format", "-show_streams", str(input_path),
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
            raise ValueError("No video stream found")

        fps_str = video_stream.get("r_frame_rate", "30/1")
        if "/" in fps_str:
            num, den = fps_str.split("/")
            fps = float(num) / float(den) if float(den) != 0 else 30.0
        else:
            fps = float(fps_str)

        duration = float(probe_data.get("format", {}).get("duration", 0))
        width = int(video_stream.get("width", 0))
        height = int(video_stream.get("height", 0))
        video_codec = video_stream.get("codec_name", "unknown")
        audio_codec = audio_stream.get("codec_name", "unknown") if audio_stream else "none"
        file_size = int(probe_data.get("format", {}).get("size", 0))

        self.logger.info(f"[{job_id}] Video: {width}x{height} @ {fps:.2f}fps, {duration:.1f}s, codec: {video_codec}")

        # extract audio as 16kHz mono wav for whisper
        if not audio_path.exists():
            extract_cmd = [
                "ffmpeg", "-i", str(input_path),
                "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
                "-y", str(audio_path),
            ]
            subprocess.run(extract_cmd, capture_output=True, check=True)
            self.logger.info(f"[{job_id}] Audio extracted: {audio_path}")

        return {
            "fps": round(fps, 3),
            "duration": round(duration, 3),
            "width": width,
            "height": height,
            "audio_path": str(audio_path),
            "video_codec": video_codec,
            "audio_codec": audio_codec,
            "file_size_bytes": file_size,
        }
