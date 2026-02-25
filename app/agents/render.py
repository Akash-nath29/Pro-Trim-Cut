"""
Render Agent — takes the edit plan and spits out the final video via FFmpeg.
Uses trim+concat filter for frame-accurate cuts with a file-based fallback.
"""

import subprocess
import shutil
from pathlib import Path

from app.agents.base import BaseAgent
from app.core.config import settings
from app.core.workspace import JobWorkspace


class RenderAgent(BaseAgent):

    @property
    def name(self) -> str:
        return "render"

    @property
    def output_artifact(self) -> str:
        return "render_result.json"

    def process(self, job_id: str, workspace: JobWorkspace) -> dict:
        edit_plan = workspace.load_artifact("edit_plan.json")
        metadata = workspace.load_artifact("metadata.json")

        keep_ranges = edit_plan["keep_ranges"]
        input_path = workspace.input_video
        output_path = workspace.output_video

        if not keep_ranges:
            self.logger.info(f"[{job_id}] No cuts needed — copying original video as output")
            shutil.copy2(str(input_path), str(output_path))
            output_size = output_path.stat().st_size
            output_duration = self._get_duration(output_path)
            return {
                "output_path": str(output_path),
                "output_size_bytes": output_size,
                "output_duration": round(output_duration, 3),
                "segments_rendered": 0,
                "input_duration": metadata["duration"],
                "note": "No edits needed — original video returned as-is",
            }
        if not input_path.exists():
            raise FileNotFoundError(f"Input video not found: {input_path}")

        self.logger.info(f"[{job_id}] Rendering {len(keep_ranges)} segments...")

        if len(keep_ranges) == 1:
            r = keep_ranges[0]
            self._trim_single(input_path, output_path, r["start"], r["end"], job_id)
        else:
            self._render_concat(input_path, output_path, keep_ranges, workspace, job_id)

        if not output_path.exists():
            raise RuntimeError("Render failed — no output file created")

        output_size = output_path.stat().st_size
        if output_size < 1000:
            raise RuntimeError(f"Output suspiciously small: {output_size} bytes")

        output_duration = self._get_duration(output_path)
        self.logger.info(f"[{job_id}] Done: {output_size / (1024*1024):.1f}MB, {output_duration:.1f}s")

        return {
            "output_path": str(output_path),
            "output_size_bytes": output_size,
            "output_duration": round(output_duration, 3),
            "segments_rendered": len(keep_ranges),
            "input_duration": metadata["duration"],
        }

    def _trim_single(self, input_path, output_path, start, end, job_id):
        cmd = [
            "ffmpeg", "-y", "-ss", str(start), "-i", str(input_path),
            "-t", str(end - start),
            "-c:v", settings.OUTPUT_VIDEO_CODEC, "-crf", str(settings.OUTPUT_CRF),
            "-preset", settings.OUTPUT_PRESET,
            "-c:a", settings.OUTPUT_AUDIO_CODEC, "-b:a", "128k",
            "-movflags", "+faststart", "-avoid_negative_ts", "make_zero",
            str(output_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg trim failed: {result.stderr[-200:]}")

    def _render_concat(self, input_path, output_path, keep_ranges, workspace, job_id):
        """Uses FFmpeg's trim+concat filter for clean, frame-accurate cuts."""
        n = len(keep_ranges)
        video_filters, audio_filters, concat_inputs = [], [], []

        for i, r in enumerate(keep_ranges):
            video_filters.append(f"[0:v]trim=start={r['start']}:end={r['end']},setpts=PTS-STARTPTS[v{i}]")
            audio_filters.append(f"[0:a]atrim=start={r['start']}:end={r['end']},asetpts=PTS-STARTPTS[a{i}]")
            concat_inputs.append(f"[v{i}][a{i}]")

        filter_complex = ";".join(video_filters + audio_filters + [
            "".join(concat_inputs) + f"concat=n={n}:v=1:a=1[outv][outa]"
        ])

        cmd = [
            "ffmpeg", "-y", "-i", str(input_path),
            "-filter_complex", filter_complex,
            "-map", "[outv]", "-map", "[outa]",
            "-c:v", settings.OUTPUT_VIDEO_CODEC, "-crf", str(settings.OUTPUT_CRF),
            "-preset", settings.OUTPUT_PRESET,
            "-c:a", settings.OUTPUT_AUDIO_CODEC, "-b:a", "128k",
            "-movflags", "+faststart", "-avoid_negative_ts", "make_zero",
            str(output_path),
        ]

        self.logger.info(f"[{job_id}] FFmpeg concat with {n} segments")
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            self.logger.warning(f"[{job_id}] Complex filter failed, trying segment fallback")
            self._render_fallback(input_path, output_path, keep_ranges, workspace, job_id)

    def _render_fallback(self, input_path, output_path, keep_ranges, workspace, job_id):
        """Fallback: extract each segment as a file, then concat them."""
        temp_dir = workspace.root / "temp_segments"
        temp_dir.mkdir(exist_ok=True)

        try:
            segment_files = []
            for i, r in enumerate(keep_ranges):
                seg_path = temp_dir / f"seg_{i:04d}.mp4"
                cmd = [
                    "ffmpeg", "-y", "-ss", str(r["start"]), "-i", str(input_path),
                    "-t", str(r["end"] - r["start"]),
                    "-c:v", settings.OUTPUT_VIDEO_CODEC, "-crf", str(settings.OUTPUT_CRF),
                    "-preset", settings.OUTPUT_PRESET,
                    "-c:a", settings.OUTPUT_AUDIO_CODEC, "-b:a", "128k",
                    "-avoid_negative_ts", "make_zero", str(seg_path),
                ]
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode == 0:
                    segment_files.append(seg_path)

            if not segment_files:
                raise RuntimeError("No segments extracted successfully")

            list_file = temp_dir / "concat_list.txt"
            with open(list_file, "w") as f:
                for seg in segment_files:
                    f.write(f"file '{str(seg).replace(chr(92), '/')}'\n")

            cmd = ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(list_file),
                   "-c", "copy", "-movflags", "+faststart", str(output_path)]

            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(f"Concat failed: {result.stderr[-200:]}")
        finally:
            if temp_dir.exists():
                shutil.rmtree(temp_dir)

    def _get_duration(self, path: Path) -> float:
        cmd = ["ffprobe", "-v", "quiet", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)]
        result = subprocess.run(cmd, capture_output=True, text=True)
        try:
            return float(result.stdout.strip())
        except ValueError:
            return 0.0
