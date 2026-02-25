"""
Vision Intelligence Agent

Samples the video at low FPS and runs MediaPipe FaceLandmarker on each frame to figure out
if the speaker is looking at the camera, if their face is even visible, and how much
movement is happening. Simple but effective.
"""

import cv2
import numpy as np
from pathlib import Path

from app.agents.base import BaseAgent
from app.core.config import settings
from app.core.workspace import JobWorkspace

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "face_landmarker.task"


class VisionAgent(BaseAgent):

    @property
    def name(self) -> str:
        return "vision"

    @property
    def output_artifact(self) -> str:
        return "vision.json"

    def process(self, job_id: str, workspace: JobWorkspace) -> dict:
        input_path = workspace.input_video
        if not input_path.exists():
            raise FileNotFoundError(f"Video not found: {input_path}")

        metadata = workspace.load_artifact("metadata.json")
        video_fps = metadata["fps"]
        sample_fps = settings.VISION_SAMPLE_FPS
        frame_interval = max(1, int(video_fps / sample_fps))

        self.logger.info(f"[{job_id}] Sampling at {sample_fps} FPS (every {frame_interval} frames)")

        frames = self._analyze_frames(str(input_path), video_fps, frame_interval, job_id)

        avg_face = sum(1 for f in frames if f["face_present"]) / max(len(frames), 1)
        avg_gaze = sum(f["looking_at_camera_score"] for f in frames) / max(len(frames), 1)

        self.logger.info(f"[{job_id}] {len(frames)} frames, avg face: {avg_face:.2f}, avg gaze: {avg_gaze:.2f}")

        return {
            "frames": frames,
            "avg_face_presence": round(avg_face, 3),
            "avg_gaze_score": round(avg_gaze, 3),
        }

    def _analyze_frames(self, video_path: str, video_fps: float, frame_interval: int, job_id: str) -> list[dict]:
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python.vision import (
            FaceLandmarker,
            FaceLandmarkerOptions,
            RunningMode,
        )

        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
            running_mode=RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )

        landmarker = FaceLandmarker.create_from_options(options)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Can't open video: {video_path}")

        frames = []
        frame_idx = 0
        prev_gray = None

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx % frame_interval == 0:
                    timestamp = round(frame_idx / video_fps, 3)
                    timestamp_ms = int(frame_idx / video_fps * 1000)

                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                    results = landmarker.detect_for_video(mp_image, timestamp_ms)

                    face_present = len(results.face_landmarks) > 0
                    face_count = len(results.face_landmarks)

                    looking_score = 0.0
                    if face_present:
                        looking_score = self._estimate_gaze(
                            results.face_landmarks[0], frame.shape[1], frame.shape[0]
                        )

                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    gray = cv2.GaussianBlur(gray, (21, 21), 0)
                    motion_energy = 0.0
                    if prev_gray is not None:
                        motion_energy = float(np.mean(cv2.absdiff(prev_gray, gray)))
                    prev_gray = gray

                    frames.append({
                        "timestamp": timestamp,
                        "face_present": face_present,
                        "face_count": face_count,
                        "looking_at_camera_score": round(looking_score, 3),
                        "motion_energy": round(motion_energy, 3),
                    })

                frame_idx += 1
        finally:
            cap.release()
            landmarker.close()

        return frames

    def _estimate_gaze(self, face_landmarks, img_w: int, img_h: int) -> float:
        """
        Rough gaze estimation using face mesh landmarks.
        Checks how centered the nose is, whether both eyes are equally visible,
        and where the irises are pointing. Not perfect, but good enough for editing decisions.
        """
        landmarks = face_landmarks

        try:
            nose_tip = landmarks[1]
            left_ear = landmarks[234]
            right_ear = landmarks[454]

            face_center_x = (left_ear.x + right_ear.x) / 2

            # how far is the nose from center? further = looking away
            nose_deviation_x = abs(nose_tip.x - face_center_x)
            nose_deviation_y = abs(nose_tip.y - (left_ear.y + right_ear.y) / 2)

            horizontal_score = max(0, 1.0 - nose_deviation_x * 8)
            vertical_score = max(0, 1.0 - nose_deviation_y * 4)

            # eye symmetry — if both eyes are equally wide, you're probably facing the camera
            left_eye_w = abs(landmarks[33].x - landmarks[133].x)
            right_eye_w = abs(landmarks[263].x - landmarks[362].x)

            eye_symmetry = min(left_eye_w, right_eye_w) / max(left_eye_w, right_eye_w, 0.001)

            # iris centering
            iris_score = 1.0
            if len(landmarks) > 473:
                left_center_x = (landmarks[133].x + landmarks[33].x) / 2
                right_center_x = (landmarks[362].x + landmarks[263].x) / 2

                left_iris_dev = abs(landmarks[468].x - left_center_x) / max(left_eye_w, 0.001)
                right_iris_dev = abs(landmarks[473].x - right_center_x) / max(right_eye_w, 0.001)

                iris_score = max(0, 1.0 - (left_iris_dev + right_iris_dev))

            score = horizontal_score * 0.35 + vertical_score * 0.15 + eye_symmetry * 0.25 + iris_score * 0.25
            return min(1.0, max(0.0, score))

        except (IndexError, AttributeError):
            return 0.0
