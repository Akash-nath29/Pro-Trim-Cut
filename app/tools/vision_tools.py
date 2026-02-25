"""vision tools"""

from pathlib import Path

from langchain_core.tools import tool

from app.core.config import settings

_MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "face_landmarker.task"


@tool
def detect_face_presence(
    video_path: str,
    sample_fps: float = 2.0,
    video_fps: float = 30.0,
) -> list:
    """Sample video frames and check for faces using MediaPipe."""
    import cv2
    import numpy as np
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions
    from mediapipe.tasks.python.vision import (
        FaceLandmarker, FaceLandmarkerOptions, RunningMode,
    )

    frame_interval = max(1, int(video_fps / sample_fps))

    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(_MODEL_PATH)),
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
                    looking_score = compute_gaze_score.func(
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


@tool
def compute_gaze_score(face_landmarks: list, img_w: int, img_h: int) -> float:
    """Estimate how directly someone is looking at the camera (0=away, 1=direct)."""
    try:
        landmarks = face_landmarks
        nose_tip = landmarks[1]
        left_ear = landmarks[234]
        right_ear = landmarks[454]

        face_center_x = (left_ear.x + right_ear.x) / 2
        nose_deviation_x = abs(nose_tip.x - face_center_x)
        nose_deviation_y = abs(nose_tip.y - (left_ear.y + right_ear.y) / 2)

        horizontal_score = max(0, 1.0 - nose_deviation_x * 8)
        vertical_score = max(0, 1.0 - nose_deviation_y * 4)

        left_eye_w = abs(landmarks[33].x - landmarks[133].x)
        right_eye_w = abs(landmarks[263].x - landmarks[362].x)
        eye_symmetry = min(left_eye_w, right_eye_w) / max(left_eye_w, right_eye_w, 0.001)

        iris_score = 1.0
        if len(landmarks) > 473:
            left_center_x = (landmarks[133].x + landmarks[33].x) / 2
            right_center_x = (landmarks[362].x + landmarks[263].x) / 2
            left_iris_dev = abs(landmarks[468].x - left_center_x) / max(left_eye_w, 0.001)
            right_iris_dev = abs(landmarks[473].x - right_center_x) / max(right_eye_w, 0.001)
            iris_score = max(0, 1.0 - (left_iris_dev + right_iris_dev))

        score = (
            horizontal_score * 0.35 + vertical_score * 0.15
            + eye_symmetry * 0.25 + iris_score * 0.25
        )
        return min(1.0, max(0.0, score))
    except (IndexError, AttributeError):
        return 0.0


@tool
def motion_energy_analysis(frames: list) -> dict:
    """Aggregate motion stats from frame data."""
    if not frames:
        return {"avg_motion": 0.0, "max_motion": 0.0, "high_motion_ratio": 0.0}

    energies = [f.get("motion_energy", 0.0) for f in frames]
    avg = sum(energies) / len(energies)
    maximum = max(energies)
    threshold = settings.MOTION_ENERGY_THRESHOLD
    high_ratio = sum(1 for e in energies if e > threshold) / len(energies)

    return {
        "avg_motion": round(avg, 3),
        "max_motion": round(maximum, 3),
        "high_motion_ratio": round(high_ratio, 3),
    }


VISION_TOOLS = [detect_face_presence, compute_gaze_score, motion_energy_analysis]
