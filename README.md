# Pro Auto-Trim — Human-Like AI Video Editor

A multi-agent pipeline that automatically edits raw talking-head videos into clean, professional cuts — removing pauses, filler words, repeated takes, rehearsal lines, and dead camera time.

## Architecture

```
Upload MP4
    │
    ▼
┌──────────────┐
│ Ingest Agent │  ── Extract audio, detect FPS/duration
└──────┬───────┘
       │
       ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────────┐
│ Speech Agent │  │ Vision Agent │  │ Semantic Agent   │
│ (WhisperX)   │  │ (MediaPipe)  │  │ (Transformers)   │
└──────┬───────┘  └──────┬───────┘  └───────┬──────────┘
       │                 │                  │
       └────────┬────────┘──────────────────┘
                │
                ▼
       ┌─────────────────┐
       │ Timeline Builder│  ── Merge all signals
       └────────┬────────┘
                │
                ▼
       ┌────────────────┐
       │  Cut Planner   │  ── Score & select best takes
       └────────┬───────┘
                │
                ▼
       ┌────────────────┐
       │  Render Agent  │  ── FFmpeg concat → final.mp4
       └────────────────┘
```

## Requirements

- **Python** 3.10+
- **FFmpeg** installed and on PATH
- **GPU** optional (auto-detects CUDA; falls back to CPU)

## Quick Setup

```bash
# 1. Clone the repo
git clone https://github.com/Akash-nath29/Pro-Trim-Cut.git
cd Pro-Trim-Cut

# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set environment variables (optional — defaults work)
# Edit .env file:
#   WHISPER_MODEL_SIZE=base    # tiny/base/small/medium/large-v2
#   DEVICE=auto                # auto/cpu/cuda
#   CUT_MODE=deterministic     # deterministic/llm
#   GITHUB_TOKEN=your_token    # required only for LLM mode
```

## Run the Server

```bash
uvicorn app.main:app --reload
```

Server starts at **http://localhost:8000**

- **Web UI:** http://localhost:8000/ — upload, preview, trim, and download right in the browser
- Swagger UI: http://localhost:8000/docs
- Health check: http://localhost:8000/health

## Usage

### Web Interface (Recommended)

Just open **http://localhost:8000** in your browser.

1. **Drop or select** your video file
2. **Preview** the original footage
3. Click **Auto Trim (Pro)**
4. Watch the **progress bar** as the AI agents process your video
5. **Play** the trimmed result to verify
6. **Download** the final cut

### API (Postman)

Once the server is running (`uvicorn app.main:app --reload`), you can also test the API directly with Postman.

### 1. Upload & Start Auto-Trim
- **Method:** `POST`
- **URL:** `http://localhost:8000/auto-trim`
- **Body:** Select the `form-data` tab.
  - Hover over the `KEY` column, click the hidden dropdown, and change it from `Text` to `File`.
  - Enter the key name: `file`
  - Under the `VALUE` column, click **Select Files** and choose your MP4 video.
- **Hit Send**. You will immediately get a response with a `job_id`:
  ```json
  {
    "job_id": "a1b2c3d4"
  }
  ```

### 2. Poll Job Status
The video goes through a multi-agent pipeline which takes some time. Poll the status using your `job_id`.
- **Method:** `GET`
- **URL:** `http://localhost:8000/jobs/a1b2c3d4` (replace with your actual `job_id`)
- **Hit Send**. This shows the real-time progress:
  ```json
  {
    "job_id": "a1b2c3d4",
    "stage": "quality_review",
    "progress": 82.0,
    "message": "Senior AI editor reviewing the cut...",
    "output_ready": false
  }
  ```
- Keep sending the request dynamically until `"output_ready": true` and `"stage": "completed"`.

### 3. Download the Final Video
Once completed, download the rendered video.
- **Method:** `GET`
- **URL:** `http://localhost:8000/jobs/a1b2c3d4/download`
- **Download:** In Postman, click the arrow next to the **Send** button and choose **Send and Download**. (Or hit Send normally, then click **Save Response** -> **Save to a file** in the response pane).

---

### Command Line (cURL) Alternative
If you prefer the terminal:
```bash
# Upload
curl -X POST "http://localhost:8000/auto-trim" -F "file=@my_video.mp4"

# Check Status
curl "http://localhost:8000/jobs/a1b2c3d4"

# Download
curl -o trimmed.mp4 "http://localhost:8000/jobs/a1b2c3d4/download"
```

## Configuration

All settings via environment variables or `.env` file:

| Variable | Default | Description |
|---|---|---|
| `WHISPER_MODEL_SIZE` | `base` | Whisper model (tiny/base/small/medium/large-v2) |
| `DEVICE` | `auto` | Compute device (auto/cpu/cuda) |
| `CUT_MODE` | `deterministic` | Cut planning mode (deterministic/llm) |
| `PAUSE_MIN_DURATION` | `0.8` | Min pause duration to flag (seconds) |
| `SIMILARITY_THRESHOLD` | `0.82` | Cosine similarity for duplicate detection |
| `MIN_SEGMENT_SCORE` | `0.4` | Below this score, segments get cut |
| `CUT_PADDING_MS` | `50` | Padding at cut points (ms) |
| `OUTPUT_CRF` | `23` | FFmpeg quality (lower = better) |
| `GITHUB_TOKEN` | — | Required for LLM cut planner mode |

## Project Structure

```
app/
├── api/routes.py            # FastAPI endpoints
├── main.py                  # App factory + UI route
├── static/
│   └── index.html           # Web interface
├── agents/
│   ├── base.py              # Abstract base agent
│   ├── ingest.py            # Audio extraction + metadata
│   ├── speech.py            # WhisperX transcription + VAD
│   ├── vision.py            # MediaPipe face/gaze analysis
│   ├── semantic.py          # Sentence embeddings + clustering
│   ├── timeline_builder.py  # Signal merger
│   ├── cut_planner.py       # Scoring + LLM planning
│   └── render.py            # FFmpeg rendering
├── orchestrator/
│   └── engine.py            # Pipeline state machine
├── schemas/
│   ├── job.py               # Job status models
│   └── agents.py            # Agent I/O models
└── core/
    ├── config.py             # Environment config
    ├── logging.py            # Structured logging
    └── workspace.py          # Per-job file management
```
