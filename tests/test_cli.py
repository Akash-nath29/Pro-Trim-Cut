"""
CLI test script — uploads a video, polls until done, downloads the result.

Usage:
    python tests/test_cli.py --video path/to/test.mp4
"""

import argparse
import os
import sys
import time
import requests


def main():
    parser = argparse.ArgumentParser(description="Pro Auto-Trim CLI Tester")
    parser.add_argument("--video", "-v", required=True, help="Path to input video")
    parser.add_argument("--server", "-s", default="http://localhost:8000")
    parser.add_argument("--output", "-o", default="trimmed_output.mp4")
    parser.add_argument("--poll-interval", type=int, default=5)
    args = parser.parse_args()

    server = args.server.rstrip("/")

    # health check
    print(f"🔍 Checking server at {server}...")
    try:
        r = requests.get(f"{server}/health", timeout=5)
        r.raise_for_status()
        h = r.json()
        print(f"✅ Server healthy — device: {h['device']}, whisper: {h['whisper_model']}, llm: {h.get('llm_model', 'n/a')}")
    except Exception as e:
        print(f"❌ Server not reachable: {e}")
        sys.exit(1)

    # upload
    video_path = args.video
    if not os.path.exists(video_path):
        print(f"❌ File not found: {video_path}")
        sys.exit(1)

    size_mb = os.path.getsize(video_path) / (1024 * 1024)
    print(f"\n📤 Uploading {video_path} ({size_mb:.1f}MB)...")

    with open(video_path, "rb") as f:
        r = requests.post(f"{server}/auto-trim", files={"file": (os.path.basename(video_path), f, "video/mp4")}, timeout=60)

    if r.status_code != 200:
        print(f"❌ Upload failed: {r.status_code} — {r.text}")
        sys.exit(1)

    job_id = r.json()["job_id"]
    print(f"✅ Job created: {job_id}")

    # poll
    print(f"\n⏳ Processing... (polling every {args.poll_interval}s)")
    start_time = time.time()

    while True:
        time.sleep(args.poll_interval)
        r = requests.get(f"{server}/jobs/{job_id}", timeout=10)
        if r.status_code != 200:
            continue

        s = r.json()
        elapsed = time.time() - start_time
        print(f"  [{elapsed:6.1f}s] {s['stage']:20s} | {s['progress']:5.1f}% | {s['message']}")

        if s["stage"] == "completed":
            print(f"\n🎉 Done!")
            if s.get("timings"):
                print(f"   Timings: {s['timings']}")
            break
        if s["stage"] == "failed":
            print(f"\n❌ Failed: {s.get('error', 'unknown')}")
            sys.exit(1)

    # download
    print(f"\n📥 Downloading...")
    r = requests.get(f"{server}/jobs/{job_id}/download", timeout=120, stream=True)
    if r.status_code != 200:
        print(f"❌ Download failed: {r.status_code}")
        sys.exit(1)

    with open(args.output, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)

    print(f"✅ Saved: {args.output} ({os.path.getsize(args.output) / (1024*1024):.1f}MB)")
    print(f"🎬 Total time: {time.time() - start_time:.1f}s")


if __name__ == "__main__":
    main()
