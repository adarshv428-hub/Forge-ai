import os
import json
import uuid
import subprocess
import traceback
from pathlib import Path
from typing import Optional
from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import yt_dlp
from google import genai
from google.genai import types

app = FastAPI(title="ClipForge AI Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

WORK_DIR = Path("/tmp/clipforge")
CLIPS_DIR = WORK_DIR / "clips"
WORK_DIR.mkdir(parents=True, exist_ok=True)
CLIPS_DIR.mkdir(parents=True, exist_ok=True)

app.mount("/clips", StaticFiles(directory=str(CLIPS_DIR)), name="clips")

JOBS = {}


class ClipRequest(BaseModel):
    url: str
    clip_count: int = 3
    target_duration: int = 45
    layout: str = "fill"
    keyword: Optional[str] = ""


def fmt_time(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m:02d}:{s:02d}"


def run_cmd(cmd: list):
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\n{res.stderr[-500:]}")
    return res.stdout


def process_video_job(job_id: str, req: ClipRequest):
    job_dir = WORK_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    try:
        gemini_key = os.environ.get("GEMINI_API_KEY", "").strip()
        if not gemini_key:
            raise RuntimeError("GEMINI_API_KEY is missing in Render Environment Variables.")
        client = genai.Client(api_key=gemini_key)

        # 1. Download video (Supports YouTube & Twitch VODs/Clips)
        JOBS[job_id].update(progress=15, message="Fetching video from YouTube / Twitch...")
        video_path = job_dir / "source.mp4"
        ydl_opts = {
            "format": "bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/best[height<=720][ext=mp4]/best",
            "outtmpl": str(video_path),
            "merge_output_format": "mp4",
            "quiet": True,
            "no_warnings": True,
            "max_filesize": 350 * 1024 * 1024,
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(req.url, download=True)
            video_title = info.get("title", "Untitled Video")
            total_duration = float(info.get("duration") or 0)

        JOBS[job_id]["title"] = video_title

        # 2. Extract lightweight MP3 audio for Gemini audio understanding
        JOBS[job_id].update(progress=40, message="Extracting audio track for Gemini AI...")
        audio_path = job_dir / "audio.mp3"
        run_cmd([
            "ffmpeg", "-y", "-i", str(video_path),
            "-vn", "-ac", "1", "-ar", "16000", "-b:a", "48k",
            str(audio_path)
        ])

        # 3. Upload audio & detect viral moments with Gemini 3.8 Flash
        JOBS[job_id].update(progress=62, message="Gemini AI listening & finding viral hooks...")
        uploaded_audio = client.files.upload(file=str(audio_path))

        keyword_rule = f"Prioritize moments related to: '{req.keyword}'." if req.keyword else ""
        prompt = f"""
You are a viral short-form video editor (TikTok, Reels, YouTube Shorts).
Listen to this audio track and identify the top {req.clip_count} most viral, high-retention, self-contained moments.
Each clip should be roughly {req.target_duration} seconds long (minimum 15s, maximum 75s).
Provide exact start and end timestamps in seconds (e.g., 14.5 and 58.0).
{keyword_rule}

Return ONLY valid JSON in this exact structure:
{{
  "clips": [
    {{
      "title": "Catchy hook title (max 8 words)",
      "start": 12.5,
      "end": 55.0,
      "score": 94,
      "reason": "Why this moment will go viral in 1 short sentence"
    }}
  ]
}}
"""
        response = client.models.generate_content(
            model="gemini-3.8-flash",
            contents=[uploaded_audio, prompt],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.4
            )
        )

        # Clean up remote file from Gemini storage
        try:
            client.files.delete(name=uploaded_audio.name)
        except Exception:
            pass

        ai_data = json.loads(response.text)
        raw_clips = ai_data.get("clips", [])[:req.clip_count]

        if not raw_clips:
            raise RuntimeError("AI could not identify clear clip segments.")

        # 4. Render 9:16 vertical clips with FFmpeg
        output_clips = []
        for idx, c in enumerate(raw_clips):
            pct = 75 + int(((idx + 1) / len(raw_clips)) * 22)
            JOBS[job_id].update(progress=pct, message=f"Rendering 9:16 vertical clip {idx + 1}/{len(raw_clips)}...")

            start_sec = max(0.0, float(c["start"]))
            end_sec = float(c["end"])
            if total_duration > 0:
                end_sec = min(total_duration, end_sec)
            duration = max(10.0, end_sec - start_sec)

            out_name = f"{job_id}_clip_{idx + 1}.mp4"
            out_path = CLIPS_DIR / out_name

            if req.layout == "blur":
                vf = (
                    "split[bg][fg];"
                    "[bg]scale=720:1280:force_original_aspect_ratio=increase,crop=720:1280,boxblur=20:10[bgb];"
                    "[fg]scale=720:-2[fgs];"
                    "[bgb][fgs]overlay=(W-w)/2:(H-h)/2"
                )
            else:
                vf = "scale=-2:1280,crop=720:1280:(iw-720)/2:0"

            run_cmd([
                "ffmpeg", "-y",
                "-ss", str(start_sec),
                "-t", str(duration),
                "-i", str(video_path),
                "-vf", vf,
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "24",
                "-c:a", "aac", "-b:a", "128k",
                "-movflags", "+faststart",
                str(out_path)
            ])

            output_clips.append({
                "title": c.get("title", f"Viral Clip #{idx + 1}"),
                "score": c.get("score", 88),
                "reason": c.get("reason", "High engagement hook."),
                "start_fmt": fmt_time(start_sec),
                "end_fmt": fmt_time(end_sec),
                "download_url": f"/clips/{out_name}"
            })

        JOBS[job_id].update(
            status="completed",
            progress=100,
            message="Done! All viral clips ready.",
            clips=output_clips
        )

    except Exception as e:
        traceback.print_exc()
        JOBS[job_id].update(status="failed", error=str(e))


@app.get("/")
def health():
    return {"status": "online", "service": "ClipForge AI (Gemini 3.8 Flash)"}


@app.post("/api/process")
def start_job(req: ClipRequest, background_tasks: BackgroundTasks):
    job_id = uuid.uuid4().hex[:10]
    JOBS[job_id] = {"status": "processing", "progress": 5, "message": "Queued...", "clips": []}
    background_tasks.add_task(process_video_job, job_id, req)
    return {"job_id": job_id}


@app.get("/api/status/{job_id}")
def get_status(job_id: str):
    if job_id not in JOBS:
        raise HTTPException(status_code=404, detail="Job not found")
    return JOBS[job_id]
        
