import os
import json
import time
import logging

import requests
import yt_dlp
import streamlit as st
from dotenv import load_dotenv

# ---------- Setup ----------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()
ASSEMBLY_AI_KEY = os.getenv("ASSEMBLY_AI_KEY")
BASE_URL = "https://api.assemblyai.com/v2"
HEADERS = {"authorization": ASSEMBLY_AI_KEY}


# ---------- Download audio from YouTube ----------
def save_audio(url):
    try:
        os.makedirs("temp", exist_ok=True)

        ydl_opts = {
            "format": "bestaudio/best",
            "postprocessors": [{
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "192",
            }],
            "outtmpl": "temp/%(id)s.%(ext)s",
            "noplaylist": True,
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            audio_filename = f"{info['id']}.mp3"

        logger.info(f"Downloaded audio: {audio_filename}")
        return audio_filename

    except Exception as e:
        logger.error(f"Error downloading audio: {e}")
        return None

# ---------- Convert audio to text ----------
def assemblyai_stt(audio_filename):
    try:
        audio_path = os.path.join("temp", audio_filename)

        # 1) Upload the audio file to AssemblyAI
        with open(audio_path, "rb") as f:
            response = requests.post(BASE_URL + "/upload", headers=HEADERS, data=f)
        response.raise_for_status()
        upload_url = response.json()["upload_url"]

        # 2) Ask AssemblyAI to transcribe the uploaded file
        response = requests.post(
            BASE_URL + "/transcript",
            json={"audio_url": upload_url},
            headers=HEADERS,
        )
        response.raise_for_status()
        transcript_id = response.json()["id"]

        # 3) Keep checking until the transcript is ready
        polling_url = f"{BASE_URL}/transcript/{transcript_id}"
        while True:
            result = requests.get(polling_url, headers=HEADERS).json()
            if result["status"] == "completed":
                break
            elif result["status"] == "error":
                raise RuntimeError(f"Transcription failed: {result['error']}")
            time.sleep(3)

        text = result["text"]
        words = result["words"]

        # 4) Save results to disk
        os.makedirs("docs", exist_ok=True)
        with open("docs/transcription.txt", "w", encoding="utf-8") as f:
            f.write(text)
        with open("docs/word_timestamps.json", "w", encoding="utf-8") as f:
            json.dump(words, f)

        logger.info("Transcription saved")
        return text, words

    except Exception as e:
        logger.error(f"Error in speech-to-text: {e}")
        return None, None
    
# ---------- TEMPORARY TEST (delete later) ----------
if __name__ == "__main__":
    name = save_audio("https://www.youtube.com/watch?v=pDyaurLPeko")
    print("Saved file:", name)
    if name:
        text, words = assemblyai_stt(name)
        print("Transcript:", text)
        print("Number of words with timestamps:", len(words) if words else 0)