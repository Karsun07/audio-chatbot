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


# ---------- TEMPORARY TEST (delete later) ----------
if __name__ == "__main__":
    print("Key loaded:", ASSEMBLY_AI_KEY is not None)
    name = save_audio("https://www.youtube.com/watch?v=pDyaurLPeko")
    print("Saved file:", name)