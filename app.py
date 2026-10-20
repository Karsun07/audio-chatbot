import os
import json
import time
import logging

import requests
import yt_dlp
import streamlit as st
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama, OllamaEmbeddings

import bisect

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

# ---------- RAG: build the searchable index ----------
LLM_MODEL = "llama3.2"
EMBED_MODEL = "nomic-embed-text"


def build_vectorstore(text):
    # 1) Split the transcript into small overlapping chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=30, add_start_index=True)
    chunks = splitter.create_documents([text])
    
    # 2) Turn every chunk into numbers (embeddings) and store them in FAISS
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)
    vectorstore = FAISS.from_documents(chunks, embeddings)

    logger.info(f"Indexed {len(chunks)} chunks")
    return vectorstore


# ---------- RAG: answer a question ----------
def answer_question(vectorstore, question):
    # 1) RETRIEVE: find the 3 chunks closest in meaning to the question
    docs = vectorstore.similarity_search(question, k=3)
    context = "\n\n".join(d.page_content for d in docs)

    # 2) AUGMENT: put those chunks into the prompt
    prompt = f"""You are a helpful assistant. Answer the question using ONLY the
transcript excerpts below. Reply in 1-3 sentences in your own words. Do not
copy the excerpts. If the answer is not in the excerpts, say
"I could not find that in the audio."

Transcript excerpts:
{context}

Question: {question}

Answer:"""

    # 3) GENERATE: let Llama write the answer
    llm = ChatOllama(model=LLM_MODEL, temperature=0)
    response = llm.invoke(prompt)
    return response.content, docs


# ---------- Summary ----------
def generate_summary(text):
    llm = ChatOllama(model=LLM_MODEL, temperature=0.3)
    prompt = f"Summarize the following transcript in 3-5 sentences:\n\n{text}"
    return llm.invoke(prompt).content

# ---------- Timestamps ----------
def build_char_offsets(text, words):
    """For each word, find where it starts inside the full transcript text."""
    offsets = []
    cursor = 0
    for w in words:
        pos = text.find(w["text"], cursor)
        if pos != -1:
            offsets.append(pos)
            cursor = pos + len(w["text"])
        else:
            offsets.append(cursor)  # fallback if a word can't be matched
    return offsets


def find_relevant_timestamps(docs, text, words):
    """Map retrieved chunks to the time (in seconds) they were spoken."""
    offsets = build_char_offsets(text, words)
    seconds = []
    for d in docs:
        start_index = d.metadata.get("start_index", 0)
        i = bisect.bisect_right(offsets, start_index) - 1
        i = max(i, 0)
        seconds.append(words[i]["start"] // 1000)  # milliseconds -> seconds
    return sorted(set(seconds))


def format_time(s):
    return f"{s // 60}:{s % 60:02d}"

# ---------- TEMPORARY TEST (delete later) ----------
if __name__ == "__main__":
    with open("docs/transcription.txt", "r", encoding="utf-8") as f:
        text = f.read()
    with open("docs/word_timestamps.json", "r", encoding="utf-8") as f:
        words = json.load(f)

    vs = build_vectorstore(text)


    for q in ["Who scored the spectacular rocket goal?",
              "What was significant about James Maddison's goal according to the commentary?",
              "Who scored the final goal to cap off the match?"]:
        answer, docs = answer_question(vs, q)
        times = find_relevant_timestamps(docs, text, words)
        print(f"\nQ: {q}\nA: {answer}")
        print("Timestamps:", [format_time(t) for t in times])