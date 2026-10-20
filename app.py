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

# ---------- Cleanup ----------
def cleanup_temp_files():
    if os.path.isdir("temp"):
        for f in os.listdir("temp"):
            try:
                os.remove(os.path.join("temp", f))
            except OSError:
                pass


# ---------- Run the whole pipeline for one video ----------
def process_video(url):
    with st.status("Processing video...", expanded=True) as status:
        st.write("⬇️ Downloading audio...")
        audio = save_audio(url)
        if not audio:
            status.update(label="Could not download audio", state="error")
            return False

        st.write("📝 Transcribing speech (this can take a minute)...")
        text, words = assemblyai_stt(audio)
        if not text:
            status.update(label="Transcription failed", state="error")
            return False

        st.write("🧠 Building the search index...")
        vectorstore = build_vectorstore(text)

        st.write("✨ Writing a summary...")
        summary = generate_summary(text)

        cleanup_temp_files()
        status.update(label="Ready! Ask your questions.", state="complete")

    st.session_state.vectorstore = vectorstore
    st.session_state.text = text
    st.session_state.words = words
    st.session_state.summary = summary
    st.session_state.url = url
    st.session_state.history = []
    st.session_state.video_start = 0
    st.session_state.chunks = vectorstore.index.ntotal
    st.session_state.duration = words[-1]["end"] // 1000
    return True


# ---------- Small helpers ----------
def jump_to(seconds):
    st.session_state.video_start = seconds


def ask(q):
    st.session_state.pending_q = q


def reset_all():
    for k in list(st.session_state.keys()):
        del st.session_state[k]


def stat_card(label, value):
    return (f'<div class="stat"><div class="stat-value">{value}</div>'
            f'<div class="stat-label">{label}</div></div>')


# ---------- Styling ----------
CUSTOM_CSS = """
<style>
.block-container {padding-top: 1.5rem; max-width: 1300px;}
#MainMenu, footer {visibility: hidden;}

.hero {
    background: linear-gradient(135deg, #4f46e5 0%, #9333ea 55%, #ec4899 100%);
    padding: 2rem 2.4rem; border-radius: 20px; color: white;
    margin-bottom: 1.2rem; box-shadow: 0 10px 30px rgba(147, 51, 234, 0.25);
}
.hero h1 {margin: 0; font-size: 2.3rem; color: white; font-weight: 800;}
.hero p {margin: 0.4rem 0 0.9rem; opacity: 0.92; font-size: 1.05rem;}
.pipe {display: flex; flex-wrap: wrap; gap: 0.5rem; align-items: center;}
.pipe span {
    background: rgba(255,255,255,0.18); padding: 0.25rem 0.8rem;
    border-radius: 999px; font-size: 0.85rem; font-weight: 600;
}

.stat, .feature {
    background: rgba(128,128,128,0.10); border: 1px solid rgba(128,128,128,0.25);
    border-radius: 14px; padding: 0.9rem 1rem; text-align: center;
}
.stat-value {font-size: 1.6rem; font-weight: 800; color: #8b5cf6;}
.stat-label {font-size: 0.8rem; opacity: 0.75; text-transform: uppercase; letter-spacing: 0.05em;}
.feature {text-align: left; height: 100%;}
.feature h4 {margin: 0 0 0.4rem;}
.feature p {margin: 0; opacity: 0.8; font-size: 0.92rem;}

div.stButton > button {border-radius: 12px; font-weight: 600;}
[data-testid="stChatMessage"] {border-radius: 14px;}
</style>
"""


def render_sidebar():
    with st.sidebar:
        st.markdown("## 🔊 ChatAudio")
        st.caption("RAG over any YouTube video, running on local LLMs.")
        st.divider()

        st.markdown("**Tech stack**")
        st.markdown(
            "- 🎞️ yt-dlp + ffmpeg\n"
            "- 🗣️ AssemblyAI (speech-to-text)\n"
            "- 🧩 LangChain\n"
            f"- 🧠 Ollama: `{LLM_MODEL}`\n"
            f"- 🔢 Embeddings: `{EMBED_MODEL}`\n"
            "- 📦 FAISS vector store\n"
            "- 🖥️ Streamlit"
        )
        st.divider()

        with st.expander("How does it work?"):
            st.markdown(
                "1. Download the audio\n"
                "2. Transcribe it with word timestamps\n"
                "3. Split the text into chunks\n"
                "4. Turn chunks into embeddings, store in FAISS\n"
                "5. Embed your question, retrieve the closest chunks\n"
                "6. Llama answers using only those chunks\n"
                "7. Chunk positions map back to video time"
            )

        if st.session_state.get("vectorstore") is not None:
            st.button("🗑️ Clear chat", on_click=lambda: st.session_state.update(history=[]),
                      use_container_width=True)
            st.button("🔄 New video", on_click=reset_all, use_container_width=True)


def render_landing():
    st.markdown("#### What you can do")
    a, b, c = st.columns(3)
    a.markdown('<div class="feature"><h4>💬 Ask anything</h4>'
               '<p>Chat with a video and get answers grounded in what was actually said.</p></div>',
               unsafe_allow_html=True)
    b.markdown('<div class="feature"><h4>⏱️ Jump to the moment</h4>'
               '<p>Every answer comes with clickable timestamps that open the video at that point.</p></div>',
               unsafe_allow_html=True)
    c.markdown('<div class="feature"><h4>🔍 See the sources</h4>'
               '<p>Check the exact transcript chunks the AI used. No black box.</p></div>',
               unsafe_allow_html=True)


# ---------- The web page ----------
def main():
    st.set_page_config(layout="wide", page_title="ChatAudio", page_icon="🔊")
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    defaults = {
        "vectorstore": None, "text": None, "words": None, "summary": None,
        "url": None, "history": [], "video_start": 0, "pending_q": None,
        "chunks": 0, "duration": 0,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    render_sidebar()

    # Hero header
    st.markdown("""
    <div class="hero">
        <h1>🔊 Chat with Your Audio</h1>
        <p>Paste a YouTube link and talk to the video. Answers come with timestamps and sources.</p>
        <div class="pipe">
            <span>Download</span> → <span>Transcribe</span> → <span>Embed</span>
            → <span>Retrieve</span> → <span>Answer</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Input row
    c1, c2 = st.columns([5, 1])
    with c1:
        url = st.text_input("YouTube URL", placeholder="https://www.youtube.com/watch?v=...",
                            label_visibility="collapsed")
    with c2:
        go = st.button("Process video", type="primary", use_container_width=True)

    if go:
        if url.strip():
            process_video(url.strip())
        else:
            st.warning("Please paste a link first.")

    if st.session_state.vectorstore is None:
        render_landing()
        return

    # Stat cards
    d = st.session_state.duration
    s1, s2, s3, s4 = st.columns(4)
    s1.markdown(stat_card("Duration", format_time(d)), unsafe_allow_html=True)
    s2.markdown(stat_card("Words", len(st.session_state.words)), unsafe_allow_html=True)
    s3.markdown(stat_card("Chunks indexed", st.session_state.chunks), unsafe_allow_html=True)
    s4.markdown(stat_card("Model", LLM_MODEL), unsafe_allow_html=True)
    st.write("")

    left, right = st.columns([5, 6], gap="large")

    # Left: video + tabs
    with left:
        st.video(st.session_state.url, start_time=st.session_state.video_start)
        tab_sum, tab_txt = st.tabs(["📌 Summary", "📄 Transcript"])
        with tab_sum:
            st.write(st.session_state.summary)
        with tab_txt:
            st.text_area("Transcript", st.session_state.text, height=260,
                         label_visibility="collapsed")
            st.download_button("Download transcript", st.session_state.text,
                               file_name="transcript.txt")

    # Right: chat
    with right:
        st.subheader("💬 Chat")

        if not st.session_state.history:
            st.caption("Try one of these:")
            q1, q2, q3 = st.columns(3)
            q1.button("What is this about?", on_click=ask,
                      args=("What is this audio about?",), use_container_width=True)
            q2.button("Key points", on_click=ask,
                      args=("What are the key points mentioned?",), use_container_width=True)
            q3.button("Who is mentioned?", on_click=ask,
                      args=("Which people are mentioned?",), use_container_width=True)

        chat_box = st.container(height=470, border=True)
        with chat_box:
            if not st.session_state.history:
                st.caption("No messages yet. Ask your first question below 👇")
            for i, msg in enumerate(st.session_state.history):
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])
                    times = msg.get("timestamps")
                    if times:
                        cols = st.columns(min(len(times), 4) + 1)
                        for col, t in zip(cols, times[:4]):
                            col.button(f"▶ {format_time(t)}", key=f"ts_{i}_{t}",
                                       on_click=jump_to, args=(t,))
                    sources = msg.get("sources")
                    if sources:
                        with st.expander("🔍 Sources used"):
                            for n, src in enumerate(sources, 1):
                                st.markdown(f"**Chunk {n}**")
                                st.caption(src)

    # Chat input (pinned to bottom of the page)
    typed = st.chat_input("Ask something about the audio...")
    question = typed or st.session_state.pending_q
    st.session_state.pending_q = None

    if question:
        with st.spinner("Thinking..."):
            answer, docs = answer_question(st.session_state.vectorstore, question)
            times = find_relevant_timestamps(
                docs, st.session_state.text, st.session_state.words
            )
        st.session_state.history.append({"role": "user", "content": question})
        st.session_state.history.append({
            "role": "assistant", "content": answer,
            "timestamps": times, "sources": [d.page_content for d in docs],
        })
        st.rerun()


if __name__ == "__main__":
    main()