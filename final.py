
import os
import io
import time
import re
import wave
import queue
import threading
import subprocess
import requests
import json
import numpy as np
import scipy.signal as signal
import soundfile as sf
import pyaudio
from piper import PiperVoice

# -----------------------------
# Environment
# -----------------------------
os.environ["OMP_NUM_THREADS"] = str(os.cpu_count())
os.environ["OPENBLAS_NUM_THREADS"] = str(os.cpu_count())
os.environ["MKL_NUM_THREADS"] = str(os.cpu_count())
os.environ["NUMEXPR_NUM_THREADS"] = str(os.cpu_count())
os.environ["OMP_WAIT_POLICY"] = "ACTIVE"

# -----------------------------
# Paths
# -----------------------------
MODEL_PATH = "amy.onnx"
CONFIG_PATH = "amy.onnx.json"

LLAMA_SERVER_EXE = r"./AI Model/llama-server.exe"
LLAMA_MODEL_PATH = r"./AI Model/Model.gguf"
LLAMA_PORT = 8080
LLAMA_URL = f"http://127.0.0.1:{LLAMA_PORT}/completion"

# -----------------------------
# Load TTS (Piper)
# -----------------------------
voice = PiperVoice.load(MODEL_PATH, config_path=CONFIG_PATH)

# -----------------------------
# PyAudio setup
# -----------------------------
p = pyaudio.PyAudio()

def play_audio(audio, sr):
    stream = p.open(
        format=pyaudio.paFloat32,
        channels=1,
        rate=sr,
        output=True
    )
    stream.write(audio.astype(np.float32).tobytes())
    stream.stop_stream()
    stream.close()

# -----------------------------
# Markdown cleaning
# -----------------------------
def clean_markdown(text: str) -> str:
    # Remove bold/italic markers: **text**, *text*, __text__, _text_
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
    text = re.sub(r'\*(.*?)\*', r'\1', text)
    text = re.sub(r'__(.*?)__', r'\1', text)
    text = re.sub(r'_(.*?)_', r'\1', text)

    # Remove inline code/backticks
    text = re.sub(r'`([^`]*)`', r'\1', text)

    # Remove strikethrough ~~text~~
    text = re.sub(r'~~(.*?)~~', r'\1', text)

    # Strip leading markdown bullets and headings
    text = re.sub(r'^[>\s]*[#*+\-]\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'^[>\s]*\d+[\).\-\:]\s+', '', text, flags=re.MULTILINE)

    return text

# -----------------------------
# TTS + DSP (your chain)
# -----------------------------
def synthesize_and_process_chunk(text_chunk):
    wav_buffer = io.BytesIO()
    with wave.open(wav_buffer, "wb") as wav_file:
        voice.synthesize_wav(text_chunk, wav_file)
    wav_buffer.seek(0)

    y, sr = sf.read(wav_buffer)
    if y.ndim > 1:
        y = y.mean(axis=1)
    y = y / max(1e-6, np.max(np.abs(y)))

    delay = int(sr / 384)
    resonance = 0.62
    b = [1.0]
    a = np.zeros(delay + 1)
    a[0] = 1.0
    a[-1] = -resonance
    y_comb = signal.lfilter(b, a, y)

    depth_ms = 4.0
    rate_hz = 0.19
    depth_samples = int(sr * depth_ms / 1000)
    lfo = (np.sin(2 * np.pi * rate_hz * np.arange(len(y)) / sr) + 1) / 2
    delays = (lfo * depth_samples).astype(int)
    idx = np.arange(len(y)) - delays
    idx[idx < 0] = 0
    delayed = y_comb[idx]
    y_chorus = (y_comb + delayed) * 0.5

    percent = 22
    left = y_chorus * (1 - percent / 100)
    right = y_chorus * (1 + percent / 100)
    y_stereo = np.stack([left, right], axis=1)

    low = signal.butter(4, 200, 'low', fs=sr, output='sos')
    mid = signal.butter(4, [200, 4000], 'bandpass', fs=sr, output='sos')
    high = signal.butter(4, 4000, 'high', fs=sr, output='sos')

    y_mono = y_stereo.mean(axis=1)
    low_band = signal.sosfilt(low, y_mono)
    mid_band = signal.sosfilt(mid, y_mono)
    high_band = signal.sosfilt(high, y_mono)

    def compress(band, threshold=0.2, ratio=4.0):
        gain = np.ones_like(band)
        over = np.abs(band) > threshold
        if np.any(over):
            gain[over] = threshold + (np.abs(band[over]) - threshold) / ratio
            gain[over] /= np.abs(band[over])
        return band * gain

    y_final = compress(low_band) + compress(mid_band) + compress(high_band)

    pad = int(sr * 0.2)
    y_final = np.concatenate([y_final, np.zeros(pad, dtype=y_final.dtype)])

    return y_final, sr

# -----------------------------
# LLaMA server
# -----------------------------
def start_server():
    return subprocess.Popen(
        [LLAMA_SERVER_EXE, "--model", LLAMA_MODEL_PATH, "--host", "127.0.0.1", "--port", str(LLAMA_PORT)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

def wait_ready(timeout=30):
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            r = requests.post(LLAMA_URL, json={"prompt": "test", "n_predict": 1}, timeout=5)
            if r.status_code == 200:
                return True
        except:
            pass
        time.sleep(1)
    return False

def stream_llama(prompt):
    payload = {
        "prompt": prompt,
        "n_predict": 1024,
        "temperature": 0.7,
        "stream": True,
    }

    with requests.post(LLAMA_URL, json=payload, stream=True, timeout=300) as resp:
        resp.raise_for_status()
        for raw in resp.iter_lines():
            if not raw:
                continue
            line = raw.decode("utf-8", errors="ignore").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                obj = json.loads(data)
            except:
                continue
            if "content" in obj and obj["content"]:
                yield obj["content"]

# -----------------------------
# Chunk extraction
# -----------------------------
def extract_chunks(buffer: str):
    chunks = []
    while True:
        # Punctuation boundary: .,!? followed by space/newline
        m_punct = re.search(r'([.,!?])(\s+)', buffer)

        # Semantic newline boundary:
        # 1) blank line (double newline)
        m_blank = re.search(r'\n\s*\n+', buffer)
        # 2) newline followed by list/numbered item
        m_list = re.search(r'\n\s*(\d+[\).\-\:]\s+|[-*]\s+)', buffer)

        candidates = []
        for m in [m_punct, m_blank, m_list]:
            if m:
                candidates.append(m.end())

        if not candidates:
            break

        cut = min(candidates)
        chunk = buffer[:cut].strip()
        if chunk:
            chunks.append(chunk)
        buffer = buffer[cut:]
    return chunks, buffer

# -----------------------------
# Background TTS worker
# -----------------------------
tts_queue = queue.Queue()
audio_queue = queue.Queue()

def tts_worker():
    while True:
        text = tts_queue.get()
        if text is None:
            break
        cleaned = clean_markdown(text)
        audio, sr = synthesize_and_process_chunk(cleaned)
        audio_queue.put((audio, sr, cleaned))
        tts_queue.task_done()

worker = threading.Thread(target=tts_worker, daemon=True)
worker.start()

# -----------------------------
# Main
# -----------------------------
def main():
    server = start_server()
    if not wait_ready():
        print("LLaMA server failed to start")
        return

    prompt = input("Ask the AI: ").strip()
    print("\nAI thinking...\n")

    buffer = ""
    enqueued = 0
    played = 0
    first = True

    for token in stream_llama(prompt):
        buffer += token

        new_chunks, buffer = extract_chunks(buffer)
        for chunk in new_chunks:
            tts_queue.put(chunk)
            enqueued += 1

            if first:
                first = False
                continue

            audio, sr, text = audio_queue.get()
            played += 1
            print(text, flush=True)
            play_audio(audio, sr)

    if buffer.strip():
        tts_queue.put(buffer.strip())
        enqueued += 1

    tts_queue.put(None)
    tts_queue.join()

    while played < enqueued:
        audio, sr, text = audio_queue.get()
        played += 1
        print(text, flush=True)
        play_audio(audio, sr)

    if server.poll() is None:
        server.terminate()

if __name__ == "__main__":
    main()
