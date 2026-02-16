
import wave
import io
import os
import time
import numpy as np
import scipy.signal as signal
import soundfile as sf
from piper import PiperVoice
import re

os.environ["OMP_NUM_THREADS"] = str(os.cpu_count())
os.environ["OPENBLAS_NUM_THREADS"] = str(os.cpu_count())
os.environ["MKL_NUM_THREADS"] = str(os.cpu_count())
os.environ["NUMEXPR_NUM_THREADS"] = str(os.cpu_count())
os.environ["OMP_WAIT_POLICY"] = "ACTIVE"

MODEL_PATH = "amy.onnx"
CONFIG_PATH = "amy.onnx.json"
OUTPUT_WAV = "processed_amy_output.wav"

voice = PiperVoice.load(MODEL_PATH, config_path=CONFIG_PATH)

def split_into_sentences(text):
    # Simple sentence splitter
    sentences = re.split(r'(?<=[.!?]) +', text)
    return [s.strip() for s in sentences if s.strip()]

def synthesize_and_process_chunk(text_chunk):
    tts_start = time.time()
    wav_buffer = io.BytesIO()
    with wave.open(wav_buffer, "wb") as wav_file:
        voice.synthesize_wav(text_chunk, wav_file)
    wav_buffer.seek(0)
    tts_time = time.time() - tts_start

    dsp_start = time.time()
    y, sr = sf.read(wav_buffer)
    if y.ndim > 1:
        y = y.mean(axis=1)
    y = y / np.max(np.abs(y))

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
        gain[over] = threshold + (np.abs(band[over]) - threshold)/ratio
        gain[over] /= np.abs(band[over])
        return band * gain

    low_res, mid_res, high_res = compress(low_band), compress(mid_band), compress(high_band)
    y_final = low_res + mid_res + high_res
    dsp_time = time.time() - dsp_start

    return y_final, sr, tts_time, dsp_time

texts = []
while True:
    line = input("Enter text (or 'quit' to finish): ")
    if line.strip().lower() == "quit":
        break
    texts.append(line)

final_audio_chunks = []
total_tts = 0
total_dsp = 0

for text in texts:
    chunks = split_into_sentences(text)
    for i, chunk in enumerate(chunks):
        y_final, sr, tts_time, dsp_time = synthesize_and_process_chunk(chunk)
        final_audio_chunks.append(y_final)
        total_tts += tts_time
        total_dsp += dsp_time
        print(f"Chunk {i+1}/{len(chunks)} | TTS: {tts_time:.2f}s | DSP: {dsp_time:.2f}s")

final_audio = np.concatenate(final_audio_chunks)
sf.write(OUTPUT_WAV, final_audio, sr)
print(f"Processed audio saved to {OUTPUT_WAV}")
print(f"Total TTS time: {total_tts:.2f}s | Total DSP time: {total_dsp:.2f}s")
