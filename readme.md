# Subnautica-Style PDA Voice Assistant

A real-time AI voice assistant that mimics the **PDA voice style from *Subnautica*** using:

* 🧠 Local LLM via `llama-server`
* 🗣️ Piper TTS (`amy.onnx`)
* 🎛️ Custom DSP chain (comb filter, chorus, stereo widening, multiband compression)
* 🔊 Real-time audio playback with PyAudio
* ⚡ Streaming response → sentence chunking → background TTS worker

The result is a synthetic, sci-fi assistant voice inspired by the PDA from
**Subnautica**.

---

## ✨ Features

* Real-time streaming from local LLaMA server
* Sentence-aware chunk extraction
* Markdown cleanup before speech synthesis
* Custom DSP chain to create:

  * Subtle metallic resonance
  * Light chorus modulation
  * Stereo widening
  * Multiband compression polish
* Background threaded TTS processing
* Low-latency playback

---

# 📦 Requirements

## Python Dependencies

Install required packages:

```bash
pip install numpy scipy soundfile pyaudio requests piper-tts
```

You also need:

* `amy.onnx`
* `amy.onnx.json`
* `llama-server.exe`
* A `.gguf` LLaMA model

---

# 📁 Project Structure

```
project/
│
├── amy.onnx
├── amy.onnx.json
├── llama-server.exe
├── Model.gguf
├── main.py
└── README.md
```

---

# 🚀 How It Works

## 1️⃣ LLaMA Server

The script launches:

```
llama-server.exe --model Model.gguf --host 127.0.0.1 --port 8080
```

It streams tokens via HTTP and processes them in real time.

---

## 2️⃣ Streaming + Chunking

Incoming tokens are buffered and split when:

* Sentence punctuation is detected (`. , ! ?`)
* A blank line appears
* A numbered or bullet list starts

This allows speech to begin before the full response is generated.

---

## 3️⃣ DSP Voice Processing Chain

After Piper generates speech:

### 🔹 Comb Filter (Resonance Layer)

Adds metallic synthetic character.

```
delay = sr / 384
resonance = 0.62
```

### 🔹 Chorus Modulation

Adds subtle sci-fi shimmer.

```
depth = 4 ms
rate = 0.19 Hz
```

### 🔹 Stereo Widening

Creates artificial spatial presence.

### 🔹 Multiband Compression

Splits into:

* Low (<200 Hz)
* Mid (200–4000 Hz)
* High (>4000 Hz)

Applies dynamic control per band.

---

# 🧠 Runtime Flow

```
User Prompt
   ↓
LLaMA Streaming
   ↓
Chunk Extraction
   ↓
Markdown Cleaning
   ↓
Piper TTS
   ↓
DSP Processing
   ↓
Background Thread
   ↓
PyAudio Playback
```

---

# ▶️ Running

```bash
python main.py
```

Then type:

```
Ask the AI:
```

The assistant will begin speaking responses in real time.

---

# 🎛️ Customization

## Change Voice Tone

Modify:

```python
resonance = 0.62
depth_ms = 4.0
rate_hz = 0.19
percent = 22
```

Higher resonance = more robotic
Higher chorus depth = more synthetic
Higher stereo percent = wider voice

---

## Reduce Latency

* Lower `n_predict`
* Reduce buffer boundaries
* Reduce DSP complexity
* Use GPU-accelerated LLaMA build

---

# ⚠️ Notes

* Designed for local offline AI usage
* CPU-intensive during streaming
* Threaded TTS prevents blocking
* Automatically terminates LLaMA server on exit




Credits to : https://github.com/sweetbbak/Neural-Amy-TTS
For providing the neural amy model used in this project 
Credits to : https://youtu.be/DFQN5P9HCP8?si=DvkYY425Sn8YItzc
For giving the exact settings used which implemented into an automatic version