# 🌊 AksharDhara (अक्षरधारा)

**AksharDhara** (meaning *Flow of Letters*) is an AI-powered, real-time multilingual and "Deaf-friendly" captioning system. It bridges the digital accessibility gap for deaf and hard-of-hearing individuals by providing low-latency, accurate captions for both live speech and digital content (YouTube) in multiple Indian regional languages.

Developed during **CodeUtsava 9.0** for **Problem Statement 4 (Accessibility)**.

---

## 🚀 Key Features

- **🎙️ Real-Time Live Translator:** Transcribes and translates live microphone input with minimal latency using WebSocket streaming.
- **📺 YouTube Sync-Captioner:** A Chrome extension that injects live multilingual captions into any YouTube video, perfectly synchronized with playback.
- **🇮🇳 Multilingual Indian Support:** High-accuracy translation for major Indian languages including Hindi, Tamil, Telugu, Kannada, Marathi, Odia, and more.
- **🔒 Privacy-First & Offline:** Powered by local AI models (`Whisper` & `NLLB-200`), ensuring all processing happens on-device with no external API dependency or data leakage.
- **⚖️ "Deaf-Friendly" UI:** High-contrast, large-font overlays designed specifically for readability and accessibility.
- **⚡ Intelligent Playback Control:** Automatically pauses and resumes YouTube videos to ensure captions never lag behind the audio.

---

## 🛠 Tech Stack

### AI Models
- **Speech-to-Text (ASR):** [OpenAI Whisper](https://github.com/openai/whisper) (Small/Turbo) via `faster-whisper`.
- **Machine Translation:** [Meta NLLB-200](https://github.com/facebookresearch/fairseq2/tree/main/nllb) (No Language Left Behind) - 600M Distilled.

### Backend
- **Framework:** `FastAPI` (Python)
- **Real-time:** `WebSockets`
- **Audio Processing:** `librosa`, `soundfile`, `webrtcvad` (Voice Activity Detection)
- **Video Tools:** `yt-dlp` (YouTube extraction)

### Frontend
- **Web App:** `React.js` (Vite), `Tailwind CSS`, `Lucide React`
- **Extension:** `Chrome Extension API` (Manifest V3)

---

## 🏗 System Architecture

### 1. The Microphone Pipeline (Live Events)
1. **Capture:** Browser captures audio at 16kHz.
2. **Stream:** 100ms audio chunks are sent via WebSockets.
3. **VAD:** Backend filters out silence to save compute.
4. **ASR:** Whisper transcribes the speech.
5. **Translation:** NLLB-200 translates into the user's chosen Indian language.
6. **Result:** Displayed instantly on the React Dashboard.

### 2. The YouTube Pipeline (Digital Content)
1. **Trigger:** Extension grabs the current Video URL.
2. **Fetch:** Backend extracts audio using `yt-dlp`.
3. **Sync:** Extension **pauses** the video while the first few chunks are processed.
4. **Playback:** Video **resumes** automatically when captions are ready to stream, ensuring 1:1 synchronization.

---

## 📦 Installation & Setup

### Prerequisites
- Python 3.9+
- Node.js & npm
- NVIDIA GPU (Optional but recommended for `CUDA` acceleration)
- FFmpeg (Required for audio processing)

### 1. Backend Setup
```bash
# Clone the repository
git clone https://github.com/your-username/AksharDhara.git
cd AksharDhara

# Install dependencies
pip install -r TextTranscribtion/backend/requirements.txt
# (Additional dependencies if needed)
pip install fastapi uvicorn faster-whisper transformers torch sounddevice
```

### 2. Frontend Setup
```bash
cd "Live Translator/frontend"
npm install
npm run dev
```

### 3. Chrome Extension Setup
1. Open Chrome and go to `chrome://extensions/`.
2. Enable **Developer mode** (top right).
3. Click **Load unpacked**.
4. Select the `TextTranscribtion/chrome_extension` folder.

## ⚡ Quick Start / CLI Testing

If you want to quickly test the transcription and translation pipeline without setting up the full web interface or Chrome extension, you can use the standalone CLI tool:

1. **Ensure dependencies are installed:**
   ```bash
   pip install faster-whisper transformers torch sounddevice webrtcvad numpy
   ```

2. **Run the core pipeline:**
   ```bash
   python "core pipeline.py"
   ```

- **How it works:** This script listens directly to your microphone, transcribes speech using Whisper, and translates it into a target Indian language (default: Tamil) using NLLB-200, all within your terminal.
- **Customization:** You can easily change the `TARGET_LANG` variable at the bottom of the script to test other languages like Hindi (`hin_Deva`), Telugu (`tel_Telg`), or Marathi (`mar_Deva`).

---

## 🖥 Usage

1. **Live Translation:**
   - Start the backend server (`python server.py`).
   - Open the React App.
   - Click **"Start Listening"** and select your target language.

2. **YouTube Captions:**
   - Start the transcription server.
   - Open a YouTube video.
   - Click the **AksharDhara Extension** icon and hit **"Start Transcription"**.
   - Watch as the video pauses briefly to sync and then displays high-quality regional captions.

---

## 🎯 Problem Statement 4: Accessibility
This project directly addresses the challenges of:
- **Equitable Access:** Making live streams and education inclusive for the deaf community.
- **Language Barriers:** Supporting diverse Indian accents and dialects.
- **Cognitive Load:** Providing "Deaf-friendly" simplified visual structures.

---



## 🤝 Contributors
Developed with ❤️ by **Team SemiColons** during **CodeUtsava 9.0**.

---
