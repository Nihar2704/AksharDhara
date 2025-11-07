import os
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")

import json
import asyncio
import numpy as np
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from faster_whisper import WhisperModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import webrtcvad
import logging

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==================== CONFIG ====================
SAMPLE_RATE = 16000
CHUNK_DURATION = 2.0
OVERLAP = 1.0
SILENCE_THRESHOLD = 0.01
MIN_SPEECH_DURATION = 0.5
BLOCK_SIZE = 1600  # 100ms chunks

# ==================== GPU/DEVICE ====================
device = "cuda" if torch.cuda.is_available() else "cpu"
compute_type = "float16" if device == "cuda" else "int8"
logger.info(f"🚀 Using device: {device.upper()}")
logger.info(f"🔢 Compute type: {compute_type}")

# ==================== LOAD MODELS ====================
logger.info("📥 Loading Whisper model (turbo)...")
whisper_model = WhisperModel(
    "turbo",
    device=device,
    compute_type=compute_type,
    num_workers=4,
    cpu_threads=4
)
logger.info("✅ Whisper model loaded!")

logger.info("📥 Loading NLLB translation model...")
nllb_name = "facebook/nllb-200-distilled-600M"
nllb_dtype = torch.float16 if device == "cuda" else torch.float32

nllb_model = AutoModelForSeq2SeqLM.from_pretrained(
    nllb_name,
    dtype=nllb_dtype,
    device_map="auto" if device == "cuda" else None
).to(device)
nllb_tokenizer = AutoTokenizer.from_pretrained(nllb_name)
logger.info("✅ NLLB model loaded!")

# ==================== VAD SETUP ====================
vad = webrtcvad.Vad(1)

# ==================== FASTAPI APP ====================
app = FastAPI(title="Speech Translator API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== UTILITY FUNCTIONS ====================

def translate_text(text: str, src_lang: str, target_lang: str) -> str:
    """Translate text using NLLB model (fully offline)"""
    if not text.strip():
        return ""
    
    try:
        nllb_tokenizer.src_lang = src_lang
        forced_bos_token_id = nllb_tokenizer.convert_tokens_to_ids(target_lang)
        
        inputs = nllb_tokenizer(text, return_tensors="pt", padding=False).to(device)
        
        with torch.inference_mode():
            outputs = nllb_model.generate(
                **inputs,
                forced_bos_token_id=forced_bos_token_id,
                max_new_tokens=256,
                num_beams=3
            )
        
        translation = nllb_tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        return translation
    except Exception as e:
        logger.error(f"Translation error: {e}")
        return ""

def get_rms(audio):
    """Calculate RMS (Root Mean Square) of audio"""
    return np.sqrt(np.mean(audio**2))

def is_speech(audio):
    """Detect if audio contains speech using VAD"""
    if len(audio) < 480:
        return False
    
    rms = get_rms(audio)
    if rms < SILENCE_THRESHOLD:
        return False
    
    pcm16 = np.int16(audio * 32767)
    frame_size = 480  # 30ms @ 16k
    speech_frames = 0
    total_frames = 0
    
    for i in range(0, len(pcm16) - frame_size, frame_size):
        frame = pcm16[i:i + frame_size]
        if len(frame) == frame_size:
            total_frames += 1
            try:
                if vad.is_speech(frame.tobytes(), SAMPLE_RATE):
                    speech_frames += 1
            except:
                pass
    
    return total_frames > 0 and (speech_frames / total_frames) > 0.3

def transcribe_audio(audio_data: np.ndarray, src_lang: str = "en") -> str:
    """Transcribe audio using Whisper"""
    try:
        if len(audio_data) == 0:
            return ""
        
        # Normalize audio
        peak = np.max(np.abs(audio_data)) + 1e-6
        audio_data = audio_data / peak
        
        segments, info = whisper_model.transcribe(
            audio_data,
            beam_size=5,
            language=src_lang,
            vad_filter=True,
            vad_parameters=dict(
                threshold=0.5,
                min_speech_duration_ms=250,
                min_silence_duration_ms=500
            ),
            condition_on_previous_text=True,
            temperature=0.0,
            compression_ratio_threshold=2.4,
            log_prob_threshold=-1.0,
            no_speech_threshold=0.6
        )
        
        text = " ".join([seg.text.strip() for seg in segments])
        return text
    except Exception as e:
        logger.error(f"Transcription error: {e}")
        return ""

# ==================== WEBSOCKET ENDPOINT ====================

class AudioProcessor:
    def __init__(self):
        self.audio_buffer = np.zeros((0,), dtype=np.float32)
    
    def add_chunk(self, chunk: np.ndarray):
        """Add audio chunk to buffer"""
        self.audio_buffer = np.concatenate((self.audio_buffer, chunk))
    
    def get_and_reset(self):
        """Get buffer and reset"""
        data = np.copy(self.audio_buffer)
        overlap_samples = int(SAMPLE_RATE * OVERLAP)
        self.audio_buffer = self.audio_buffer[-overlap_samples:] if len(self.audio_buffer) > overlap_samples else np.zeros((0,), dtype=np.float32)
        return data
    
    def should_process(self):
        """Check if buffer has enough data"""
        return len(self.audio_buffer) >= SAMPLE_RATE * CHUNK_DURATION

@app.websocket("/ws/translate")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    processor = AudioProcessor()
    
    logger.info("🔗 Client connected")
    
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            message = json.loads(data)
            
            msg_type = message.get("type")
            
            if msg_type == "config":
                # Receive language configuration
                src_lang = message.get("srcLang", "eng_Latn")
                target_lang = message.get("targetLang", "tam_Taml")
                src_lang_code = message.get("srcLangCode", "en")
                
                # Extract language code (e.g., "en" from "eng_Latn")
                if src_lang == "eng_Latn":
                    src_lang_code = "en"
                elif src_lang == "fra_Latn":
                    src_lang_code = "fr"
                elif src_lang == "spa_Latn":
                    src_lang_code = "es"
                elif src_lang == "deu_Latn":
                    src_lang_code = "de"
                elif src_lang == "jpn_Jpan":
                    src_lang_code = "ja"
                elif src_lang == "zho_Hans":
                    src_lang_code = "zh"
                elif src_lang == "kor_Hang":
                    src_lang_code = "ko"
                
                processor.src_lang_code = src_lang_code
                processor.src_lang = src_lang
                processor.target_lang = target_lang
                
                logger.info(f"🎯 Config: {src_lang} -> {target_lang}")
                
                await websocket.send_text(json.dumps({
                    "type": "config_ack",
                    "status": "ready"
                }))
            
            elif msg_type == "audio":
                # Receive audio chunk (base64 encoded float32 array)
                audio_chunk_str = message.get("data")
                
                # Decode base64 to numpy array
                import base64
                audio_bytes = base64.b64decode(audio_chunk_str)
                audio_chunk = np.frombuffer(audio_bytes, dtype=np.float32)
                
                # Add to buffer
                processor.add_chunk(audio_chunk)
                
                # Process if enough data accumulated
                if processor.should_process():
                    audio_data = processor.get_and_reset()
                    
                    # Check if there's actual speech
                    if not is_speech(audio_data):
                        await websocket.send_text(json.dumps({
                            "type": "silence",
                            "message": "Silence detected"
                        }))
                        continue
                    
                    # Check minimum duration
                    duration = len(audio_data) / SAMPLE_RATE
                    if duration < MIN_SPEECH_DURATION:
                        await websocket.send_text(json.dumps({
                            "type": "short_speech",
                            "message": f"Speech too short ({duration:.2f}s)"
                        }))
                        continue
                    
                    # Transcribe
                    logger.info("🎤 Transcribing...")
                    transcript = transcribe_audio(audio_data, processor.src_lang_code)
                    
                    if not transcript:
                        await websocket.send_text(json.dumps({
                            "type": "no_speech",
                            "message": "Could not transcribe"
                        }))
                        continue
                    
                    logger.info(f"✅ Transcript: {transcript}")
                    
                    # Translate
                    logger.info("🌐 Translating...")
                    translation = translate_text(
                        transcript,
                        processor.src_lang,
                        processor.target_lang
                    )
                    logger.info(f"✅ Translation: {translation}")
                    
                    # Send result
                    await websocket.send_text(json.dumps({
                        "type": "result",
                        "transcript": transcript,
                        "translation": translation,
                        "duration": round(duration, 2)
                    }))
            
            elif msg_type == "stop":
                logger.info("🛑 Client stopped listening")
                await websocket.send_text(json.dumps({
                    "type": "stopped",
                    "status": "ok"
                }))
    
    except WebSocketDisconnect:
        logger.info("🔌 Client disconnected")
    except Exception as e:
        logger.error(f"❌ WebSocket error: {e}")
        await websocket.send_text(json.dumps({
            "type": "error",
            "message": str(e)
        }))

@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "device": device,
        "compute_type": compute_type,
        "models": ["whisper-turbo", "nllb-200-distilled-600M"]
    }

@app.get("/languages")
async def get_languages():
    return {
        "languages": {
            "eng_Latn": "English",
            "hin_Deva": "Hindi",
            "fra_Latn": "French",
            "spa_Latn": "Spanish",
            "deu_Latn": "German",
            "jpn_Jpan": "Japanese",
            "zho_Hans": "Mandarin",
            "kor_Hang": "Korean",
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")