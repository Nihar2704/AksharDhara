import asyncio
import websockets
import json
import time
import yt_dlp
import numpy as np
import soundfile as sf
import librosa
import whisper
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Configuration
PORT = 8765
WHISPER_MODEL = "tiny.en"  # Change to "base", "small", or "turbo" for better accuracy
TRANSLATION_MODEL = "facebook/nllb-200-distilled-600M"
TARGET_LANGUAGE = "hin_Deva"  # Hindi, change as needed

# Global state
whisper_model = None
translation_model = None
translation_tokenizer = None
clients = set()
chunks = []
models_loaded = False

def load_models():
    """Load all models at startup"""
    global whisper_model, translation_model, translation_tokenizer, models_loaded
    
    if models_loaded:
        return
    
    print("\n" + "="*60)
    print("  LOADING MODELS...")
    print("="*60)
    
    # Load Whisper model
    print(f"\n📝 Loading Whisper model: {WHISPER_MODEL}")
    start = time.time()
    whisper_model = whisper.load_model(WHISPER_MODEL)
    if torch.cuda.is_available():
        whisper_model = whisper_model.to("cuda")
        print(f"   ✓ Loaded on GPU in {time.time() - start:.2f}s")
    else:
        print(f"   ✓ Loaded on CPU in {time.time() - start:.2f}s")
    
    # Load Translation model
    print(f"\n🌐 Loading Translation model: {TRANSLATION_MODEL}")
    start = time.time()
    translation_tokenizer = AutoTokenizer.from_pretrained(TRANSLATION_MODEL)
    translation_model = AutoModelForSeq2SeqLM.from_pretrained(TRANSLATION_MODEL)
    
    if torch.cuda.is_available():
        translation_model = translation_model.to("cuda")
        print(f"   ✓ Loaded on GPU in {time.time() - start:.2f}s")
    else:
        print(f"   ✓ Loaded on CPU in {time.time() - start:.2f}s")
    
    models_loaded = True
    print("\n" + "="*60)
    print("  ✓ ALL MODELS LOADED AND READY!")
    print("="*60 + "\n")

def download_and_chunk_audio(url, chunk_duration=2):
    """Download YouTube audio and chunk it"""
    print(f"Downloading: {url}")
    
    # Create output dir
    Path("data/raw_audio").mkdir(parents=True, exist_ok=True)
    
    # Download audio
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': 'data/raw_audio/yt_audio.%(ext)s',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192',
        }],
        'quiet': True
    }
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    
    audio_file = "data/raw_audio/yt_audio.wav"
    print(f"✓ Downloaded: {audio_file}")
    
    # Load and chunk audio
    audio, sr = sf.read(audio_file)
    
    # Convert stereo to mono
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    
    # Resample to 16kHz
    if sr != 16000:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        sr = 16000
    
    # Create chunks
    chunk_size = sr * chunk_duration
    audio_chunks = []
    
    for start in range(0, len(audio), chunk_size):
        end = start + chunk_size
        chunk = audio[start:end]
        
        if len(chunk) < chunk_size:
            chunk = np.pad(chunk, (0, chunk_size - len(chunk)), mode='constant')
        
        audio_chunks.append(chunk.astype(np.float32))
    
    print(f"✓ Created {len(audio_chunks)} chunks")
    return audio_chunks

def translate_text(text, target_lang=TARGET_LANGUAGE):
    """Translate text using NLLB model"""
    if not text or text.strip() == "":
        return ""
    
    try:
        # Set source language to English
        translation_tokenizer.src_lang = "eng_Latn"
        
        # Tokenize
        inputs = translation_tokenizer(text, return_tensors="pt", padding=True, max_length=512, truncation=True)
        
        if torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
        
        # Get the target language token ID
        target_lang_id = translation_tokenizer.convert_tokens_to_ids(target_lang)
        
        # Generate translation
        translated_tokens = translation_model.generate(
            **inputs,
            forced_bos_token_id=target_lang_id,
            max_length=512
        )
        
        # Decode
        translated_text = translation_tokenizer.batch_decode(
            translated_tokens, skip_special_tokens=True
        )[0]
        
        return translated_text
    
    except Exception as e:
        print(f"Translation error: {e}")
        return f"[Translation Error: {e}]"

async def register(ws):
    """Register new client"""
    clients.add(ws)
    print(f"✓ Client connected: {ws.remote_address}")
    await ws.send(json.dumps({
        "event": "connected",
        "whisper_model": WHISPER_MODEL,
        "translation_model": TRANSLATION_MODEL,
        "target_language": TARGET_LANGUAGE,
        "models_ready": models_loaded
    }))

async def unregister(ws):
    """Unregister client"""
    clients.remove(ws)
    print(f"✗ Client disconnected")

async def broadcast(message):
    """Broadcast message to all clients"""
    if clients:
        await asyncio.gather(
            *[client.send(json.dumps(message)) for client in clients],
            return_exceptions=True
        )

async def process_video(url):
    """Download and prepare video for transcription"""
    global chunks
    
    try:
        await broadcast({"event": "status", "message": "Downloading audio..."})
        chunks = download_and_chunk_audio(url)
        
        await broadcast({
            "event": "ready",
            "message": f"Ready! {len(chunks)} chunks",
            "total_chunks": len(chunks)
        })
        
        return True
    except Exception as e:
        print(f"Error processing video: {e}")
        await broadcast({"event": "error", "message": str(e)})
        return False

async def transcribe_and_translate_chunks():
    """Transcribe and translate all audio chunks"""
    
    # Check if models are loaded
    if not models_loaded:
        await broadcast({"event": "error", "message": "Models not loaded yet!"})
        return
    
    await broadcast({"event": "transcription_start"})
    
    for i, chunk in enumerate(chunks):
        chunk_start_time = time.time()
        
        # Transcribe
        try:
            fp16 = torch.cuda.is_available()
            result = whisper_model.transcribe(chunk, fp16=fp16)
            text = result.get("text", "").strip()
        except Exception as e:
            text = f"[Error: {e}]"
        
        transcribe_time = time.time() - chunk_start_time
        
        # Translate
        translate_start = time.time()
        translated_text = translate_text(text)
        translate_time = time.time() - translate_start
        
        total_time = time.time() - chunk_start_time
        
        message = {
            "event": "caption",
            "chunk_index": i,
            "total_chunks": len(chunks),
            "text": text,
            "translated_text": translated_text,
            "start_time_s": i * 2,
            "end_time_s": (i + 1) * 2,
            "transcribe_time_s": round(transcribe_time, 3),
            "translate_time_s": round(translate_time, 3),
            "processing_time_s": round(total_time, 3),
            "progress": round((i + 1) / len(chunks) * 100, 1)
        }
        
        print(f"  Sending caption {i+1}/{len(chunks)} to clients...")
        
        print(f"\n[{i+1}/{len(chunks)}] Video time: {i*2}s - {(i+1)*2}s")
        print(f"  EN: {text}")
        print(f"  TR: {translated_text}")
        print(f"  Processing: {total_time:.2f}s (transcribe: {transcribe_time:.2f}s, translate: {translate_time:.2f}s)")
        
        # Broadcast caption immediately
        await broadcast(message)
        
        # Small delay to allow client to process
        await asyncio.sleep(0.05)
    
    await broadcast({"event": "complete", "message": "Done!"})
    print("\n✓ Transcription and translation complete")

async def handle_message(ws, message):
    """Handle incoming WebSocket messages"""
    try:
        data = json.loads(message)
        action = data.get("action")
        
        if action == "set_url":
            url = data.get("url")
            await process_video(url)
        
        elif action == "start_transcription":
            await transcribe_and_translate_chunks()
        
        elif action == "stop":
            await broadcast({"event": "stopped", "message": "Stopped"})
        
        elif action == "ping":
            await ws.send(json.dumps({"event": "pong"}))
        
    except Exception as e:
        print(f"Error handling message: {e}")

async def handler(websocket):
    """Main WebSocket connection handler"""
    await register(websocket)
    try:
        async for message in websocket:
            await handle_message(websocket, message)
    except websockets.exceptions.ConnectionClosed:
        print("Connection closed")
    finally:
        await unregister(websocket)

async def main():
    """Start WebSocket server"""
    print("="*60)
    print("  REAL-TIME TRANSCRIPTION & TRANSLATION SERVER")
    print("="*60)
    print(f"  Whisper Model: {WHISPER_MODEL}")
    print(f"  Translation Model: {TRANSLATION_MODEL}")
    print(f"  Target Language: {TARGET_LANGUAGE}")
    print(f"  Port: {PORT}")
    print(f"  URL: ws://localhost:{PORT}")
    print("="*60 + "\n")
    
    # Load all models at startup
    load_models()
    
    async with websockets.serve(handler, "0.0.0.0", PORT):
        print("✓ Server running and ready to accept connections!")
        print("  Press Ctrl+C to stop.\n")
        await asyncio.Future()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n✓ Server stopped")