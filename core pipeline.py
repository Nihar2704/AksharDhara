import os
# OPTIONAL: silence Windows symlink warning from huggingface_hub
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")
import sounddevice as sd
import numpy as np
from faster_whisper import WhisperModel
import webrtcvad
import queue, threading, time
import torch
# Translation imports
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# Audio settings
SAMPLE_RATE = 16000
BLOCK_SIZE = 1600  # 100ms chunks
CHUNK_DURATION = 3.0
OVERLAP = 1.0
SILENCE_THRESHOLD = 0.01
MIN_SPEECH_DURATION = 0.5

# VAD settings
vad = webrtcvad.Vad(1)

# GPU Check
device = "cuda" if torch.cuda.is_available() else "cpu"
compute_type = "float16" if device == "cuda" else "int8"
print(f"🚀 Using device: {device.upper()}")
print(f"🔢 Compute type: {compute_type}")

# Load Whisper (speech-to-text)
model = WhisperModel(
    "turbo",
    device=device,
    compute_type=compute_type,
    num_workers=4,
    cpu_threads=4
)

# Translation config
TARGET_LANG = "tam_Taml"  # change as needed (e.g., "fra_Latn", "spa_Latn", "deu_Latn")
SRC_LANG = "eng_Latn"
nllb_name = "facebook/nllb-200-distilled-600M"
nllb_dtype = torch.float16 if device == "cuda" else torch.float32

# Load NLLB translation model
print(f"\n📥 Loading translation model ({nllb_name})...")
nllb_model = AutoModelForSeq2SeqLM.from_pretrained(
    nllb_name,
    dtype=nllb_dtype
).to(device)
nllb_tok = AutoTokenizer.from_pretrained(nllb_name)
nllb_tok.src_lang = SRC_LANG
forced_bos_token_id = nllb_tok.convert_tokens_to_ids(TARGET_LANG)
print("✅ Translation model loaded!\n")

# Offline translation function
def translate_text(text: str) -> str:
    """Translate text using NLLB model (fully offline)"""
    if not text.strip():
        return ""
    
    inputs = nllb_tok(text, return_tensors="pt", padding=False).to(device)
    
    with torch.inference_mode():
        outputs = nllb_model.generate(
            **inputs,
            forced_bos_token_id=forced_bos_token_id,
            max_new_tokens=256,
            num_beams=3
        )
    
    return nllb_tok.batch_decode(outputs, skip_special_tokens=True)[0]

# Audio processing
audio_queue = queue.Queue()
audio_buffer = np.zeros((0,), dtype=np.float32)
running = True
last_speech_time = time.time()

def audio_callback(indata, frames, time_info, status):
    if status:
        print(f"⚠️ {status}")
    audio_queue.put(indata.copy())

def get_rms(audio):
    return np.sqrt(np.mean(audio**2))

def is_speech(audio):
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

def transcribe_audio():
    global audio_buffer, last_speech_time
    print("🎧 Listening... (Ctrl+C to stop)\n")
    
    while running:
        if not audio_queue.empty():
            chunk = audio_queue.get()
            audio_buffer = np.concatenate((audio_buffer, chunk[:, 0]))
            
            if len(audio_buffer) >= SAMPLE_RATE * CHUNK_DURATION:
                temp_audio = np.copy(audio_buffer)
                overlap_samples = int(SAMPLE_RATE * OVERLAP)
                audio_buffer = audio_buffer[-overlap_samples:]
                
                if not is_speech(temp_audio):
                    continue
                
                if (len(temp_audio) / SAMPLE_RATE) < MIN_SPEECH_DURATION:
                    continue
                
                peak = np.max(np.abs(temp_audio)) + 1e-6
                temp_audio = temp_audio / peak
                
                try:
                    segments, info = model.transcribe(
                        temp_audio,
                        beam_size=5,
                        language="en",
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
                    
                    for seg in segments:
                        text = seg.text.strip()
                        if not text:
                            continue
                        
                        print(f"🗣️  You said: {text}")
                        
                        # Offline translation
                        try:
                            translated = translate_text(text)
                            print(f"🌐 [{TARGET_LANG}] {translated}\n")
                        except Exception as te:
                            print(f"❌ Translation error: {te}\n")
                        
                        last_speech_time = time.time()
                        
                except Exception as e:
                    print(f"❌ Transcription error: {e}")
        else:
            time.sleep(0.01)

# Start audio stream
stream = sd.InputStream(
    samplerate=SAMPLE_RATE,
    channels=1,
    blocksize=BLOCK_SIZE,
    dtype='float32',
    callback=audio_callback
)

thread = threading.Thread(target=transcribe_audio, daemon=True)
thread.start()

with stream:
    try:
        print("✅ Ready! Start speaking...\n")
        print(f"📝 Source Language: {SRC_LANG}")
        print(f"🎯 Target Language: {TARGET_LANG}\n")
        
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n🛑 Stopping...")
        running = False
        thread.join(timeout=2)
        print("👋 Goodbye!")


"hin_Deva"      # Hindi
"tam_Taml"      # Tamil
"tel_Telg"      # Telugu
"kan_Knda"      # Kannada
"mal_Mlym"      # Malayalam
"mar_Deva"      # Marathi
"guj_Gujr"      # Gujarati
"pan_Guru"      # Punjabi
"ory_Orya"      # Odia
"asm_Beng"      # Assamese