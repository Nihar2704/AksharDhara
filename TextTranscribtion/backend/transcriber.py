import whisper
import torch
import time

class Transcriber:
    """Handle Whisper transcription"""
    
    def __init__(self, model_name="tiny.en", device=None):
        """
        Initialize Whisper model
        
        Parameters:
        -----------
        model_name : str
            Whisper model: "tiny.en", "base", "small", "medium", "turbo"
        device : str
            "cuda" or "cpu" (auto-detected if None)
        """
        self.model_name = model_name
        
        # Auto-detect device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        print(f"Loading Whisper model: {model_name} on {self.device}")
        self.model = whisper.load_model(model_name)
        
        if self.device == "cuda":
            self.model = self.model.to("cuda")
        
        print("✓ Model loaded successfully")
    
    def transcribe_chunk(self, audio_chunk, language=None):
        """
        Transcribe a single audio chunk
        
        Parameters:
        -----------
        audio_chunk : np.ndarray
            Audio data as numpy array
        language : str
            Language code (e.g., 'en', 'hi', 'ta') or None for auto-detect
        
        Returns:
        --------
        dict : Transcription result with text and timing
        """
        start_time = time.time()
        
        try:
            # Use FP16 for GPU, FP32 for CPU
            fp16 = (self.device == "cuda")
            
            # Transcribe
            result = self.model.transcribe(
                audio_chunk,
                fp16=fp16,
                language=language
            )
            
            text = result.get("text", "").strip()
            processing_time = time.time() - start_time
            
            return {
                "text": text,
                "language": result.get("language", "unknown"),
                "processing_time": processing_time
            }
            
        except Exception as e:
            print(f"Transcription error: {e}")
            return {
                "text": "",
                "language": "error",
                "processing_time": time.time() - start_time
            }
