import yt_dlp
import os
import numpy as np
import soundfile as sf
import librosa
from pathlib import Path

class AudioProcessor:
    """Handle YouTube audio extraction and chunking"""
    
    def __init__(self, output_dir="data/raw_audio"):
        self.output_dir = output_dir
        Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    def download_youtube_audio(self, url, filename="yt_audio"):
        """
        Download audio from YouTube URL
        
        Parameters:
        -----------
        url : str
            YouTube video URL
        filename : str
            Output filename (without extension)
        
        Returns:
        --------
        str : Path to downloaded WAV file
        """
        output_template = os.path.join(self.output_dir, f"{filename}.%(ext)s")
        
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': output_template,
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'wav',
                'preferredquality': '192',
            }],
            'quiet': False,
            'no_warnings': False,
        }
        
        try:
            print(f"Downloading audio from: {url}")
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                
            output_file = os.path.join(self.output_dir, f"{filename}.wav")
            print(f"✓ Audio saved: {output_file}")
            
            return output_file
            
        except Exception as e:
            print(f"✗ Download failed: {e}")
            raise
    
    def chunk_audio(self, audio_path, chunk_duration_s=2):
        """
        Split audio into chunks
        
        Parameters:
        -----------
        audio_path : str
            Path to audio file
        chunk_duration_s : int
            Chunk duration in seconds
        
        Returns:
        --------
        list : List of audio chunks (numpy arrays)
        """
        print(f"Loading audio from: {audio_path}")
        
        # Load audio
        audio, sr = sf.read(audio_path)
        print(f"Original SR: {sr}, shape: {audio.shape}")
        
        # Convert stereo to mono
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        
        # Resample to 16kHz (Whisper standard)
        if sr != 16000:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
            sr = 16000
        
        # Create chunks
        chunk_size = sr * chunk_duration_s
        chunks = []
        total_samples = len(audio)
        
        for start in range(0, total_samples, chunk_size):
            end = start + chunk_size
            chunk = audio[start:end]
            
            # Pad last chunk if needed
            if len(chunk) < chunk_size:
                chunk = np.pad(chunk, (0, chunk_size - len(chunk)), mode='constant')
            
            chunks.append(chunk.astype(np.float32))
        
        print(f"✓ Created {len(chunks)} chunks ({chunk_duration_s}s each)")
        return chunks, sr
