// Create caption overlay
let captionBox = null;
let videoElement = null;
let isTranscribing = false;
let videoPlaybackStarted = false;
let videoCheckInterval = null;

function createCaptionBox() {
  if (captionBox) return;
  
  captionBox = document.createElement('div');
  captionBox.id = 'live-caption-box';
  captionBox.style.cssText = `
    position: fixed;
    bottom: 100px;
    left: 50%;
    transform: translateX(-50%);
    background: rgba(0, 0, 0, 0.9);
    color: white;
    padding: 20px 30px;
    border-radius: 8px;
    font-size: 20px;
    font-family: Arial, sans-serif;
    max-width: 85%;
    text-align: center;
    z-index: 999999;
    box-shadow: 0 4px 12px rgba(0,0,0,0.5);
  `;
  
  document.body.appendChild(captionBox);
}

function findYouTubeVideo() {
  // Try multiple methods to find the video
  let video = document.querySelector('video.html5-main-video');
  
  if (!video) {
    video = document.querySelector('.video-stream.html5-main-video');
  }
  
  if (!video) {
    const videos = document.querySelectorAll('video');
    if (videos.length > 0) {
      video = videos[0];
    }
  }
  
  if (video) {
    console.log('✓ YouTube video found:', video);
  } else {
    console.error('✗ YouTube video NOT found');
  }
  
  return video;
}

function updateCaption(text, translatedText = null) {
  if (!captionBox) createCaptionBox();
  
  if (translatedText) {
    captionBox.innerHTML = `
      <div style="margin-bottom: 10px; font-size: 18px; font-weight: bold;">${text}</div>
      <div style="color: #64B5F6; font-style: italic; font-size: 16px;">${translatedText}</div>
    `;
  } else {
    captionBox.textContent = text;
  }
  
  captionBox.style.display = text ? 'block' : 'none';
}

function forceVideoToStart() {
  videoElement = findYouTubeVideo();
  
  if (!videoElement) {
    console.error('Cannot start video - element not found');
    setTimeout(forceVideoToStart, 500); // Retry
    return;
  }
  
  console.log('🎬 FORCING video to start from 0s');
  
  // Remove any event listeners that might pause the video
  videoElement.removeEventListener('pause', null);
  
  // Seek to start
  videoElement.currentTime = 0;
  
  // Unmute if muted
  videoElement.muted = false;
  
  // Set playback rate to normal
  videoElement.playbackRate = 1.0;
  
  // Force play with multiple attempts
  const playAttempt = () => {
    const playPromise = videoElement.play();
    
    if (playPromise !== undefined) {
      playPromise.then(() => {
        console.log('✅ Video is NOW PLAYING from 0s');
        videoPlaybackStarted = true;
        
        // Monitor video to ensure it keeps playing
        startVideoMonitoring();
        
      }).catch(error => {
        console.warn('Play attempt failed:', error.message);
        console.log('Retrying in 200ms...');
        setTimeout(playAttempt, 200);
      });
    }
  };
  
  playAttempt();
}

function startVideoMonitoring() {
  // Clear any existing interval
  if (videoCheckInterval) {
    clearInterval(videoCheckInterval);
  }
  
  // Check every 500ms if video is still playing
  videoCheckInterval = setInterval(() => {
    if (!videoElement || !videoPlaybackStarted) {
      clearInterval(videoCheckInterval);
      return;
    }
    
    // If video got paused somehow, restart it
    if (videoElement.paused && isTranscribing) {
      console.log('⚠️ Video paused unexpectedly! Restarting...');
      videoElement.play().catch(err => console.error('Restart failed:', err));
    }
    
    // Log current time for debugging
    if (videoElement) {
      console.log(`Video time: ${videoElement.currentTime.toFixed(2)}s`);
    }
    
  }, 500);
}

function pauseAndResetVideo() {
  videoElement = findYouTubeVideo();
  
  if (videoElement) {
    videoElement.pause();
    videoElement.currentTime = 0;
    console.log('✓ Video PAUSED and reset to 0s');
  }
  
  videoPlaybackStarted = false;
  
  if (videoCheckInterval) {
    clearInterval(videoCheckInterval);
    videoCheckInterval = null;
  }
}

function stopVideoPlayback() {
  videoElement = findYouTubeVideo();
  
  if (videoElement) {
    videoElement.pause();
    console.log('✓ Video STOPPED');
  }
  
  isTranscribing = false;
  videoPlaybackStarted = false;
  
  if (videoCheckInterval) {
    clearInterval(videoCheckInterval);
    videoCheckInterval = null;
  }
}

// Listen for messages from background script
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.type === 'caption') {
    const data = message.data;
    
    if (data.event === 'status') {
      updateCaption(data.message);
      console.log('Status:', data.message);
      
      // Pause and reset when download starts
      if (data.message.includes('Downloading')) {
        pauseAndResetVideo();
      }
    }
    else if (data.event === 'ready') {
      updateCaption('Ready! Transcription starting...');
      console.log('Ready! Total chunks:', data.total_chunks);
      
      // Ensure video is paused and at start
      pauseAndResetVideo();
    }
    else if (data.event === 'transcription_start') {
      isTranscribing = true;
      videoPlaybackStarted = false;
      updateCaption('Processing first chunk...');
      console.log('🎙️ Transcription started');
      
      // Make absolutely sure video is paused at start
      pauseAndResetVideo();
    }
    else if (data.event === 'caption') {
      const chunkIndex = data.chunk_index;
      const startTime = data.start_time_s;
      const endTime = data.end_time_s || (startTime + 2);
      
      console.log('\n' + '='.repeat(50));
      console.log(`📝 Caption ${chunkIndex + 1}/${data.total_chunks}`);
      console.log(`   Time: ${startTime}s - ${endTime}s`);
      console.log(`   EN: ${data.text}`);
      console.log(`   TR: ${data.translated_text}`);
      console.log('='.repeat(50));
      
      // Update caption display
      updateCaption(data.text, data.translated_text);
      
      // START VIDEO when second chunk (index 1) arrives
      if (chunkIndex === 1 && !videoPlaybackStarted) {
        console.log('\n' + '🚀'.repeat(20));
        console.log('🎬 SECOND CHUNK ARRIVED!');
        console.log('🎬 STARTING VIDEO PLAYBACK NOW!');
        console.log('🚀'.repeat(20) + '\n');
        
        // Wait a tiny bit then force start
        setTimeout(() => {
          forceVideoToStart();
        }, 100);
      }
    }
    else if (data.event === 'complete') {
      updateCaption('✅ Transcription complete!');
      console.log('✅ Transcription complete!');
      isTranscribing = false;
      
      // Stop monitoring but let video continue
      if (videoCheckInterval) {
        clearInterval(videoCheckInterval);
      }
      
      setTimeout(() => {
        updateCaption('');
      }, 3000);
    }
    else if (data.event === 'stopped') {
      stopVideoPlayback();
      updateCaption('⏹️ Stopped');
      console.log('⏹️ Stopped');
      setTimeout(() => updateCaption(''), 2000);
    }
    else if (data.event === 'error') {
      updateCaption('❌ Error: ' + data.message);
      console.error('❌ Error:', data.message);
      stopVideoPlayback();
      setTimeout(() => updateCaption(''), 5000);
    }
  }
});

// Find video on page load
window.addEventListener('load', () => {
  setTimeout(() => {
    videoElement = findYouTubeVideo();
    if (videoElement) {
      console.log('✓ Video element initialized');
    }
  }, 1000);
});

// Keep checking for video element
setInterval(() => {
  if (!videoElement) {
    videoElement = findYouTubeVideo();
  }
}, 2000);

// Log when script loads
console.log('🎬 YouTube Caption Sync script loaded!');