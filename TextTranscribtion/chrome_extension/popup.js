const startBtn = document.getElementById('startBtn');
const stopBtn = document.getElementById('stopBtn');
const clearBtn = document.getElementById('clearBtn');
const status = document.getElementById('status');
const transcriptContainer = document.getElementById('transcriptContainer');
const progressFill = document.getElementById('progressFill');

let captions = [];

// Get current YouTube URL
async function getCurrentYouTubeURL() {
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  if (tab && tab.url && tab.url.includes('youtube.com/watch')) {
    return tab.url;
  }
  return null;
}

// Update status display
function updateStatus(message, statusType = 'disconnected') {
  status.textContent = message;
  status.className = statusType;
}

// Update progress bar
function updateProgress(percent) {
  progressFill.style.width = percent + '%';
}

// Format time in MM:SS
function formatTime(seconds) {
  const mins = Math.floor(seconds / 60);
  const secs = Math.floor(seconds % 60);
  return `${mins}:${secs.toString().padStart(2, '0')}`;
}

// Add caption to display
function addCaption(data) {
  const captionItem = document.createElement('div');
  captionItem.className = 'caption-item';
  
  const header = document.createElement('div');
  header.className = 'caption-header';
  
  const timeInfo = document.createElement('span');
  timeInfo.className = 'caption-time';
  timeInfo.textContent = `⏱️ ${formatTime(data.start_time_s)}`;
  
  const progressInfo = document.createElement('span');
  progressInfo.className = 'caption-progress';
  progressInfo.textContent = `${data.chunk_index + 1}/${data.total_chunks} (${data.progress}%)`;
  
  header.appendChild(timeInfo);
  header.appendChild(progressInfo);
  
  // Original text
  const originalSection = document.createElement('div');
  originalSection.className = 'caption-text';
  
  const originalLabel = document.createElement('div');
  originalLabel.className = 'caption-label';
  originalLabel.textContent = '🎤 Original';
  
  const originalText = document.createElement('div');
  originalText.className = 'original-text';
  originalText.textContent = data.text || '[No transcription]';
  
  originalSection.appendChild(originalLabel);
  originalSection.appendChild(originalText);
  
  // Translated text
  const translatedSection = document.createElement('div');
  translatedSection.className = 'caption-text';
  
  const translatedLabel = document.createElement('div');
  translatedLabel.className = 'caption-label';
  translatedLabel.textContent = '🌐 Translation';
  
  const translatedText = document.createElement('div');
  translatedText.className = 'translated-text';
  translatedText.textContent = data.translated_text || '[No translation]';
  
  translatedSection.appendChild(translatedLabel);
  translatedSection.appendChild(translatedText);
  
  // Assemble
  captionItem.appendChild(header);
  captionItem.appendChild(originalSection);
  captionItem.appendChild(translatedSection);
  
  // Clear empty state if exists
  const emptyState = transcriptContainer.querySelector('.empty-state');
  if (emptyState) {
    emptyState.remove();
  }
  
  transcriptContainer.appendChild(captionItem);
  
  // Auto-scroll to bottom
  transcriptContainer.scrollTop = transcriptContainer.scrollHeight;
  
  // Store caption
  captions.push(data);
}

// Clear all captions
function clearCaptions() {
  captions = [];
  transcriptContainer.innerHTML = '<div class="empty-state">Captions cleared</div>';
  updateProgress(0);
}

// Listen for messages from background
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.type === 'caption') {
    const data = message.data;
    
    if (data.event === 'connected') {
      updateStatus('Connected to server', 'connected');
    }
    else if (data.event === 'status') {
      updateStatus(data.message, 'processing');
    }
    else if (data.event === 'ready') {
      updateStatus(`Ready: ${data.total_chunks} chunks`, 'connected');
    }
    else if (data.event === 'transcription_start') {
      updateStatus('Transcribing...', 'processing');
      clearCaptions();
    }
    else if (data.event === 'caption') {
      addCaption(data);
      updateProgress(data.progress);
      updateStatus(`Processing: ${data.progress}%`, 'processing');
    }
    else if (data.event === 'complete') {
      updateStatus('Complete! ✓', 'connected');
      updateProgress(100);
    }
    else if (data.event === 'error') {
      updateStatus('Error: ' + data.message, 'disconnected');
    }
    else if (data.event === 'stopped') {
      updateStatus('Stopped', 'disconnected');
    }
  }
});

// Start transcription
startBtn.addEventListener('click', async () => {
  const url = await getCurrentYouTubeURL();
  
  if (!url) {
    updateStatus('Please open a YouTube video', 'disconnected');
    return;
  }
  
  updateStatus('Starting...', 'processing');
  clearCaptions();
  
  // Send message to background script
  chrome.runtime.sendMessage({
    action: 'start',
    url: url
  }, (response) => {
    if (chrome.runtime.lastError) {
      console.error('Runtime error:', chrome.runtime.lastError);
      updateStatus('Connection error. Please refresh.', 'disconnected');
      return;
    }
    
    if (!response) {
      updateStatus('No response from server. Is it running?', 'disconnected');
      return;
    }
    
    if (response.success) {
      updateStatus('Processing...', 'processing');
    } else {
      updateStatus('Error: ' + (response.error || 'Unknown error'), 'disconnected');
    }
  });
});

// Stop transcription
stopBtn.addEventListener('click', () => {
  chrome.runtime.sendMessage({ action: 'stop' }, (response) => {
    if (chrome.runtime.lastError) {
      console.error('Runtime error:', chrome.runtime.lastError);
      updateStatus('Already stopped', 'disconnected');
      return;
    }
    
    updateStatus('Stopped', 'disconnected');
  });
});

// Clear captions
clearBtn.addEventListener('click', () => {
  clearCaptions();
  updateStatus('Captions cleared', 'disconnected');
});

// Check connection status on popup open
chrome.runtime.sendMessage({ action: 'status' }, (response) => {
  if (chrome.runtime.lastError) {
    console.error('Runtime error:', chrome.runtime.lastError);
    updateStatus('Server not running', 'disconnected');
    return;
  }
  
  if (response && response.connected) {
    updateStatus('Connected', 'connected');
  } else {
    updateStatus('Server not connected', 'disconnected');
  }
});