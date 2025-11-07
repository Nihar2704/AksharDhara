let websocket = null;
let keepAliveInterval = null;

// Connect to WebSocket server
function connectWebSocket() {
  if (websocket && websocket.readyState === WebSocket.OPEN) {
    console.log('Already connected');
    return;
  }
  
  websocket = new WebSocket('ws://localhost:8765');
  
  websocket.onopen = () => {
    console.log('✓ Connected to server');
    keepAlive();
  };
  
  websocket.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data);
      console.log('Received:', data);
      
      // Forward to popup
      chrome.runtime.sendMessage({
        type: 'caption',
        data: data
      }).catch(err => {
        // Popup might be closed, that's okay
        console.log('Popup not open:', err);
      });
      
      // Forward to content script for overlay display
      chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
        if (tabs[0]) {
          chrome.tabs.sendMessage(tabs[0].id, {
            type: 'caption',
            data: data
          }).catch(err => {
            console.log('Content script not ready:', err);
          });
        }
      });
      
    } catch (error) {
      console.error('Message parse error:', error);
    }
  };
  
  websocket.onerror = (error) => {
    console.error('WebSocket error:', error);
  };
  
  websocket.onclose = () => {
    console.log('✗ Disconnected from server');
    websocket = null;
    if (keepAliveInterval) {
      clearInterval(keepAliveInterval);
      keepAliveInterval = null;
    }
  };
}

// Keep service worker alive
function keepAlive() {
  keepAliveInterval = setInterval(() => {
    if (websocket && websocket.readyState === WebSocket.OPEN) {
      websocket.send(JSON.stringify({ action: 'ping' }));
    } else {
      clearInterval(keepAliveInterval);
      keepAliveInterval = null;
    }
  }, 20000);
}

// Handle messages from popup
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if (request.action === 'start') {
    connectWebSocket();
    
    setTimeout(() => {
      if (websocket && websocket.readyState === WebSocket.OPEN) {
        // Send URL to server
        websocket.send(JSON.stringify({
          action: 'set_url',
          url: request.url
        }));
        
        // Start transcription
        setTimeout(() => {
          websocket.send(JSON.stringify({
            action: 'start_transcription'
          }));
        }, 1000);
        
        sendResponse({ success: true });
      } else {
        sendResponse({ success: false, error: 'Connection failed' });
      }
    }, 1000);
    
    return true;
  }
  
  else if (request.action === 'stop') {
    if (websocket) {
      websocket.send(JSON.stringify({ action: 'stop' }));
      websocket.close();
    }
    sendResponse({ success: true });
  }
  
  else if (request.action === 'status') {
    sendResponse({
      connected: websocket && websocket.readyState === WebSocket.OPEN
    });
  }
});

// Connect on startup
connectWebSocket();