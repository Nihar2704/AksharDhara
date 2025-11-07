import { useState, useEffect, useRef } from 'react';
import { Mic, MicOff, Settings, Copy, Trash2, AlertCircle } from 'lucide-react';

export default function SpeechTranslator() {
  const [isListening, setIsListening] = useState(false);
  const [transcript, setTranscript] = useState('');
  const [translation, setTranslation] = useState('');
  const [history, setHistory] = useState([]);
  const [srcLang, setSrcLang] = useState('eng_Latn');
  const [targetLang, setTargetLang] = useState('hin_Deva');
  const [showSettings, setShowSettings] = useState(false);
  const [status, setStatus] = useState('ready');
  const [error, setError] = useState('');
  const wsRef = useRef(null);
  const messagesEndRef = useRef(null);
  const mediaRecorderRef = useRef(null);
  const audioContextRef = useRef(null);
  const processorRef = useRef(null);

  const languages = {
    eng_Latn: 'English',
    hin_Deva: 'Hindi',
    fra_Latn: 'French',
    spa_Latn: 'Spanish',
    deu_Latn: 'German',
    jpn_Jpan: 'Japanese',
    zho_Hans: 'Mandarin',
    kor_Hang: 'Korean',
  };

  const WEBSOCKET_URL = 'ws://localhost:8000/ws/translate';

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [history]);

  const initializeWebSocket = () => {
    return new Promise((resolve, reject) => {
      try {
        wsRef.current = new WebSocket(WEBSOCKET_URL);

        wsRef.current.onopen = () => {
          console.log('WebSocket connected');
          wsRef.current.send(JSON.stringify({
            type: 'config',
            srcLang: srcLang,
            targetLang: targetLang,
          }));
          resolve();
        };

        wsRef.current.onmessage = (event) => {
          const message = JSON.parse(event.data);
          if (message.type === 'result') {
            setTranscript(message.transcript);
            setTranslation(message.translation);
            setStatus('ready');
          } else if (message.type === 'config_ack') {
            setStatus('ready');
          } else if (message.type === 'silence') {
            setStatus('silence detected');
          } else if (message.type === 'error') {
            setError(message.message);
            setStatus('error');
          }
        };

        wsRef.current.onerror = (err) => {
          console.error('WebSocket error:', err);
          setError('Connection error');
          setStatus('error');
          reject(err);
        };

        wsRef.current.onclose = () => {
          console.log('WebSocket disconnected');
        };
      } catch (err) {
        reject(err);
      }
    });
  };

  const startListening = async () => {
    try {
      setError('');
      setStatus('connecting');
      await initializeWebSocket();
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const audioContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
      audioContextRef.current = audioContext;
      const source = audioContext.createMediaStreamSource(stream);
      const processor = audioContext.createScriptProcessor(4096, 1, 1);
      processorRef.current = processor;
      processor.onaudioprocess = (e) => {
        const inputData = e.inputBuffer.getChannelData(0);
        const audioData = new Float32Array(inputData);
        if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
          const audioBase64 = arrayBufferToBase64(audioData);
          wsRef.current.send(JSON.stringify({
            type: 'audio',
            data: audioBase64,
          }));
        }
      };
      source.connect(processor);
      processor.connect(audioContext.destination);
      setIsListening(true);
      setStatus('listening');
      setTranscript('');
      setTranslation('');
    } catch (err) {
      console.error('Error starting listening:', err);
      setError('Microphone access denied or not available');
      setStatus('error');
    }
  };

  const stopListening = () => {
    setIsListening(false);
    setStatus('stopped');
    if (audioContextRef.current) audioContextRef.current.close();
    if (processorRef.current) processorRef.current.disconnect();
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ type: 'stop' }));
    }
  };

  const toggleListening = () => {
    if (isListening) stopListening();
    else startListening();
  };

  const arrayBufferToBase64 = (buffer) => {
    let binary = '';
    const bytes = new Uint8Array(buffer.buffer);
    for (let i = 0; i < bytes.byteLength; i++) binary += String.fromCharCode(bytes[i]);
    return btoa(binary);
  };

  const copyToClipboard = (text) => navigator.clipboard.writeText(text);
  const clearHistory = () => setHistory([]);
  const addToHistory = () => {
    if (transcript.trim()) {
      setHistory([...history, { id: Date.now(), transcript, translation, timestamp: new Date() }]);
      setTranscript('');
      setTranslation('');
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-[#FFFDD0] via-[#FFC175] to-[#A67B5B] text-gray-800 p-4">
      <div className="max-w-4xl mx-auto">
        {/* Header */}
        <div className="text-center mb-8 pt-4">
          <div>
            <nav className="flex justify-center gap-6 text-[#A67B5B] font-medium">
              <a href="#home" className="hover:text-[#8B5E3C]">Home</a>
              <a href="#about" className="hover:text-[#8B5E3C]">About</a>
              <a href="#services" className="hover:text-[#8B5E3C]">Services</a>
              <a href="#contact" className="hover:text-[#8B5E3C]">Contact</a>
            </nav>
          </div>
          <hr className="my-4 border-[#A67B5B]" />
          <h1 className="text-5xl font-bold mb-2 bg-clip-text text-transparent bg-gradient-to-r from-[#A67B5B] to-[#8B5E3C]">
            AksharDhara
          </h1>
          <h3 className="text-[#8B5E3C]">Real-Time Speech Translator</h3>
          <p className="text-[#A67B5B]/80">Translate speech instantly with AI</p>
        </div>

        {/* Error Alert */}
        {error && (
          <div className="bg-red-500/20 border border-red-500 rounded-lg p-4 mb-6 flex items-center gap-3">
            <AlertCircle size={20} />
            <p>{error}</p>
          </div>
        )}

        {/* Main Controls */}
        <div className="bg-[#FFC175] rounded-2xl p-8 mb-6 border border-[#A67B5B]">
          {/* Status Badge */}
          <div className="flex items-center justify-between mb-6">
            <div className="flex items-center gap-2">
              <div className={`w-3 h-3 rounded-full ${isListening ? 'bg-green-400 animate-pulse' : 'bg-gray-400'}`}></div>
              <span className="text-sm font-medium capitalize">{status}</span>
            </div>
            <button
              onClick={() => setShowSettings(!showSettings)}
              className="p-2 hover:bg-[#A67B5B]/20 rounded-lg transition"
            >
              <Settings size={20} />
            </button>
          </div>

          {/* Settings */}
          {showSettings && (
            <div className="bg-[#A67B5B]/20 rounded-lg p-4 mb-6 border border-[#8B5E3C]/40">
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="text-sm text-[#8B5E3C] block mb-2">Source Language</label>
                  <select
                    value={srcLang}
                    onChange={(e) => setSrcLang(e.target.value)}
                    className="w-full bg-[#A67B5B]/20 border border-[#8B5E3C] rounded-lg p-2 text-[#FFFDD0] text-sm focus:outline-none"
                  >
                    {Object.entries(languages).map(([code, name]) => (
                      <option key={code} value={code}>{name}</option>
                    ))}
                  </select>
                </div>
                <div>
                  <label className="text-sm text-[#8B5E3C] block mb-2">Target Language</label>
                  <select
                    value={targetLang}
                    onChange={(e) => setTargetLang(e.target.value)}
                    className="w-full bg-[#A67B5B]/20 border border-[#8B5E3C] rounded-lg p-2 text-[#FFFDD0] text-sm focus:outline-none"
                  >
                    {Object.entries(languages).map(([code, name]) => (
                      <option key={code} value={code}>{name}</option>
                    ))}
                  </select>
                </div>
              </div>
            </div>
          )}

          {/* Microphone Button */}
          <button
            onClick={toggleListening}
            className={`w-full py-6 rounded-xl font-bold text-lg transition-all transform hover:scale-105 mb-6 flex items-center justify-center gap-3 ${
              isListening
                ? 'bg-gradient-to-r from-[#8B5E3C] to-[#A67B5B] hover:from-[#A67B5B] hover:to-[#8B5E3C]'
                : 'bg-gradient-to-r from-[#A67B5B] to-[#FFC175] hover:from-[#FFC175] hover:to-[#A67B5B]'
            }`}
          >
            {isListening ? (
              <>
                <MicOff size={24} /> Stop Listening
              </>
            ) : (
              <>
                <Mic size={24} /> Start Listening
              </>
            )}
          </button>

          {/* Transcript Display */}
          <div className="bg-[#FFFDD0]/50 rounded-lg p-4 mb-4 border border-[#A67B5B]/40 min-h-20">
            <p className="text-xs text-[#8B5E3C] mb-2 uppercase tracking-wide">
              🗣️ Transcript ({languages[srcLang]})
            </p>
            <p className="text-lg break-words">
              {transcript || <span className="text-[#A67B5B]/60 italic">Start speaking...</span>}
            </p>
            {transcript && (
              <button
                onClick={() => copyToClipboard(transcript)}
                className="mt-2 text-xs bg-[#A67B5B]/20 hover:bg-[#8B5E3C]/30 px-2 py-1 rounded transition flex items-center gap-1"
              >
                <Copy size={12} /> Copy
              </button>
            )}
          </div>

          {/* Translation Display */}
          <div className="bg-[#FFFDD0]/50 rounded-lg p-4 border border-[#A67B5B]/40 min-h-20">
            <p className="text-xs text-[#8B5E3C] mb-2 uppercase tracking-wide">
              🌐 Translation ({languages[targetLang]})
            </p>
            <p className="text-lg break-words">
              {translation || <span className="text-[#A67B5B]/60 italic">Translation will appear here...</span>}
            </p>
            {translation && (
              <button
                onClick={() => copyToClipboard(translation)}
                className="mt-2 text-xs bg-[#A67B5B]/20 hover:bg-[#8B5E3C]/30 px-2 py-1 rounded transition flex items-center gap-1"
              >
                <Copy size={12} /> Copy
              </button>
            )}
          </div>

          {/* Add to History Button */}
          {transcript && (
            <button
              onClick={addToHistory}
              className="w-full mt-4 py-2 bg-[#8B5E3C] hover:bg-[#A67B5B] rounded-lg transition font-medium text-white"
            >
              Save to History
            </button>
          )}
        </div>

        {/* History Section */}
        {history.length > 0 && (
          <div className="bg-[#FFC175]/30 rounded-2xl p-8 border border-[#A67B5B]">
            <div className="flex items-center justify-between mb-6">
              <h2 className="text-2xl font-bold text-[#8B5E3C]">📋 History</h2>
              <button
                onClick={clearHistory}
                className="p-2 hover:bg-[#A67B5B]/20 rounded-lg transition text-red-400"
              >
                <Trash2 size={20} />
              </button>
            </div>

            <div className="space-y-4 max-h-96 overflow-y-auto">
              {history.map((item) => (
                <div key={item.id} className="bg-[#FFFDD0]/50 rounded-lg p-4 border border-[#A67B5B]/40">
                  <p className="text-xs text-[#8B5E3C] mb-2">
                    {item.timestamp.toLocaleTimeString()}
                  </p>
                  <p className="font-semibold mb-2">{item.transcript}</p>
                  <p className="text-[#8B5E3C]">{item.translation}</p>
                  <div className="flex gap-2 mt-2">
                    <button
                      onClick={() => copyToClipboard(item.transcript)}
                      className="text-xs bg-[#A67B5B]/20 hover:bg-[#8B5E3C]/30 px-2 py-1 rounded transition"
                    >
                      Copy Original
                    </button>
                    <button
                      onClick={() => copyToClipboard(item.translation)}
                      className="text-xs bg-[#A67B5B]/20 hover:bg-[#8B5E3C]/30 px-2 py-1 rounded transition"
                    >
                      Copy Translation
                    </button>
                  </div>
                </div>
              ))}
            </div>
            <div ref={messagesEndRef} />
          </div>
        )}
      </div>
    </div>
  );
}
