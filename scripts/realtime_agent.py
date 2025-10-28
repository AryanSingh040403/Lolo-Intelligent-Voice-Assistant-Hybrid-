import os
import time
import queue
import threading
from typing import Optional
import numpy as np
import torch

# Conditional imports (requires installation of dependencies)
try:
    import pyaudio # Mic capture
    import faster_whisper # ASR
    # import sounddevice as sd # Audio output (optional, can be used with Coqui)
    # from TTS.api import TTS # Coqui TTS
except ImportError:
    print("Warning: Voice I/O libraries (pyaudio/faster-whisper/coqui-tts) not found. Running in simulation mode.")

# --- Configuration and Constants ---
# STT Configuration (16kHz standard for ASR)
FRAME_RATE = 16000 
CHUNK = 1024       
CHANNELS = 1
AUDIO_FORMAT = 8   # pyaudio.paInt16
# TTS Configuration (Coqui XTTSv2 requires 24kHz) [12]
TTS_SAMPLE_RATE = 24000
XTTS_MODEL_ID = "tts_models/multilingual/multi-dataset/xtts_v2"

# Agent Core Import (Placeholder for integration)
from agent_core import QwenLLMAgent

class RealTimeVoiceAgent:
    """
    Implements a three-threaded Producer-Consumer pipeline for real-time voice interaction:
    1. STT Producer (Mic Capture)
    2. LLM Worker (ASR Processing, Agent Execution, Streaming SBD)
    3. TTS Consumer (Audio Synthesis & Playback)
    """
    def __init__(self, llm_agent_instance: QwenLLMAgent):
        self.agent = llm_agent_instance
        self.is_running = threading.Event()
        self.is_running.set() 

        # Queues for decoupled communication [10]
        self.audio_in_q = queue.Queue()    # Raw audio chunks (Mic -> ASR)
        self.text_to_llm_q = queue.Queue() # Complete transcription (ASR -> LLM)
        self.tts_out_q = queue.Queue()     # Streaming response sentences (LLM -> TTS)

        # Thread containers
        self.mic_thread = threading.Thread(target=self._run_stt_producer, daemon=True)
        self.llm_thread = threading.Thread(target=self._run_llm_worker, daemon=True)
        self.tts_thread = threading.Thread(target=self._run_tts_consumer, daemon=True)

    # --- Thread 1: STT Producer (Mic Capture) [10] ---
    def _run_stt_producer(self):
        """Captures raw audio from the microphone and puts it in the audio_in_q."""
        try:
            p = pyaudio.PyAudio()
            stream = p.open(format=AUDIO_FORMAT, channels=CHANNELS, rate=FRAME_RATE,
                            input=True, frames_per_buffer=CHUNK)

            print("🎤 STT Producer: Recording started.")
            while self.is_running.is_set():
                # Read chunks from mic and immediately buffer them [10]
                data = stream.read(CHUNK, exception_on_overflow=False)
                self.audio_in_q.put(data)
            
            stream.stop_stream()
            stream.close()
            p.terminate()
            print("🛑 STT Producer: Recording stopped.")

        except Exception as e:
            print(f"STT Producer Error: {e}")
            self.stop() 

    # --- Thread 2: LLM Worker (ASR Processing & Agent Orchestration) ---
    def _run_llm_worker(self):
        """
        Transcribes audio, executes the Qwen Agent, and streams
        response sentences using Sentence Boundary Detection (SBD).
        """
        if not 'faster_whisper' in globals():
             print("LLM Worker running in transcription simulation mode.")
             return
             
        whisper_model = faster_whisper.WhisperModel("small", device="cuda" if torch.cuda.is_available() else "cpu")
        audio_buffer =
        
        while self.is_running.is_set() or not self.audio_in_q.empty():
            # 1. ASR Processing [10]
            if not self.audio_in_q.empty():
                data = self.audio_in_q.get()
                audio_buffer.append(np.frombuffer(data, dtype=np.int16))

                # Check if enough audio for approx. 1 second of transcription (FRAME_RATE/CHUNK)
                if (len(audio_buffer) * CHUNK) >= FRAME_RATE: 
                    audio_chunk = np.concatenate(audio_buffer).astype(np.float32) / 32768
                    audio_buffer =

                    # Use VAD filter for low-latency end-of-speech detection [31]
                    segments, _ = whisper_model.transcribe(
                        audio_chunk, 
                        language="en", 
                        vad_filter=True,
                        vad_parameters=dict(min_silence_duration_ms=500)
                    )
                    
                    full_transcript = " ".join([seg.text for seg in segments]).strip()
                    if full_transcript:
                        print(f"Transcription: {full_transcript}")
                        self.text_to_llm_q.put(full_transcript)
            
            # 2. LLM Agent Execution (Blocks until a transcript is ready)
            if not self.text_to_llm_q.empty():
                user_query = self.text_to_llm_q.get()
                print(f"\n🧠 LLM Worker: Processing query: {user_query}")
                
                # --- LLM Streaming Logic (Conceptual) ---
                # This should interface with vLLM's streaming API via ChatOpenAI.stream()
                # and apply SBD to feed the TTS queue sentence-by-sentence.
                
                simulated_response = "The Qwen agent processed your query. The three-threaded architecture allows the generation and synthesis to run concurrently, minimizing the time to first audio output."
                
                # Simple SBD simulation to feed the TTS consumer
                sentences = simulated_response.split(". ")
                for sentence in sentences:
                    if sentence:
                        self.tts_out_q.put(sentence.strip() + ".")
                        time.sleep(0.05) # Simulate token decoding time
                
                self.tts_out_q.put(None) # Signal end of response stream

        print("🛑 LLM Worker: Stopping.")

    # --- Thread 3: TTS Consumer (Synthesis and Playback) [32, 33] ---
    def _run_tts_consumer(self):
        """
        Consumes sentences from the LLM output queue and streams audio 
        chunks using Coqui XTTSv2 and sounddevice (conceptual).
        """
        try:
            # tts = TTS(XTTS_MODEL_ID, gpu=torch.cuda.is_available())
            import sounddevice as sd
            print("🔊 TTS Consumer: Ready to synthesize audio.")
            # Placeholder for speaker conditioning latents
            # gpt_cond_latent, speaker_embedding = tts.get_conditioning_latents(audio_path=...)
            
        except (ImportError, RuntimeError):
            print("TTS Consumer: Coqui TTS/Sounddevice not initialized. Running in simulation.")
            return

        while self.is_running.is_set() or not self.tts_out_q.empty():
            sentence = self.tts_out_q.get()
            
            if sentence is None and not self.is_running.is_set():
                break
            if sentence is None:
                continue
                
            print(f"TTS Synthesizing: '{sentence}'")
            
            # --- XTTS Streaming Logic Placeholder [33, 34] ---
            # with sd.OutputStream(samplerate=TTS_SAMPLE_RATE, channels=1) as stream:
            #     for chunk in tts.inference_stream(sentence, language='en',...):
            #         stream.write(chunk)
            
            # Simulate low-latency audio playback (Critical for TTFA)
            time.sleep(len(sentence) * 0.02) 

        print("🛑 TTS Consumer: Stopping.")


    def start(self):
        """Starts all three worker threads."""
        print("Starting Lolo v2.0 Real-Time Voice Agent...")
        self.mic_thread.start()
        self.llm_thread.start()
        self.tts_thread.start()

    def stop(self):
        """Signals all threads to stop and waits for them to join."""
        self.is_running.clear()
        
        # Unblock queues to allow threads to exit gracefully
        self.audio_in_q.put(None)
        self.text_to_llm_q.put(None)
        self.tts_out_q.put(None)

        self.mic_thread.join(timeout=1)
        self.llm_thread.join(timeout=1)
        self.tts_thread.join(timeout=1)
        print("Agent shutdown complete.")

if __name__ == "__main__":
    # Ensure all components are ready before running the agent
    print("Pre-flight check: Ensure vLLM is running and FAISS index is created.")
    
    # Placeholder initialization for the QwenLLMAgent
    try:
        agent_instance = QwenLLMAgent()
        voice_agent = RealTimeVoiceAgent(agent_instance)
        
        voice_agent.start()
        
        # Keep the main thread alive until user interruption
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nKeyboard interrupt received. Shutting down agent...")
        voice_agent.stop()
    except Exception as e:
        print(f"Failed to start real-time agent: {e}")