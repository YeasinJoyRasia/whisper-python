import sounddevice as sd
import numpy as np
import queue
import time
import tempfile
import os
from scipy.io.wavfile import write
from pywhispercpp.model import Model

MODEL_PATH = "base.en"
SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK_SECONDS = 4
OVERLAP_SECONDS = 1

audio_queue = queue.Queue()
model = Model(MODEL_PATH)

def audio_callback(indata, frames, time_info, status):
    if status:
        print(status)
    audio_queue.put(indata.copy())

def record_and_transcribe():
    buffer = np.zeros((0, CHANNELS), dtype=np.float32)
    last_printed = ""

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        callback=audio_callback,
        blocksize=int(SAMPLE_RATE * 0.5),
        dtype="float32",
    ):
        print("🎙️ Listening... Press Ctrl+C to stop\n")

        while True:
            while not audio_queue.empty():
                buffer = np.concatenate((buffer, audio_queue.get()))

            if len(buffer) >= SAMPLE_RATE * CHUNK_SECONDS:
                chunk = buffer[: SAMPLE_RATE * CHUNK_SECONDS]
                buffer = buffer[int(SAMPLE_RATE * (CHUNK_SECONDS - OVERLAP_SECONDS)) :]

                # --- WRITE PCM16 WAV (fix for wave.Error: unknown format: 3) ---
                mono = chunk[:, 0] if chunk.ndim > 1 else chunk
                pcm16 = np.clip(mono, -1.0, 1.0)
                pcm16 = (pcm16 * 32767).astype(np.int16)

                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                    write(f.name, SAMPLE_RATE, pcm16)
                    wav_path = f.name

                segments = model.transcribe(wav_path)
                os.unlink(wav_path)

                text = " ".join(seg.text for seg in segments).strip()

                if text and text != last_printed:
                    print(text)
                    last_printed = text

            time.sleep(0.05)

if __name__ == "__main__":
    try:
        record_and_transcribe()
    except KeyboardInterrupt:
        print("\n🛑 Stopped")
