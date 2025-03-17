import streamlit as st
import torch
import numpy as np
import time
import io
import librosa
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq

st.title("🎙️ Whisper‑Tiny: Hindi → English Transcription")

# Cache model to avoid reloading on every interaction
@st.cache_resource
def load_model():
    st.info("Loading Whisper‑Tiny model... This may take a moment on first run.")
    processor = AutoProcessor.from_pretrained("openai/whisper-tiny")
    model = AutoModelForSpeechSeq2Seq.from_pretrained("openai/whisper-tiny").to("cpu")
    st.success("Model loaded successfully!")
    return processor, model

processor, model = load_model()

def transcribe_audio(audio, sample_rate=16000):
    # Prepare the input features from the uploaded audio
    inputs = processor(audio, sampling_rate=sample_rate, return_tensors="pt")
    start_time = time.time()
    with torch.no_grad():
        generated_ids = model.generate(inputs.input_features)
    transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    end_time = time.time()
    exec_time = end_time - start_time
    return transcription, exec_time

st.markdown("### Upload a WAV file for transcription")
uploaded_file = st.file_uploader("Choose a WAV file", type=["wav"])

if uploaded_file is not None:
    # Read uploaded file bytes
    audio_bytes = uploaded_file.read()
    audio_buffer = io.BytesIO(audio_bytes)
    # Load the audio using librosa (resample to 16000 Hz)
    audio, sr = librosa.load(audio_buffer, sr=16000)
    # Display the audio so the user can listen to it
    st.audio(audio_bytes, format="audio/wav", sample_rate=16000)
    
    with st.spinner("Transcribing audio..."):
        transcript, exec_time = transcribe_audio(audio, sample_rate=sr)
    st.success(f"Transcription completed in {exec_time:.2f} seconds.")
    st.subheader("Transcription:")
    st.write(transcript)
