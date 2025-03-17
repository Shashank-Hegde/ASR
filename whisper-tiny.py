import streamlit as st
import torch
import sounddevice as sd
import numpy as np
import time
import io
import librosa
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq

# Cache the model and processor so they load only once
@st.cache_resource
def load_model():
    st.info("Loading Whisper-Tiny model (this may take a moment)...")
    processor = AutoProcessor.from_pretrained("openai/whisper-tiny")
    model = AutoModelForSpeechSeq2Seq.from_pretrained("openai/whisper-tiny")
    st.success("Model loaded successfully!")
    return processor, model.to("cpu")  # Using CPU for Streamlit Cloud

processor, model = load_model()

st.title("🎙️ Whisper-Tiny: Hindi → English Transcription")

# Let user choose input method
input_method = st.radio("Select input method:", ("Live Audio", "Upload WAV File"))

def transcribe_audio(audio, sample_rate=16000):
    # Process audio and generate transcription
    inputs = processor(audio, sampling_rate=sample_rate, return_tensors="pt")
    start_time = time.time()
    with torch.no_grad():
        generated_ids = model.generate(inputs.input_features)
    transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    exec_time = time.time() - start_time
    return transcription, exec_time

if input_method == "Live Audio":
    st.markdown("### Live Audio Input")
    duration = st.slider("Recording duration (seconds)", min_value=2, max_value=10, value=5)
    if st.button("Start Recording"):
        st.info("Recording... Please speak in Hindi now!")
        audio = sd.rec(int(duration * 16000), samplerate=16000, channels=1, dtype="float32")
        sd.wait()
        audio = audio.flatten()
        st.audio(audio, sample_rate=16000, format="audio/wav")
        with st.spinner("Transcribing audio..."):
            transcript, exec_time = transcribe_audio(audio)
        st.success(f"Transcription completed in {exec_time:.2f} seconds!")
        st.subheader("Transcription:")
        st.write(transcript)
elif input_method == "Upload WAV File":
    st.markdown("### Upload Audio File")
    uploaded_file = st.file_uploader("Upload a WAV file", type=["wav"])
    if uploaded_file is not None:
        # Read the uploaded file as bytes, then load with librosa
        audio_bytes = uploaded_file.read()
        audio_buffer = io.BytesIO(audio_bytes)
        # librosa.load accepts a file-like object
        audio, sr = librosa.load(audio_buffer, sr=16000)
        st.audio(audio_bytes, format="audio/wav")
        with st.spinner("Transcribing audio..."):
            transcript, exec_time = transcribe_audio(audio, sample_rate=sr)
        st.success(f"Transcription completed in {exec_time:.2f} seconds!")
        st.subheader("Transcription:")
        st.write(transcript)
