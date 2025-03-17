import streamlit as st
import torch
import numpy as np
import soundfile as sf
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq

# Cache model to avoid re-downloading
@st.cache_resource
def load_model():
    st.info("🔄 Loading Whisper-Tiny model...")
    processor = AutoProcessor.from_pretrained("openai/whisper-tiny")
    model = AutoModelForSpeechSeq2Seq.from_pretrained("openai/whisper-tiny")
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    return processor, model

# Load model once
processor, model = load_model()

# Transcribe function
def transcribe_audio(audio):
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        generated_ids = model.generate(inputs.input_features)
    
    transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return transcription

# Streamlit UI
st.title("🎙️ Whisper-Tiny Hindi to English Transcription")

st.markdown("Upload a Hindi audio file (WAV format) to transcribe.")

# Upload audio file
uploaded_file = st.file_uploader("Choose an audio file", type=["wav"])

if uploaded_file is not None:
    st.audio(uploaded_file, format="audio/wav")

    # Read the file
    audio_data, sample_rate = sf.read(uploaded_file)

    st.write("📄 **Transcribing... Please wait...**")

    # Transcribe audio
    transcript = transcribe_audio(audio_data)

    # Display result
    st.success("✅ Transcription Complete!")
    st.subheader("🎙️ Hindi → English Transcription")
    st.write(transcript)
