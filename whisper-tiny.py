import streamlit as st
import numpy as np
import torch
import time
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase

st.title("🎙️ Whisper-Tiny: Live Hindi-to-English Transcription")

# Load the model and processor (cache them so they load only once)
@st.cache_resource
def load_model():
    st.info("Loading Whisper-Tiny model... This might take a few moments.")
    processor = AutoProcessor.from_pretrained("openai/whisper-tiny")
    model = AutoModelForSpeechSeq2Seq.from_pretrained("openai/whisper-tiny").to("cpu")
    st.success("Model loaded successfully!")
    return processor, model

processor, model = load_model()

# Define an audio processor for live audio recording using st-webrtc
class AudioRecorder(AudioProcessorBase):
    def __init__(self):
        self.frames = []

    def recv(self, frame):
        # frame is an av.AudioFrame
        arr = frame.to_ndarray()  # shape: (channels, samples)
        # Convert to mono by averaging channels if necessary
        if arr.ndim == 2:
            arr = np.mean(arr, axis=0, keepdims=True)
        self.frames.append(arr)
        return frame

# Start the webrtc streamer to capture live audio
webrtc_ctx = webrtc_streamer(
    key="audio",
    media_stream_constraints={"audio": True, "video": False},
    audio_processor_factory=AudioRecorder,
)

st.markdown("**Instructions:** Click the built-in stop button in the browser widget to finish recording.")

# Once the recording is finished, process the audio when the "Transcribe" button is clicked.
if webrtc_ctx.audio_processor:
    if st.button("Transcribe Audio"):
        # Concatenate all recorded audio frames along the sample axis
        recorded_frames = webrtc_ctx.audio_processor.frames
        if recorded_frames:
            audio_data = np.concatenate(recorded_frames, axis=1).flatten()
            st.audio(audio_data, sample_rate=16000, format="audio/wav")
            
            st.info("Transcribing...")
            start_time = time.time()
            # Prepare inputs for the model
            inputs = processor(audio_data, sampling_rate=16000, return_tensors="pt")
            with torch.no_grad():
                generated_ids = model.generate(inputs.input_features)
            transcript = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            end_time = time.time()
            st.success(f"Transcription completed in {end_time - start_time:.2f} seconds.")
            st.subheader("🎙️ Hindi → English Transcription")
            st.write(transcript)
        else:
            st.error("No audio was recorded. Please try recording again.")
