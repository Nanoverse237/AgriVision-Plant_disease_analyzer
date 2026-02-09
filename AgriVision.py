import streamlit as st
import cv2
import time
import os
import json
import base64
from PIL import Image
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration, WebRtcMode
from concurrent.futures import ThreadPoolExecutor
from google import genai
from google.genai import types

# --- Configuration ---
st.set_page_config(
    page_title="AgriVision - AI Crop Diagnostics",
    page_icon="🌿",
    layout="wide"
)

# --- Styles ---
st.markdown("""
<style>
    .report-box {
        border: 2px solid #e0e7ff;
        padding: 20px;
        border-radius: 10px;
        background-color: #f0fdf4;
    }
    .disease-title {
        color: #166534;
        font-size: 24px;
        font-weight: bold;
    }
    .confidence-high {
        color: #15803d;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)
# --- API Setup ---
# Get key from environment variable
api_key = os.environ.get("GEMINI_API_KEY")

# --- Initialization ---
if 'api_key' not in st.session_state:
    st.session_state.api_key = os.environ.get("GEMINI_API_KEY", "")

if 'analysis_result' not in st.session_state:
    st.session_state.analysis_result = None

if 'thread_pool' not in st.session_state:
    st.session_state.thread_pool = ThreadPoolExecutor(max_workers=1)

# --- Helper Functions ---

def get_gemini_client(api_key):
    if not api_key:
        return None
    return genai.Client(api_key=api_key)

def analyze_image(api_key: str, image):
    """
    Sends an image (PIL or numpy array) to Gemini for disease analysis.
    Creates its own client to be safe in background threads.
    """
    try:
        client = get_gemini_client(api_key)
        if client is None:
            return {"error": "Missing API key. Please provide GEMINI_API_KEY."}

        # Convert numpy array (OpenCV) to PIL
        if isinstance(image, np.ndarray):
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)

        prompt = """
You are AgriVision, an expert plant pathologist. Analyze this image.
1. Identify the plant.
2. Detect if there is a disease, pest, or nutrient deficiency.
3. If healthy, state "Healthy".
4. Provide treatment recommendations if an issue is found.

Return the result as a raw JSON object with this schema:
{
  "plant": "str",
  "condition": "str",
  "is_healthy": bool,
  "confidence": "float (0-1)",
  "description": "str (concise)",
  "treatments": ["str", "str"]
}
"""

        cfg = types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema={
                "type": "object",
                "properties": {
                    "plant": {"type": "string"},
                    "condition": {"type": "string"},
                    "is_healthy": {"type": "boolean"},
                    "confidence": {"type": "number"},
                    "description": {"type": "string"},
                    "treatments": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["plant", "condition", "is_healthy", "confidence", "description", "treatments"],
            },
        )

        response = client.models.generate_content(
            model="gemini-3-pro-preview",
            contents=[image, prompt],
            config=cfg,
        )

        return json.loads(response.text)

    except Exception as e:
        return {"error": str(e)}

def display_report(result, confidence_threshold: float = 0.0):
    if not result:
        st.info("No analysis yet.")
        return

    if "error" in result:
        st.error(f"Analysis Failed: {result['error']}")
        return

    conf = float(result.get("confidence", 0.0))
    if conf < confidence_threshold:
        st.warning(f"Low confidence result ({conf:.2f}) — try a clearer close-up photo/lighting.")
        # Still show what it guessed:
        # return  # optionally stop here

    color = "green" if result.get("is_healthy") else "red"

    st.markdown(f"""
    <div class="report-box">
        <div class="disease-title" style="color: {color}">
            {result.get('condition', 'Unknown Condition')}
        </div>
        <p><b>Plant:</b> {result.get('plant', 'Unknown')} |
           <b>Confidence:</b> {conf:.2f}</p>
        <p>{result.get('description', '')}</p>
    </div>
    """, unsafe_allow_html=True)

    if not result.get("is_healthy") and result.get("treatments"):
        st.subheader("💊 Recommended Treatments")
        for treatment in result["treatments"]:
            st.info(treatment)

# --- streamlit-webrtc: Video Processor ---
class VideoProcessor(VideoProcessorBase):
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        st.session_state.latest_frame_bgr = img
        return frame


RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# --- Sidebar ---
with st.sidebar:
    st.title("AgriVision")
    st.write("Real-time AI Crop Analyzer for Better Crop Production")
    
    if 'api_key' not in st.session_state:
        st.session_state.api_key = os.environ.get("GEMINI_API_KEY", "")

        st.session_state.api_key = st.text_input(
        "Enter Google API Key",
        type="password",
        value=st.session_state.api_key
    )
        
    st.divider()
    st.write("### Settings")
    analysis_interval = st.slider("Analysis Interval (seconds)", 1, 10, 3)
    confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.6)

# --- Main App ---
tab1, tab2 = st.tabs(["Live Stream", "Upload Image/Video"])

with tab1:
    col_video, col_info = st.columns([2, 1])

    with col_video:
        st.subheader("Live Feed")
        ctx = webrtc_streamer(
            key="agrivision-live",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTC_CONFIGURATION,
            media_stream_constraints={"video": True, "audio": False},
            video_processor_factory=VideoProcessor,
            async_processing=True,
        )

    with col_info:
        st.subheader("Real-time Analysis")
        st_report = st.empty()

    # --- Analysis loop (Streamlit-friendly "tick") ---
    if ctx.state.playing:
        now = time.time()

        # Update UI with last known result immediately
        with st_report.container():
            display_report(st.session_state.analysis_result, confidence_threshold)

        # If a background task finished, collect result
        if st.session_state.future and st.session_state.future.done():
            st.session_state.analysis_result = st.session_state.future.result()
            st.session_state.future = None

            with st_report.container():
                display_report(st.session_state.analysis_result, confidence_threshold)

        # Schedule new analysis if interval passed and we have a frame
        if (
            st.session_state.future is None
            and (now - st.session_state.last_analysis_time) >= analysis_interval
            and st.session_state.latest_frame_bgr is not None
        ):
            frame_copy = st.session_state.latest_frame_bgr.copy()
            st.session_state.future = st.session_state.thread_pool.submit(
                analyze_image,
                st.session_state.api_key,
                frame_copy
            )
            st.session_state.last_analysis_time = now
    else:
        with st_report.container():
            st.info("Click **Start** above and allow camera permissions in your browser.")

with tab2:
    st.header("Upload Analysis")
    uploaded_file = st.file_uploader("Choose an image or video", type=['jpg', 'png', 'jpeg', 'mp4'])
    
    if uploaded_file:
        # If it's an image
        if uploaded_file.type.startswith('image'):
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", width=400)
            
            if st.button("Analyze Image"):
                with st.spinner("Analyzing..."):
                    result = analyze_image(st.session_state.api_key, image)
                    display_report(result)
        
        # If it's a video
        elif uploaded_file.type.startswith('video'):
            st.video(uploaded_file)
            st.info("Video analysis: Extracting frames for analysis...")
            
            if st.button("Analyze Video"):
                # Save temp file
                with open("temp_video.mp4", "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # Use File API for full video understanding (better for temporal context)
                with st.spinner("Uploading video to Gemini..."):
                    video_file = client.files.upload(file="temp_video.mp4")
                    
                    # Wait for processing
                    while video_file.state.name == "PROCESSING":
                        time.sleep(2)
                        video_file = client.files.get(name=video_file.name)
                        
                    if video_file.state.name == "FAILED":
                        st.error("Video processing failed.")
                    else:
                        st.success("Video processed. Generating insight...")
                        prompt = "Analyze this video of a crop. Identify any diseases present in the footage and suggest treatments. Return JSON."
                        
                        response = client.models.generate_content(
                            model='gemini-3-pro-preview',
                            contents=[video_file, prompt],
                             config={
                                'response_mime_type': 'application/json',
                                'response_schema': {
                                    'type': types.Type.OBJECT,
                                    'properties': {
                                        'plant': {'type': types.Type.STRING},
                                        'condition': {'type': types.Type.STRING},
                                        'is_healthy': {'type': types.Type.BOOLEAN},
                                        'confidence': {'type': types.Type.NUMBER},
                                        'description': {'type': types.Type.STRING},
                                        'treatments': {
                                            'type': types.Type.ARRAY,
                                            'items': {'type': types.Type.STRING}
                                        }
                                    }
                                }
                            }
                        )
                        display_report(json.loads(response.text))

