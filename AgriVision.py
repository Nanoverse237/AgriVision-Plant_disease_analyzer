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
import streamlit as st
import cv2
import time
import os
import json
import tempfile
import threading
from PIL import Image, UnidentifiedImageError
import numpy as np
from concurrent.futures import ThreadPoolExecutor

from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration, WebRtcMode

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
</style>
""", unsafe_allow_html=True)

# --- API Setup ---
api_key_env = os.environ.get("GEMINI_API_KEY", "")

# --- Initialization ---
if "api_key" not in st.session_state:
    st.session_state.api_key = api_key_env

if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None

if "thread_pool" not in st.session_state:
    st.session_state.thread_pool = ThreadPoolExecutor(max_workers=1)

if "future" not in st.session_state:
    st.session_state.future = None

if "last_analysis_time" not in st.session_state:
    st.session_state.last_analysis_time = 0.0

# Shared frame store (thread-safe) for WebRTC worker -> Streamlit main thread
if "frame_lock" not in st.session_state:
    st.session_state.frame_lock = threading.Lock()
if "latest_frame_bgr" not in st.session_state:
    st.session_state.latest_frame_bgr = None


# --- Helper Functions ---
def get_gemini_client(api_key: str):
    if not api_key:
        return None
    return genai.Client(api_key=api_key)


def analyze_image(api_key: str, image):
    """Analyze an image (PIL or BGR numpy) via Gemini, returning JSON dict."""
    try:
        client = get_gemini_client(api_key)
        if client is None:
            return {"error": "Missing API key. Please provide GEMINI_API_KEY."}

        if isinstance(image, np.ndarray):
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)

        prompt = """
You are AgriVision, an expert plant pathologist. Analyze this image.
1. Identify the plant.
2. Detect if there is a disease, pest, or nutrient deficiency.
3. If healthy, state "Healthy".
4. Provide treatment recommendations if an issue is found.

Return ONLY a raw JSON object with this schema:
{
  "plant": "str",
  "condition": "str",
  "is_healthy": bool,
  "confidence": float (0-1),
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


def analyze_video_file(api_key: str, video_path: str, max_frames: int = 6) -> dict:
    """
    Robust "any video" analysis by sampling frames and aggregating with Gemini.
    (Works even when MIME types are wrong.)
    """
    try:
        client = get_gemini_client(api_key)
        if client is None:
            return {"error": "Missing API key. Please provide GEMINI_API_KEY."}

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {"error": "Could not open video file."}

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        step = max(total // max_frames, 1)

        frames = []
        idx = 0
        grabbed = 0
        while grabbed < max_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
            grabbed += 1
            idx += step
        cap.release()

        if not frames:
            return {"error": "Could not read frames from video."}

        prompt = """
You are AgriVision, an expert plant pathologist.
You will receive multiple frames sampled from a crop video.
Infer the plant and any disease/pest/deficiency present across frames.
Return ONLY one raw JSON object in this schema:
{
  "plant": "str",
  "condition": "str",
  "is_healthy": bool,
  "confidence": float (0-1),
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

        # Convert frames to PIL and send multiple images + prompt
        contents = []
        for fr in frames:
            rgb = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
            contents.append(Image.fromarray(rgb))
        contents.append(prompt)

        resp = client.models.generate_content(
            model="gemini-3-pro-preview",
            contents=contents,
            config=cfg,
        )
        return json.loads(resp.text)

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
        st.warning(f"Low confidence ({conf:.2f}). Try closer framing + better lighting.")

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
    """
    IMPORTANT: Do NOT touch st.session_state inside recv() (runs in worker thread).
    Store frame into a thread-safe place.
    """
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        with st.session_state.frame_lock:
            st.session_state.latest_frame_bgr = img
        return frame


# --- WebRTC ICE configuration ---
# STUN-only works on many networks but may fail on restrictive Wi-Fi.
# For maximum reliability (Wi-Fi included), add a TURN server.
RTC_CONFIGURATION = RTCConfiguration(
    {
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]},
            # Example TURN (replace with your own):
            # {"urls": ["turn:YOUR_TURN_HOST:3478"], "username": "USER", "credential": "PASS"},
        ]
    }
)

# --- Sidebar ---
with st.sidebar:
    st.title("AgriVision")
    st.write("Real-time AI Crop Analyzer for Better Crop Production")

    st.session_state.api_key = st.text_input(
        "Enter Google API Key",
        type="password",
        value=st.session_state.api_key
    )

    st.divider()
    st.write("### Settings")
    analysis_interval = st.slider("Analysis Interval (seconds)", 1, 10, 3)
    confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.6)

    st.caption("If live camera fails on Wi-Fi, your network likely blocks WebRTC UDP. Add TURN to fix.")


# --- Main App ---
tab1, tab2 = st.tabs(["Live Stream", "Upload Image/Video"])

with tab1:
    col_video, col_info = st.columns([2, 1])

    with col_video:
        st.subheader("Live Feed (WebRTC)")
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

    # "Tick" loop: schedule analysis every interval, update UI when future completes
    if ctx.state.playing:
        now = time.time()

        # collect result from background thread
        if st.session_state.future and st.session_state.future.done():
            st.session_state.analysis_result = st.session_state.future.result()
            st.session_state.future = None

        # schedule next analysis
        with st.session_state.frame_lock:
            frame_available = st.session_state.latest_frame_bgr is not None

        if (
            st.session_state.future is None
            and frame_available
            and (now - st.session_state.last_analysis_time) >= analysis_interval
        ):
            with st.session_state.frame_lock:
                frame_copy = st.session_state.latest_frame_bgr.copy()

            st.session_state.future = st.session_state.thread_pool.submit(
                analyze_image,
                st.session_state.api_key,
                frame_copy,
            )
            st.session_state.last_analysis_time = now

        with st_report.container():
            display_report(st.session_state.analysis_result, confidence_threshold)

        # Optional: force periodic reruns so the report updates smoothly.
        # Comment this out if you don't want auto refresh.
        time.sleep(0.2)
        st.rerun()

    else:
        with st_report.container():
            st.info("Click **Start** above and allow camera permissions in your browser.")


with tab2:
    st.header("Upload Analysis")
    uploaded_file = st.file_uploader("Choose an image or video (any extension)", type=None)

    if uploaded_file:
        # 1) Try to decode as image first (robust, ignores filename/mime)
        try:
            img = Image.open(uploaded_file)
            img = img.convert("RGB")
            st.image(img, caption="Uploaded file (decoded as image)", width=400)

            if st.button("Analyze (Image)"):
                with st.spinner("Analyzing image..."):
                    res = analyze_image(st.session_state.api_key, img)
                    display_report(res, confidence_threshold)

        except UnidentifiedImageError:
            # 2) If not an image, treat as video: save temp and open with OpenCV
            st.info("File is not an image. Treating it as a video...")
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
                tmp.write(uploaded_file.getbuffer())
                video_path = tmp.name

            st.video(video_path)

            if st.button("Analyze (Video)"):
                with st.spinner("Sampling frames and analyzing video..."):
                    res = analyze_video_file(st.session_state.api_key, video_path, max_frames=6)
                    display_report(res, confidence_threshold)

            # cleanup is optional; Streamlit Cloud ephemeral FS anyway
            # os.remove(video_path)
