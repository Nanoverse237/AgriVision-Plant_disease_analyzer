import streamlit as st
import cv2
import time
import os
import json
from PIL import Image
import numpy as np
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

# --- Side Bar ---
with st.sidebar:
    st.title("AgriVision")
    st.write("Real-time AI Crop Analyzer for Better Crop Production")
    
    if not api_key:
        api_key = st.text_input("Enter Google API Key", type="password")
        
    st.divider()
    st.write("### Settings")
    analysis_interval = st.slider("Analysis Interval (seconds)", 1, 10, 3)
    confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.6)

# --- Helper Functions ---

def get_gemini_client():
    if not api_key:
        return None
    return genai.Client(api_key=api_key)

def analyze_image(client, image):
    """
    Sends an image (PIL or numpy array) to Gemini for disease analysis.
    """
    try:
        # Convert numpy array (OpenCV) to PIL
        if isinstance(image, np.ndarray):
            # RGB conversion
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

        response = client.models.generate_content(
            model='gemini-3-pro-preview', 
            contents=[image, prompt],
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
        return json.loads(response.text)
    except Exception as e:
        return {"error": str(e)}

def display_report(result):
    if "error" in result:
        st.error(f"Analysis Failed: {result['error']}")
        return

    color = "green" if result.get('is_healthy') else "red"
    
    st.markdown(f"""
    <div class="report-box">
        <div class="disease-title" style="color: {color}">
            {result.get('condition', 'Unknown Condition')}
        </div>
        <p><b>Plant:</b> {result.get('plant', 'Unknown')} | 
           <b>Confidence:</b> {result.get('confidence', 0):.2f}</p>
        <p>{result.get('description', '')}</p>
    </div>
    """, unsafe_allow_html=True)

    if not result.get('is_healthy') and result.get('treatments'):
        st.subheader("Recommended Treatments")
        for treatment in result['treatments']:
            st.info(treatment)
# --- Main UI ---

if not api_key:
    st.warning("Please enter your Google API Key in the sidebar to continue.")
    st.stop()

client = get_gemini_client()

tab1, tab2 = st.tabs(["Live Stream Analysis", "Upload Image/Video"])

with tab1:
    st.header("Live Video Analysis")
    st.write("This mode uses your local webcam to analyze crops in real-time.")
    
    run_camera = st.checkbox("Start Camera Stream")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st_frame = st.empty()
        
    with col2:
        st_report = st.empty()

    if run_camera:
        # NOTE: cv2.VideoCapture(0) works on LOCAL machines (Jupyter/Localhost).
        # It may not work on cloud hosting (Streamlit Cloud) without webrtc.
        cap = cv2.VideoCapture(0)
        last_analysis_time = 0
        
        while run_camera:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to capture video frame.")
                break
            
            # Display frame
            st_frame.image(frame, channels="BGR", use_container_width=True)
            
            # Analyze every N seconds
            current_time = time.time()
            if current_time - last_analysis_time > analysis_interval:
                with st_report.container():
                    with st.spinner("Analyzing frame..."):
                        result = analyze_image(client, frame)
                        display_report(result)
                last_analysis_time = current_time
            
            # Small sleep to reduce CPU usage
            time.sleep(0.1)
            
        cap.release()

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
                    result = analyze_image(client, image)
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
                            model='gemini-2.0-flash-exp',
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
