import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io

# --- Page Configuration ---
st.set_page_config(page_title="Neural Image Enhancer", layout="wide")

# Custom UI Styling
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stHeader { color: #00d4ff; }
    .css-10trblm { color: #00d4ff; } /* Subheader color */
    </style>
    """, unsafe_allow_html=True)

# --- AI Model Placeholder ---
# Load your model globally so it stays in memory
@st.cache_resource
def load_trained_model():
    # model = torch.load('your_model.pth')
    # model.eval()
    return None 

model = load_trained_model()

def run_ai_inference(img_np, model):
    """
    img_np: Input image as a NumPy array (RGB)
    This is where your trained AI logic goes.
    """
    # 1. Pre-processing (Example: Normalization for AI)
    # img_input = img_np.astype(np.float32) / 255.0
    
    # 2. Model Prediction (Placeholder Logic)
    # ---------------------------------------------------------
    # REPLACE THIS with your: output = model(img_input)
    # For now, we simulate an AI 'Pop' effect using OpenCV
    enhanced = cv2.detailEnhance(img_np, sigma_s=10, sigma_r=0.15)
    # ---------------------------------------------------------
    
    return enhanced

# --- Main UI ---
st.title("🤖 DeepSense AI Enhancer")
st.write("Upload your photo to trigger the trained Neural Network pipeline.")

# Sidebar for Upload
st.sidebar.header("📥 Input Panel")
uploaded_file = st.sidebar.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # --- Step 1: Read Image with OpenCV ---
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    # OpenCV reads BGR by default, we convert to RGB for Streamlit/AI
    img_bgr = cv2.imdecode(file_bytes, 1)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # --- Step 2: Run AI Inference ---
    with st.spinner("🧠 AI Model is analyzing pixels..."):
        ai_output_rgb = run_ai_inference(img_rgb, model)

    # --- Step 3: Display Results Side-by-Side ---
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Original Image")
        st.image(img_rgb, use_container_width=True)
        
    with col2:
        st.subheader("AI Enhanced Result")
        st.image(ai_output_rgb, use_container_width=True)

    # --- Step 4: Download Results ---
    st.sidebar.markdown("---")
    st.sidebar.header("📩 Export")
    
    # Convert back to BGR for encoding to JPG
    output_bgr = cv2.cvtColor(ai_output_rgb, cv2.COLOR_RGB2BGR)
    _, buffer = cv2.imencode('.jpg', output_bgr)
    
    st.sidebar.download_button(
        label="Download AI Enhanced Image",
        data=buffer.tobytes(),
        file_name="ai_enhanced.jpg",
        mime="image/jpeg"
    )

else:
    st.info("Waiting for image upload...")
    # Empty Placeholders for look and feel
    c1, c2 = st.columns(2)
    c1.image("https://via.placeholder.com/500x400?text=Awaiting+Input", use_container_width=True)
    c2.image("https://via.placeholder.com/500x400?text=AI+Result", use_container_width=True)

st.divider()
st.caption("Powered by OpenCV and Custom Neural Network | Akash AI Projects")
