import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io

# --- Page Configuration ---
st.set_page_config(page_title="DeepSense AI Lab", layout="wide", initial_sidebar_state="expanded")

# --- Custom Styling ---
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .hero-text {
        font-size: 50px !important;
        font-weight: 700;
        color: #00d4ff;
        text-align: center;
        margin-bottom: 10px;
    }
    .sub-text {
        font-size: 20px !important;
        text-align: center;
        color: #888;
        margin-bottom: 40px;
    }
    .feature-card {
        background-color: #1e2130;
        padding: 20px;
        border-radius: 15px;
        border: 1px solid #333;
        text-align: center;
    }
    .stImage { border-radius: 10px; border: 2px solid #333; }
    </style>
    """, unsafe_allow_html=True)

# --- AI Model Placeholder ---
@st.cache_resource
def load_trained_model():
    # model = torch.load('your_model.pth')
    return None 

def run_ai_inference(img_np, model):
    """
    Yahan aapka trained AI model chalega.
    Abhi ke liye OpenCV ka filter use ho raha hai as a placeholder.
    """
    # Placeholder Logic: AI-like detail enhancement
    enhanced = cv2.detailEnhance(img_np, sigma_s=10, sigma_r=0.15)
    return enhanced

# --- Sidebar ---
st.sidebar.header("📥 Upload Center")
uploaded_file = st.sidebar.file_uploader("Drop your photo here", type=["jpg", "jpeg", "png"])

# --- Main App Logic ---
if uploaded_file is None:
    # === HERO SECTION (When no image is uploaded) ===
    st.markdown('<p class="hero-text">Transform Your Images with AI</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-text">Experience the power of custom-trained Neural Networks for instant restoration.</p>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('<div class="feature-card"><h3>🔍 Super Resolution</h3><p>Upscale low-res images without losing texture or details.</p></div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="feature-card"><h3>💡 Low-Light Boost</h3><p>Proprietary AI architecture designed to bring out colors from the shadows.</p></div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="feature-card"><h3>🛡️ Deep Denoise</h3><p>Smart neural filtering to eliminate sensor noise and grain.</p></div>', unsafe_allow_html=True)
    
    st.markdown("---")
    st.info("👈 Get started by uploading an image from the sidebar!")

else:
    # === COMPARISON MODE (When image is uploaded) ===
    st.markdown("### ⚡ AI Processing Result")
    
    # 1. Load and Process
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, 1)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    with st.spinner("🧠 Analyzing pixels with Neural Engine..."):
        ai_output = run_ai_inference(img_rgb, load_trained_model())

    # 2. Side-by-Side Display
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.markdown("#### 📸 ORIGINAL INPUT")
        st.image(img_rgb, use_container_width=True)
        
    with col_b:
        st.markdown("#### ✨ AI ENHANCED")
        st.image(ai_output, use_container_width=True)

    # 3. Export Options
    st.sidebar.markdown("---")
    st.sidebar.success("Enhancement Complete!")
    
    output_bgr = cv2.cvtColor(ai_output, cv2.COLOR_RGB2BGR)
    _, buffer = cv2.imencode('.jpg', output_bgr)
    
    st.sidebar.download_button(
        label="Download AI Enhanced Photo",
        data=buffer.tobytes(),
        file_name="ai_result.jpg",
        mime="image/jpeg"
    )
    
    if st.button("⬅️ Upload Another Image"):
        st.rerun()

# --- Footer ---
st.divider()
st.caption("DeepSense AI Lab | Built with Python, OpenCV & Streamlit")
