import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io

# --- Page Configuration ---
st.set_page_config(page_title="DeepSense AI Lab", layout="wide")

# --- Custom Styling ---
st.markdown("""
    <style>
    .hero-text { font-size: 50px !important; font-weight: 700; color: #00d4ff; text-align: center; }
    .sub-text { font-size: 20px !important; text-align: center; color: #888; margin-bottom: 40px; }
    .feature-card { background-color: #1e2130; padding: 20px; border-radius: 15px; border: 1px solid #333; text-align: center; height: 100%; }
    .stDownloadButton>button { width: 100% !important; background-color: #28a745 !important; color: white !important; border-radius: 8px !important; height: 3em !important; font-weight: bold !important; }
    .stDownloadButton>button:hover { background-color: #218838 !important; border-color: #1e7e34 !important; }
    </style>
    """, unsafe_allow_html=True)

# --- AI Model Placeholder ---
def run_ai_inference(img_np):
    # REPLACE THIS: Input your trained model inference here
    # Placeholder: Using OpenCV Detail Enhance to simulate AI
    enhanced = cv2.detailEnhance(img_np, sigma_s=10, sigma_r=0.15)
    return enhanced

# --- Sidebar ---
st.sidebar.header("📥 Upload Center")
uploaded_file = st.sidebar.file_uploader("Drop your photo here", type=["jpg", "jpeg", "png"])

if uploaded_file is None:
    # === LANDING PAGE ===
    st.markdown('<p class="hero-text">Next-Gen AI Image Restoration</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-text">Upload a photo to see our custom-trained Neural Network in action.</p>', unsafe_allow_html=True)
    
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown('<div class="feature-card"><h3>🔍 Neural Upscale</h3><p>Smartly fill in missing pixels using deep learning.</p></div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="feature-card"><h3>💡 Low-Light AI</h3><p>Advanced exposure recovery for night-time photography.</p></div>', unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="feature-card"><h3>🛡️ Artifact Removal</h3><p>Clear out compression noise and digital grain.</p></div>', unsafe_allow_html=True)
    st.info("👈 Please upload an image from the sidebar to begin.")

else:
    # === PROCESSING & DISPLAY ===
    # Convert uploaded file to OpenCV format
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, 1)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    with st.spinner("🧠 AI Model is processing..."):
        ai_output_rgb = run_ai_inference(img_rgb)

    # UI Layout: Side-by-Side
    st.markdown("### ⚡ AI Enhancement Result")
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.markdown("#### 📸 ORIGINAL")
        st.image(img_rgb, use_container_width=True)
        st.caption("Input Image Details preserved")

    with col_right:
        st.markdown("#### ✨ AI ENHANCED")
        st.image(ai_output_rgb, use_container_width=True)
        
        # --- DOWNLOAD BUTTON DIRECTLY BELOW IMAGE ---
        # Convert RGB back to BGR for encoding
        output_bgr = cv2.cvtColor(ai_output_rgb, cv2.COLOR_RGB2BGR)
        _, buffer = cv2.imencode('.jpg', output_bgr)
        
        st.download_button(
            label="📩 DOWNLOAD ENHANCED IMAGE",
            data=buffer.tobytes(),
            file_name="ai_enhanced_result.jpg",
            mime="image/jpeg"
        )

    st.sidebar.success("Process Complete!")
    if st.sidebar.button("Clear & Upload New"):
        st.rerun()

st.divider()
st.caption("AI Model Lab | Optimized for Realme 3 Pro & Low-Light Sensors")
