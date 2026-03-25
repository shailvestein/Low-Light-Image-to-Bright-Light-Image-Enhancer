import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import time

# --- Page Configuration ---
st.set_page_config(page_title="DeepSense AI Lab", layout="wide")

# --- Custom Styling ---
st.markdown("""
    <style>
    .hero-text { font-size: clamp(24px, 5vw, 48px) !important; font-weight: 800; color: #00d4ff; text-align: center; }
    .sub-text { font-size: clamp(14px, 2vw, 18px) !important; text-align: center; color: #888; margin-bottom: 30px; }
    .feature-card { background-color: #1e2130; padding: 20px; border-radius: 15px; border: 1px solid #333; text-align: center; height: 100%; }
    
    /* Responsive Download Button */
    div.stDownloadButton > button {
        width: 100% !important;
        background-color: #00c853 !important;
        color: white !important;
        border-radius: 10px !important;
        font-weight: bold !important;
        height: 3.5em !important;
    }
    
    /* Navigate / Reset Button Styling */
    .stButton > button {
        width: 100% !important;
        border-radius: 10px !important;
        height: 3.5em !important;
        margin-top: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- AI Logic Placeholder ---
def run_ai_inference(img_np):
    start = time.time()
    # Replace with your: output = model(input)
    enhanced = cv2.detailEnhance(img_np, sigma_s=12, sigma_r=0.15)
    return enhanced, round((time.time() - start), 3)

# --- Session State to handle Navigation ---
if 'uploaded' not in st.session_state:
    st.session_state.uploaded = False

# --- Sidebar ---
st.sidebar.header("📥 Control Panel")
uploaded_file = st.sidebar.file_uploader("Upload Image", type=["jpg", "jpeg", "png"], key="file_uploader")

if uploaded_file is None:
    # === LANDING PAGE ===
    st.markdown('<p class="hero-text">AI Image Restoration Lab</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-text">Upload your photos to see the power of Neural Enhancement.</p>', unsafe_allow_html=True)
    
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown('<div class="feature-card"><h3>🔍 HD Upscale</h3><p>Neural reconstruction of low-res textures.</p></div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="feature-card"><h3>💡 Night Vision</h3><p>Deep-learning based exposure recovery.</p></div>', unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="feature-card"><h3>🛡️ Zero Noise</h3><p>Remove digital grain while keeping edges sharp.</p></div>', unsafe_allow_html=True)
    
    st.divider()
    st.info("👈 Please select an image from the sidebar to begin processing.")

else:
    # === OUTPUT & NAVIGATION MODE ===
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, 1)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    with st.spinner("🧠 AI Model is processing..."):
        ai_output, p_time = run_ai_inference(img_rgb)

    st.markdown(f"### ✨ AI Result (Processed in `{p_time}s`)")
    
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.markdown("#### 📸 INPUT")
        st.image(img_rgb, use_container_width=True)

    with col_b:
        st.markdown("#### 🚀 AI OUTPUT")
        st.image(ai_output, use_container_width=True)
        
        # --- BUTTONS GROUP ---
        # 1. Download Button
        output_bgr = cv2.cvtColor(ai_output, cv2.COLOR_RGB2BGR)
        _, buffer = cv2.imencode('.jpg', output_bgr)
        
        st.download_button(
            label="📩 DOWNLOAD ENHANCED IMAGE",
            data=buffer.tobytes(),
            file_name="ai_result.jpg",
            mime="image/jpeg"
        )
        
        # 2. Navigate Back Button
        if st.button("➕ UPLOAD ANOTHER IMAGE"):
            # This triggers a rerun and since the key is fixed, clearing session works
            st.rerun()

# --- Footer ---
st.divider()
st.caption("DeepSense AI | Responsive Dashboard v3.0")
