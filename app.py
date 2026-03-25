import streamlit as st
import cv2
import numpy as np
import io
import time

# --- Page Configuration ---
st.set_page_config(page_title="DeepSense AI Lab", layout="wide")

# --- Responsive CSS Magic ---
st.markdown("""
    <style>
    /* Desktop vs Mobile Logic */
    @media (min-width: 800px) {
        .mobile-only-upload { display: none !important; }
    }
    
    @media (max-width: 799px) {
        /* Sidebar ko mobile par hide karne ke liye (Optional) */
        [data-testid="stSidebarNav"] { display: none; }
        .desktop-only-info { display: none !important; }
    }

    /* Styling for Hero Upload Card */
    .upload-card {
        background-color: #1e2130;
        padding: 30px;
        border-radius: 20px;
        border: 2px dashed #00d4ff;
        text-align: center;
        margin-bottom: 20px;
    }
    
    .hero-text { font-size: clamp(28px, 6vw, 48px); font-weight: 800; color: #00d4ff; text-align: center; }
    
    /* Buttons Styling */
    .stDownloadButton > button, .stButton > button {
        width: 100% !important;
        border-radius: 12px !important;
        height: 3.5em !important;
        font-weight: bold !important;
    }
    </style>
    """, unsafe_allow_html=True)

# --- AI Logic Placeholder ---
def run_ai_inference(img_np):
    start = time.time()
    # Placeholder for your model
    enhanced = cv2.detailEnhance(img_np, sigma_s=12, sigma_r=0.15)
    return enhanced, round((time.time() - start), 3)

# --- App Logic ---

# 1. Sidebar Upload (For Desktop)
with st.sidebar:
    st.markdown("### 🖥️ Desktop Upload")
    file_desktop = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"], key="desktop_up")

# 2. Main Page Header
if file_desktop is None:
    st.markdown('<p class="hero-text">AI Image Restoration Lab</p>', unsafe_allow_html=True)
    
    # Main Page Upload (For Mobile - Wrapped in a CSS Class)
    st.markdown('<div class="mobile-only-upload">', unsafe_allow_html=True)
    st.markdown('<div class="upload-card">', unsafe_allow_html=True)
    st.write("### 📱 Mobile Upload")
    file_mobile = st.file_uploader("Tap to select image", type=["jpg", "jpeg", "png"], key="mobile_up")
    st.markdown('</div></div>', unsafe_allow_html=True)

    # Feature Cards (Always visible as Hero content)
    st.markdown('<p style="text-align:center; color:#888;">Professional Neural Enhancement at your fingertips.</p>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1: st.info("🔍 **HD Upscale**")
    with c2: st.info("💡 **Low-Light AI**")
    with c3: st.info("🛡️ **Zero Noise**")
else:
    file_mobile = None # Desktop priority

# Determine which file to process
active_file = file_desktop if file_desktop else file_mobile

if active_file:
    # --- PROCESSING MODE ---
    file_bytes = np.asarray(bytearray(active_file.read()), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, 1)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    with st.spinner("🧠 AI Model is processing..."):
        ai_output, p_time = run_ai_inference(img_rgb)

    st.markdown(f"### ✨ Result Ready (`{p_time}s`)")
    
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("#### 📸 INPUT")
        st.image(img_rgb, use_container_width=True)

    with col_b:
        st.markdown("#### 🚀 AI OUTPUT")
        st.image(ai_output, use_container_width=True)
        
        # Download
        output_bgr = cv2.cvtColor(ai_output, cv2.COLOR_RGB2BGR)
        _, buffer = cv2.imencode('.jpg', output_bgr)
        st.download_button("📩 DOWNLOAD IMAGE", buffer.tobytes(), "ai_result.jpg", "image/jpeg")
        
        # Clear/Navigate Back
        if st.button("➕ UPLOAD ANOTHER"):
            st.rerun()

st.divider()
st.caption("DeepSense AI | Responsive Adaptive Dashboard v4.0")
