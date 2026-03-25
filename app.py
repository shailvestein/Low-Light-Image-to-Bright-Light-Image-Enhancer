import streamlit as st
import cv2
import numpy as np
import io
import time
from PIL import Image

# --- Page Configuration ---
st.set_page_config(page_title="DeepSense AI Lab", layout="wide")

# --- Optimized Responsive CSS ---
st.markdown("""
    <style>
    /* Desktop vs Mobile Logic */
    @media (min-width: 800px) { .mobile-only { display: none !important; } }
    @media (max-width: 799px) { .desktop-only { display: none !important; } }

    /* Maintain Aspect Ratio and prevent stretching */
    .stImage > img {
        max-width: 100%;
        height: auto;
        object-fit: contain; /* Isse image stretch nahi hogi */
        border-radius: 12px;
        border: 1px solid #333;
    }

    .hero-text { font-size: clamp(26px, 5vw, 42px); font-weight: 800; color: #00d4ff; text-align: center; }
    .upload-card { background: #1e2130; padding: 25px; border-radius: 20px; border: 2px dashed #444; text-align: center; }
    
    /* Responsive Buttons */
    .stDownloadButton > button, .stButton > button {
        width: 100% !important;
        border-radius: 10px !important;
        height: 3.2em !important;
        font-weight: 600 !important;
    }
    </style>
    """, unsafe_allow_html=True)

# --- AI Core ---
def run_ai_inference(img_np):
    start = time.time()
    # Dummy AI Logic: Real AI model inference goes here
    enhanced = cv2.detailEnhance(img_np, sigma_s=10, sigma_r=0.15)
    return enhanced, round((time.time() - start), 3)

# --- Sidebar (Desktop) ---
with st.sidebar:
    st.markdown("### 🖥️ Image Upload")
    file_desktop = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"], key="d_up")

# --- Main Logic ---
active_file = file_desktop

if active_file is None:
    st.markdown('<p class="hero-text">AI Image Restoration Lab</p>', unsafe_allow_html=True)
    
    # Mobile Specific Upload (Only shows on small screens)
    st.markdown('<div class="mobile-only"><div class="upload-card">', unsafe_allow_html=True)
    active_file = st.file_uploader("📱 Tap to Select Photo", type=["jpg", "jpeg", "png"], key="m_up")
    st.markdown('</div></div>', unsafe_allow_html=True)

    if active_file is None:
        st.markdown('<p style="text-align:center; color:#888;">Professional Neural Enhancement in one click.</p>', unsafe_allow_html=True)
        # Feature Highlights
        c1, c2, c3 = st.columns(3)
        with c1: st.info("🔍 **High Res**")
        with c2: st.info("💡 **Bright AI**")
        with c3: st.info("🛡️ **Denoise**")

if active_file:
    # --- PROCESSING ---
    # File reading using OpenCV for performance
    file_bytes = np.asarray(bytearray(active_file.read()), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, 1)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    with st.spinner("🤖 Neural Engine is active..."):
        ai_output, p_time = run_ai_inference(img_rgb)

    st.markdown(f"### ✨ Processing Finished in `{p_time}s`")

    # --- RESPONSIVE DISPLAY ---
    # Desktop: Side-by-Side (50/50), Mobile: Vertical Stack (100%)
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.markdown("#### 📸 ORIGINAL")
        # use_container_width=True aspect ratio ko auto-adjust karta hai screen ke according
        st.image(img_rgb, use_container_width=True)

    with col_b:
        st.markdown("#### 🚀 AI ENHANCED")
        st.image(ai_output, use_container_width=True)
        
        # Download Logic
        out_bgr = cv2.cvtColor(ai_output, cv2.COLOR_RGB2BGR)
        _, buf = cv2.imencode('.jpg', out_bgr)
        st.download_button("📩 DOWNLOAD ENHANCED IMAGE", buf.tobytes(), "enhanced.jpg", "image/jpeg")
        
        if st.button("➕ UPLOAD ANOTHER"):
            st.rerun()

st.divider()
st.caption("DeepSense AI | Optimized for Aspect-Ratio & Mobile v6.0")
