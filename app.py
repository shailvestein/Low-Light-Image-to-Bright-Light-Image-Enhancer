import streamlit as st
import cv2
import numpy as np
from streamlit_image_comparison import image_comparison
import time

# --- Page Configuration ---
st.set_page_config(page_title="DeepSense AI Lab", layout="wide")

# --- Adaptive CSS ---
st.markdown("""
    <style>
    @media (min-width: 800px) { .mobile-only { display: none !important; } }
    @media (max-width: 799px) { .desktop-only { display: none !important; } }
    
    .hero-text { font-size: clamp(28px, 6vw, 48px); font-weight: 800; color: #00d4ff; text-align: center; }
    .upload-card { background: #1e2130; padding: 30px; border-radius: 20px; border: 2px dashed #00d4ff; text-align: center; }
    
    /* Full-width Buttons */
    .stDownloadButton > button, .stButton > button {
        width: 100% !important; border-radius: 12px !important; height: 3.5em !important; font-weight: bold !important;
    }
    </style>
    """, unsafe_allow_html=True)

# --- AI Logic ---
def run_ai_inference(img_np):
    start = time.time()
    # Placeholder: Replace with your PyTorch/Model logic
    enhanced = cv2.detailEnhance(img_np, sigma_s=12, sigma_r=0.15)
    return enhanced, round((time.time() - start), 3)

# --- Sidebar (Desktop Only) ---
with st.sidebar:
    st.markdown("### 🖥️ Desktop Upload")
    file_desktop = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"], key="d_up")

# --- Main App Logic ---
active_file = file_desktop

if active_file is None:
    st.markdown('<p class="hero-text">AI Image Restoration Lab</p>', unsafe_allow_html=True)
    
    # Mobile Upload Card (Home Screen)
    st.markdown('<div class="mobile-only"><div class="upload-card">', unsafe_allow_html=True)
    active_file = st.file_uploader("📱 Tap to Enhance Photo", type=["jpg", "jpeg", "png"], key="m_up")
    st.markdown('</div></div>', unsafe_allow_html=True)

    if active_file is None:
        st.markdown('<p style="text-align:center; color:#888;">Experience high-definition AI restoration instantly.</p>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1: st.info("🔍 **Super-Res**")
        with c2: st.info("💡 **Low-Light**")
        with c3: st.info("🛡️ **Denoise**")

if active_file:
    # --- PROCESSING ---
    file_bytes = np.asarray(bytearray(active_file.read()), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, 1)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    with st.spinner("🧠 AI Model is analyzing pixels..."):
        ai_output, p_time = run_ai_inference(img_rgb)

    st.markdown(f"### ✨ AI Result Ready (`{p_time}s`)")
    st.write("Drag the slider to compare details. Click image to expand.")

    # --- INTERACTIVE COMPARISON SLIDER ---
    # Ye mobile aur desktop dono par swipe/drag support karta hai
    image_comparison(
        img1=img_rgb,
        img2=ai_output,
        label1="Original",
        label2="AI Enhanced",
        width=1100, # Desktop width (responsive automatically)
        starting_position=50,
        show_labels=True,
        make_responsive=True,
        in_memory=True
    )

    # --- ACTION BUTTONS ---
    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        # Download
        out_bgr = cv2.cvtColor(ai_output, cv2.COLOR_RGB2BGR)
        _, buf = cv2.imencode('.jpg', out_bgr)
        st.download_button("📩 DOWNLOAD IMAGE", buf.tobytes(), "ai_result.jpg", "image/jpeg")
    
    with col_btn2:
        if st.button("➕ ENHANCE ANOTHER"):
            st.rerun()

st.divider()
st.caption("DeepSense AI Lab | Adaptive UI v5.0")
