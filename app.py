import streamlit as st
import cv2
import numpy as np
import time
from utils import load_weights
from Enhancer import Enhancer

@st.cache_resource
def load_enhancer():
    enhancer = Enhancer(load_weights(), batch_size=4)
    return enhancer

enhancer = load_enhancer()

# --- Page Configuration ---
st.set_page_config(page_title="DeepSense AI Lab", layout="wide")

# --- Custom Styling ---
st.markdown("""
    <style>
    @media (min-width: 800px) { .mobile-only { display: none !important; } }
    @media (max-width: 799px) { .desktop-only { display: none !important; } }
    .stImage > img { max-width: 100%; height: auto; object-fit: contain; border-radius: 12px; border: 1px solid #333; }
    .hero-text { font-size: clamp(26px, 5vw, 42px); font-weight: 800; color: #00d4ff; text-align: center; }
    .upload-card { background: #1e2130; padding: 25px; border-radius: 20px; border: 2px dashed #444; text-align: center; }
    div.stDownloadButton > button { background-color: #00c853 !important; color: white !important; width: 100% !important; border-radius: 10px !important; height: 3.5em !important; font-weight: bold !important; }
    .stButton > button { width: 100% !important; border-radius: 10px !important; height: 3.5em !important; }
    </style>
    """, unsafe_allow_html=True)

# --- Session State for Reset ---
if 'uploader_key' not in st.session_state:
    st.session_state.uploader_key = 0

def reset_app():
    st.session_state.uploader_key += 1
    st.rerun()

# --- AI Core ---
def run_ai_inference(img_np):
    start = time.time()
    # Replace this with your actual model: enhanced = model(img_np)
    enhanced =  enhancer.enhance_image(img_np)
    return enhanced, round((time.time() - start), 3)

# --- UI LOGIC ---
active_file = None

# Sidebar (Desktop)
with st.sidebar:
    st.markdown("### 🖥️ Desktop Upload")
    file_desktop = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"], 
                                    key=f"d_up_{st.session_state.uploader_key}")

# Main Hero UI
if file_desktop is None:
    st.markdown('<p class="hero-text">AI Image Restoration Lab</p>', unsafe_allow_html=True)
    st.markdown('<div class="mobile-only"><div class="upload-card">', unsafe_allow_html=True)
    file_mobile = st.file_uploader("📱 Tap to Select Photo", type=["jpg", "jpeg", "png"], 
                                   key=f"m_up_{st.session_state.uploader_key}")
    st.markdown('</div></div>', unsafe_allow_html=True)
    active_file = file_mobile
else:
    active_file = file_desktop

# --- PROCESS & DISPLAY ---
if active_file:
    # Read Image
    file_bytes = np.asarray(bytearray(active_file.read()), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, 1)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    with st.spinner("🤖 AI Model is analyzing..."):
        ai_output, p_time = enhancer.enhance_image(img_rgb)

    st.markdown(f"### ✨ Result Ready (`{p_time}s`)")

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("#### 📸 ORIGINAL")
        st.image(img_rgb, use_container_width=True)

    with col_b:
        st.markdown("#### 🚀 AI ENHANCED")
        st.image(ai_output, use_container_width=True)
        
        # --- FIXED DOWNLOAD LOGIC ---
        # Sirf tabhi encode karega jab ai_output mil chuka ho
        try:
            output_bgr_final = cv2.cvtColor(ai_output, cv2.COLOR_RGB2BGR)
            is_success, buffer = cv2.imencode(".jpg", output_bgr_final)
            if is_success:
                st.download_button(
                    label="📩 DOWNLOAD IMAGE",
                    data=buffer.tobytes(),
                    file_name="enhanced_output.jpg",
                    mime="image/jpeg"
                )
        except Exception as e:
            st.error("Download button generate karne mein dikkat aayi.")

        # Upload Another Button
        if st.button("➕ UPLOAD ANOTHER"):
            reset_app()
else:
    if file_desktop is None:
        st.markdown('<p style="text-align:center; color:#888;">Professional Neural Enhancement tuned for your device.</p>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1: st.info("🔍 **High Res**")
        with c2: st.info("💡 **Bright AI**")
        with c3: st.info("🛡️ **Denoise**")

st.divider()
st.caption("DeepSense AI | Stable Build v8.0")
