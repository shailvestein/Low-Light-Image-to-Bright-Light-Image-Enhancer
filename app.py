import streamlit as st
import cv2
import numpy as np
import time
import io
import os
from PIL import Image
from utils import load_weights
from Enhancer import Enhancer

# --- 1. SET PAGE CONFIG (Must be first Streamlit command) ---
st.set_page_config(page_title="DeepSense AI Restoration Lab", layout="wide", page_icon="🚀")

# --- 2. HELPER FUNCTIONS ---
@st.cache_resource
def load_ai_model():
    """Weights load karke Enhancer object return karta hai aur cache karta hai"""
    try:
        model = load_weights()
        # Batch size 1 or 2 is stable for Streamlit Cloud
        return Enhancer(model, batch_size=2)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def get_webp_bytes(image_rgb, quality=85):
    """Numpy RGB array ko optimized WebP bytes mein badalta hai"""
    img = Image.fromarray(image_rgb)
    buf = io.BytesIO()
    img.save(buf, format='WEBP', quality=quality, method=6)
    return buf.getvalue()

def pre_process_resize(image_rgb, target_width=1200):
    """4K images ko 2K/HD mein resize karta hai prediction se pehle"""
    h, w = image_rgb.shape[:2]
    if w <= target_width:
        return image_rgb
    aspect_ratio = h / w
    target_height = int(target_width * aspect_ratio)
    # INTER_AREA is best for downscaling
    return cv2.resize(image_rgb, (target_width, target_height), interpolation=cv2.INTER_AREA)

# --- 3. CUSTOM CSS STYLING ---
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stImage > img { border-radius: 15px; border: 1px solid #333; transition: 0.3s; }
    .stImage > img:hover { border-color: #00d4ff; }
    .hero-text { font-size: 42px; font-weight: 800; color: #00d4ff; text-align: center; margin-bottom: 10px; }
    .sub-text { text-align: center; color: #888; margin-bottom: 30px; }
    div.stDownloadButton > button { 
        background-color: #00c853 !important; 
        color: white !important; 
        width: 100% !important; 
        border-radius: 12px !important; 
        font-weight: bold !important;
        border: none !important;
        height: 3em !important;
    }
    .stButton > button { width: 100% !important; border-radius: 12px !important; }
    </style>
    """, unsafe_allow_html=True)

# --- 4. INITIALIZATION ---
if 'uploader_key' not in st.session_state:
    st.session_state.uploader_key = 0

def reset_app():
    st.session_state.uploader_key += 1
    st.rerun()

# Load Enhancer
enhancer = load_ai_model()

# --- 5. UI LOGIC (Header & Sidebar) ---
st.markdown('<p class="hero-text">DeepSense AI Lab</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-text">Neural Low-Light Image Restoration & Enhancement</p>', unsafe_allow_html=True)

active_file = None

with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2103/2103633.png", width=80)
    st.title("Control Panel")
    file_upload = st.file_uploader("Upload Dark Photo", type=["jpg", "jpeg", "png"], 
                                   key=f"up_{st.session_state.uploader_key}")
    st.divider()
    st.info("Tip: 4K images are auto-resized to 2K for faster AI processing.")

if file_upload:
    active_file = file_upload
else:
    # Landing Page Cards
    col_info1, col_info2, col_info3 = st.columns(3)
    with col_info1: st.help("🔍 **Low-Light Recovery**: Restores details from shadows.")
    with col_info2: st.help("💡 **Adaptive Brightness**: Smart exposure without over-blowing.")
    with col_info3: st.help("🛡️ **WebP Optimized**: High quality, ultra-small file size.")

# --- 6. PROCESS & DISPLAY ---
if active_file and enhancer:
    # Read and Convert
    file_bytes = np.asarray(bytearray(active_file.read()), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, 1)
    img_rgb_raw = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Pre-Prediction Optimization (Resize 4K to 2K)
    img_rgb_input = pre_process_resize(img_rgb_raw, target_width=1200)

    with st.status("🚀 Processing with Neural Engine...", expanded=True) as status:
        st.write("Optimizing resolution...")
        st.write("Applying Zero-DCE & Retinex-Unet Fusion...")
        
        start_time = time.time()
        # AI Inference
        ai_output = enhancer.enhance_image(img_rgb_input)
        process_time = round(time.time() - start_time, 3)
        
        status.update(label=f"Finished in {process_time}s!", state="complete", expanded=False)

    # Results Layout
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.markdown("##### 🌑 Original Input")
        st.image(img_rgb_input, use_container_width=True)

    with col_right:
        st.markdown("##### 🌟 AI Enhanced Output")
        st.image(ai_output, use_container_width=True)
        
        # WebP Download Logic
        try:
            webp_data = get_webp_bytes(ai_output, quality=85)
            
            st.download_button(
                label="📩 DOWNLOAD WEBP (Optimized)",
                data=webp_data,
                file_name="deepsense_enhanced.webp",
                mime="image/webp"
            )
            
            with st.expander("Advanced Options"):
                # JPG Option
                output_bgr = cv2.cvtColor(ai_output, cv2.COLOR_RGB2BGR)
                _, jpg_buf = cv2.imencode(".jpg", output_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
                st.download_button(
                    label="Download as High-Res JPG",
                    data=jpg_buf.tobytes(),
                    file_name="enhanced_output.jpg",
                    mime="image/jpeg"
                )
        except Exception as e:
            st.error(f"Download error: {e}")

    # Reset
    if st.button("➕ PROCESS ANOTHER IMAGE"):
        reset_app()

st.divider()
st.caption("© 2026 DeepSense AI Lab | Built with PyTorch & Streamlit")
