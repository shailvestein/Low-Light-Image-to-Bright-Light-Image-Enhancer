import streamlit as st
import cv2
import numpy as np
import time
import io
from PIL import Image
from models import load_weights
from Enhancer import Enhancer

# --- 1. SET PAGE CONFIG ---
st.set_page_config(layout="wide", page_title="DeepSense AI Lab", page_icon="✨")

# --- 2. SESSION STATE FOR RESET ---
if 'reset_counter' not in st.session_state:
    st.session_state.reset_counter = 0

def trigger_reset():
    st.session_state.reset_counter += 1
    st.rerun()

# --- 3. MODEL LOADING ---
@st.cache_resource
def get_enhancer():
    gfmn_model, retinex_model = load_weights()
    enhancer_1 = Enhancer(gfmn_model, name='gfmn', batch_size=4)
    enhancer_2 = Enhancer(retinex_model, name='retinex', batch_size=4)
    return enhancer_1, enhancer_2

enhancer_1, enhancer_2 = get_enhancer()

# --- 4. SIMPLE DOWNLOAD HELPER ---
def get_image_bytes(image_np):
    """Directly converts uint8 numpy array to bytes without extra processing."""
    img = Image.fromarray(image_np)
    buf = io.BytesIO()
    img.save(buf, format='PNG') # PNG is safe and lossless
    return buf.getvalue()

# --- 5. UI HEADER ---
st.markdown("<h1 style='text-align: center; color: #00d4ff;'>📸 DeepSense AI Light Restoration</h1>", unsafe_allow_html=True)

# --- 6. UPLOADER ---
uploader_key = f"uploader_{st.session_state.reset_counter}"
uploaded_file = st.file_uploader("Upload Low-light Image", type=["jpg", "jpeg", "png"], key=uploader_key)

if uploaded_file is not None:
    # Load Image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, 1)
    # img_input = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # --- PROCESSING ---
    with st.status("🚀 AI Engine is working...", expanded=True) as status:
        # Direct enhancement call
        # Make sure your Enhancer class returns a proper uint8 numpy array
        img_input = (img_input/255.0).astype(np.float32))
        enhc_img, p_time = enhancer_1.enhance_image(img_input)
        status.update(label=f"✨ Magic Done in {p_time:.2f}s!", state="complete", expanded=False)

    # --- DISPLAY ---
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<h4 style='text-align: center;'>🌑 Original</h4>", unsafe_allow_html=True)
        st.image(img_input, use_container_width=True)
    with col2:
        st.markdown("<h4 style='text-align: center; color: #00d4ff;'>🌟 Enhanced</h4>", unsafe_allow_html=True)
        st.image(enhc_img, use_container_width=True)

    # --- ACTIONS ---
    st.divider()
    c1, c2, _ = st.columns([1, 1, 1])
    with c1:
        # Minimal download logic
        img_bytes = get_image_bytes(enhc_img)
        st.download_button("📩 Download Result", data=img_bytes, file_name="enhanced.png", mime="image/png")
    with c2:
        if st.button("🔄 Enhance Another Photo"):
            trigger_reset()

else:
    st.info("👋 Welcome! Please upload a photo to start.")

# --- 7. FOOTER ---
st.markdown("<br><div style='text-align: center; color: #888;'>Powered by Shailesh Vishwakarma</div>", unsafe_allow_html=True)
