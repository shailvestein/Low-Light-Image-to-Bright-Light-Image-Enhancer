import streamlit as st
import cv2
import numpy as np
import time
import io
from PIL import Image
import torch
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
    # Make sure load_weights handles device internally or move to device here
    gfmn_model, retinex_model = load_weights()
    enhancer_1 = Enhancer(gfmn_model, name='gfmn', batch_size=4)
    enhancer_2 = Enhancer(retinex_model, name='retinex', batch_size=4)
    return enhancer_1, enhancer_2

enhancer_1, enhancer_2 = get_enhancer()

def get_webp_bytes(image_uint8, quality=90):
    """Safely converts uint8 numpy array to WebP bytes."""
    try:
        img = Image.fromarray(image_uint8)
        buf = io.BytesIO()
        img.save(buf, format='WEBP', quality=quality, method=6)
        return buf.getvalue()
    except Exception as e:
        st.error(f"Error in image conversion: {e}")
        return None

def pre_process_resize(image_rgb, target_width=1200):
    h, w = image_rgb.shape[:2]
    if w <= target_width: return image_rgb
    aspect_ratio = h / w
    return cv2.resize(image_rgb, (target_width, int(target_width * aspect_ratio)), interpolation=cv2.INTER_AREA)

# --- 5. UI HEADER ---
st.markdown("<h1 style='text-align: center; color: #00d4ff;'>📸 DeepSense AI Light Restoration</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #888;'>Pro-grade Low-Light Image Enhancement Powered by PyTorch</p>", unsafe_allow_html=True)

# --- 6. UPLOADER ---
uploader_key = f"uploader_{st.session_state.reset_counter}"
uploaded_file = st.file_uploader("Upload Low-light Image", type=["jpg", "jpeg", "png"], key=uploader_key)

if uploaded_file is not None:
    # Load Image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, 1)
    img_rgb_raw = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    # img_input = pre_process_resize(img_rgb_raw, target_width=1024)

    # --- FANCY PROCESSING ---
    enhc_img_display = None
    p_time = 0

    with st.status("🚀 AI Engine is working...", expanded=True) as status:
        try:
            st.write("🧪 Analyzing scene lighting...")
            # Run Enhancer (Ensure enhancer_2 uses the model correctly)
            raw_output, p_time = enhancer_2.enhance_image(img_rgb_raw)
            
            st.write("🎨 Balancing color channels...")
            # Convert to uint8 RGB for display
            enhc_img_display = process_output_for_display(raw_output)
            
            status.update(label=f"✨ Magic Done in {p_time:.2f}s!", state="complete", expanded=False)
        except Exception as e:
            status.update(label="❌ Error occurred during enhancement", state="error")
            st.error(f"Processing Error: {e}")

    # --- DISPLAY & DOWNLOAD ---
    if enhc_img_display is not None:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("<h5 style='text-align: center;'>🌑 Original</h5>", unsafe_allow_html=True)
            st.image(img_input, use_container_width=True)
        with col2:
            st.markdown("<h5 style='text-align: center; color: #00d4ff;'>🌟 Enhanced</h5>", unsafe_allow_html=True)
            st.image(raw_output, use_container_width=True)

        # --- ACTIONS ---
        st.divider()
        c1, c2, _ = st.columns([1, 1, 1])
        with c1:
            webp_data = get_webp_bytes(enhc_img_display, quality=95)
            if webp_data:
                st.download_button(
                    label="📩 Download High-Res Result",
                    data=webp_data,
                    file_name="deepsense_enhanced.webp",
                    mime="image/webp"
                )
        with c2:
            if st.button("🔄 Enhance Another Photo"):
                trigger_reset()
else:
    st.info("👋 Welcome! Please upload a photo to start the restoration.")

# --- 7. FOOTER ---
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown(
    """
    <div style='text-align: center; border-top: 1px solid #333; padding-top: 20px;'>
        <p style='color: #888; font-size: 13px;'>Built with PyTorch & OpenCV | Powered by Shailesh Vishwakarma</p>
    </div>
    """, 
    unsafe_allow_html=True
)
