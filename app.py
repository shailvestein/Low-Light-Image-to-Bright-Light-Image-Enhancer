import streamlit as st
import cv2
import numpy as np
import time
import io
from PIL import Image
from utils import load_weights
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
    model = load_weights()
    return Enhancer(model, batch_size=4)

enhancer = get_enhancer()

# --- 4. HELPERS ---
def get_webp_bytes(image_rgb, quality=85):
    img = Image.fromarray(image_rgb)
    buf = io.BytesIO()
    img.save(buf, format='WEBP', quality=quality, method=6)
    return buf.getvalue()

def pre_process_resize(image_rgb, target_width=1200):
    h, w = image_rgb.shape[:2]
    if w <= target_width: return image_rgb
    aspect_ratio = h / w
    return cv2.resize(image_rgb, (target_width, int(target_width * aspect_ratio)), interpolation=cv2.INTER_AREA)

# --- 5. UI HEADER ---
st.markdown("<h1 style='text-align: center; color: #00d4ff;'>📸 DeepSense AI Restoration</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #888;'>Pro-grade Low Light Enhancement Powered by Neural Networks</p>", unsafe_allow_html=True)

# --- 6. UPLOADER ---
uploader_key = f"uploader_{st.session_state.reset_counter}"
uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"], key=uploader_key)

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, 1)
    img_rgb_raw = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_input = pre_process_resize(img_rgb_raw, target_width=1200)

    # --- FANCY PROCESSING ---
    with st.status("🚀 AI Engine is working...", expanded=True) as status:
        st.write("🧪 Analyzing scene lighting...")
        ai_output, p_time = enhancer.enhance_image(img_input)
        ai_output = cv2.cvtColor(ai_output, cv2.COLOR_BGR2RGB) # Blue tint fix
        
        st.write("🎨 Balancing color channels...")
        st.write("✅ Ready for download!")
        status.update(label=f"✨ Magic Done in {p_time:.0f}s!", state="complete", expanded=False)

    # --- DISPLAY ---
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<h4 style='text-align: center;'>🌑 Original</h4>", unsafe_allow_html=True)
        st.image(img_input, width='stretch')
    with col2:
        st.markdown("<h4 style='text-align: center; color: #00d4ff;'>🌟 Enhanced</h4>", unsafe_allow_html=True)
        st.image(ai_output, width='stretch')

    # --- DOWNLOAD & RESET ---
    st.divider()
    c1, c2, _ = st.columns([1, 1, 1])
    with c1:
        webp_data = get_webp_bytes(ai_output, quality=90)
        st.download_button("📩 Download High-Res Result", data=webp_data, file_name="enhanced.webp", mime="image/webp")
    with c2:
        if st.button("🔄 Enhance Another Photo"):
            trigger_reset()

else:
    st.info("👋 Welcome! Please upload a photo to start the restoration.")

# --- 7. FOOTER (With Clickable Email) ---
st.markdown("<br><br><br>", unsafe_allow_html=True)
st.markdown(
    """
    <div style='text-align: center; border-top: 1px solid #333; padding-top: 20px;'>
        <p style='color: #888; font-size: 13px; margin-bottom: 5px;'>
            Built with PyTorch & OpenCV
        </p>
        <p style='font-size: 14px;'>
            <span style='color: #555;'>Have a suggestion? </span>
            <a href="mailto:shailvestein.careers@gmail.com?subject=Suggestion for DeepSense AI Lab" 
               style="color: #00d4ff; text-decoration: none; font-weight: bold;">
               📩 Contact Developer
            </a>
        </p>
        <p style='color: #00d4ff; font-weight: bold; font-size: 15px; margin-top: 10px;'>
            Powered by Shailesh Vishwakarma
        </p>
    </div>
    """, 
    unsafe_allow_html=True
)
