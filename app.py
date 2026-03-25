import streamlit as st
import cv2
import numpy as np
import time
import io
from PIL import Image
from utils import load_weights
from Enhancer import Enhancer

# --- 1. SET PAGE CONFIG ---
st.set_page_config(layout="wide", page_title="AI Photo Lab")

# --- 2. SESSION STATE FOR RESET ---
# Agar key nahi hai toh 0 se shuru karein
if 'reset_counter' not in st.session_state:
    st.session_state.reset_counter = 0

def trigger_reset():
    # Counter badhane se file_uploader ki 'key' badal jayegi aur wo khali ho jayega
    st.session_state.reset_counter += 1
    # Purana processed data clear karne ke liye session state ko clean karein (optional)
    st.rerun()

# --- 3. MODEL LOADING (CACHED) ---
@st.cache_resource
def get_enhancer():
    model = load_weights()
    return Enhancer(model, batch_size=1)

enhancer = get_enhancer()

# --- 4. HELPER FUNCTIONS ---
def get_webp_bytes(image_rgb, quality=85):
    img = Image.fromarray(image_rgb)
    buf = io.BytesIO()
    img.save(buf, format='WEBP', quality=quality, method=6)
    return buf.getvalue()

def pre_process_resize(image_rgb, target_width=1200):
    h, w = image_rgb.shape[:2]
    if w <= target_width:
        return image_rgb
    aspect_ratio = h / w
    target_height = int(target_width * aspect_ratio)
    return cv2.resize(image_rgb, (target_width, target_height), interpolation=cv2.INTER_AREA)

# --- 5. UI HEADER ---
st.title("🚀 AI Image Restoration Lab")

# --- 6. IMAGE UPLOAD (With Dynamic Key) ---
# Jab reset button dabega, key 'uploader_1', 'uploader_2' aise badlegi, jisse reset pakka hoga
uploader_key = f"uploader_{st.session_state.reset_counter}"
uploaded_file = st.file_uploader("Drop your image here", type=["jpg", "jpeg", "png"], key=uploader_key)

if uploaded_file is not None:
    # Read Image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, 1)
    img_rgb_raw = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    # STEP 1: Resize to 1200px
    img_input = pre_process_resize(img_rgb_raw, target_width=1200)

    # STEP 2: AI Enhancement
    with st.spinner("AI is working..."):
        ai_output, p_time = enhancer.enhance_image(img_input) 

    # --- 7. DISPLAY SIDE-BY-SIDE ---
    st.success(f"Restoration Complete in {p_time} seconds!")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original (1200px)")
        st.image(img_input, width='stretch')

    with col2:
        st.subheader("AI Enhanced")
        st.image(ai_output, width='stretch')

    # --- 8. DOWNLOAD & RESET SECTION ---
    st.divider()
    
    # Download Button
    webp_data = get_webp_bytes(ai_output, quality=90)
    st.download_button(
        label="📩 DOWNLOAD ENHANCED IMAGE",
        data=webp_data,
        file_name="deepsense_result.webp",
        mime="image/webp"
    )

    # RESET BUTTON (Ab ye kaam karega)
    if st.button("🔄 UPLOAD ANOTHER IMAGE"):
        trigger_reset()

else:
    st.info("Please upload an image to begin.")
