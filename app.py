import streamlit as st
import cv2
import numpy as np
import time
import io
from PIL import Image
from utils import load_weights
from Enhancer import Enhancer

# --- Helper Functions (Top par rakhein) ---
@st.cache_resource
def load_enhancer():
    enhancer = Enhancer(load_weights(), batch_size=4)
    return enhancer

def get_webp_bytes(image_rgb, quality=85):
    img = Image.fromarray(image_rgb)
    buf = io.BytesIO()
    img.save(buf, format='WEBP', quality=quality, method=6)
    return buf.getvalue()

def pre_process_resize(image_rgb, target_width=1000):
    h, w = image_rgb.shape[:2]
    if w <= target_width:
        return image_rgb
    aspect_ratio = h/w
    target_height = int(target_width * aspect_ratio)
    return cv2.resize(image_rgb, (target_width, target_height), interpolation=cv2.INTER_AREA)

# Initialize Enhancer
enhancer = load_enhancer()

# --- [Page Config & Styles Same Rehenge...] ---
st.set_page_config(page_title="DeepSense AI Lab", layout="wide")
# ... (Aapka CSS Style block yahan aayega) ...

# --- UI Logic & File Uploader ---
# ... (Aapka Session State aur Sidebar logic yahan aayega) ...

# --- PROCESS & DISPLAY ---
if active_file:
    # Read Image
    file_bytes = np.asarray(bytearray(active_file.read()), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, 1)
    img_rgb_raw = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # 1. Resize before AI (Performance ke liye)
    img_rgb_input = pre_process_resize(img_rgb_raw, target_width=1000)

    with st.spinner("🤖 AI Model is analyzing..."):
        start_t = time.time()
        # AI Processing
        ai_output = enhancer.enhance_image(img_rgb_input)
        p_time = round(time.time() - start_t, 3)
        
    st.markdown(f"### ✨ Result Ready (`{p_time}s`)")

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("#### 📸 ORIGINAL")
        st.image(img_rgb_input, use_container_width=True)

    with col_b:
        st.markdown("#### 🚀 AI ENHANCED")
        st.image(ai_output, use_container_width=True)
        
        # --- FIXED DOWNLOAD LOGIC (WebP) ---
        try:
            # WebP logic use kar rahe hain yahan
            webp_data = get_webp_bytes(ai_output, quality=85)
            
            st.download_button(
                label="📩 DOWNLOAD OPTIMIZED IMAGE (WebP)",
                data=webp_data,
                file_name="deep_sense_enhanced.webp",
                mime="image/webp"
            )
            
            # Optional: JPG Download (Expandable)
            with st.expander("More formats"):
                output_bgr_final = cv2.cvtColor(ai_output, cv2.COLOR_RGB2BGR)
                _, buffer = cv2.imencode(".jpg", output_bgr_final, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
                st.download_button(
                    label="Download as JPG",
                    data=buffer.tobytes(),
                    file_name="enhanced_output.jpg",
                    mime="image/jpeg"
                )
        except Exception as e:
            st.error(f"Error: {e}")

        if st.button("➕ UPLOAD ANOTHER"):
            st.rerun()
