import streamlit as st
import cv2
import numpy as np
from PIL import Image
from utils import load_weights
from Enhancer import Enhancer

# --- 1. Model Loading ---
@st.cache_resource
def get_enhancer():
    # Weights load karke enhancer return karega
    model = load_weights()
    return Enhancer(model, batch_size=1)

enhancer = get_enhancer()

# --- 2. Simple UI Setup ---
st.set_page_config(layout="wide", page_title="AI Photo Enhancer")
st.title("🌓 Simple AI Photo Enhancer")
st.write("Upload a dark photo to see the AI magic.")

# --- 3. Image Upload ---
uploaded_file = st.file_uploader("Choose a photo...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # --- 4. Read & Resize Logic ---
    # Convert uploaded file to OpenCV format
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Simple Resize (2K limit) to maintain quality and speed
    h, w = img_rgb.shape[:2]
    target_w = 1000
    if w > target_w:
        aspect_ratio = h / w
        img_rgb = cv2.resize(img_rgb, (target_w, int(target_w * aspect_ratio)), interpolation=cv2.INTER_AREA)

    # --- 5. AI Enhancement ---
    with st.spinner("Processing... Please wait."):
        # Enhancer class ka use karke output nikalna
        enhanced_img = enhancer.enhance_image(img_rgb)

    # --- 6. Side by Side Display ---
    st.divider()
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Image")
        st.image(img_rgb, use_container_width=True)

    with col2:
        st.subheader("AI Enhanced Image")
        st.image(enhanced_img, use_container_width=True)

    # --- 7. Simple Download ---
    st.divider()
    # Convert back to BGR for saving
    result_bgr = cv2.cvtColor(enhanced_img, cv2.COLOR_RGB2BGR)
    _, buffer = cv2.imencode('.jpg', result_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    
    st.download_button(
        label="Download Enhanced Photo",
        data=buffer.tobytes(),
        file_name="enhanced_result.jpg",
        mime="image/jpeg"
    )
