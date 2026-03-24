import streamlit as st
from PIL import Image, ImageEnhance
import numpy as np

# --- Page Configuration ---
st.set_page_config(page_title="AI Image Enhancer", layout="wide")

def adjust_rgb(image, r_weight, g_weight, b_weight):
    """Adjusts individual RGB channels of the image."""
    # Convert to array to manipulate channels
    img_array = np.array(image).astype(np.float32)
    
    # Apply weights to each channel
    img_array[:, :, 0] *= r_weight  # Red
    img_array[:, :, 1] *= g_weight  # Green
    img_array[:, :, 2] *= b_weight  # Blue
    
    # Clip values to stay within [0, 255] and convert back to uint8
    img_array = np.clip(img_array, 0, 255).astype(np.uint8)
    return Image.fromarray(img_array)

def enhance_image(image, brightness, contrast, sharpness, r_w, g_w, b_w):
    # 1. Apply RGB Channel Weights first
    image = adjust_rgb(image, r_w, g_w, b_w)
    
    # 2. Apply standard enhancements
    image = ImageEnhance.Brightness(image).enhance(brightness)
    image = ImageEnhance.Contrast(image).enhance(contrast)
    image = ImageEnhance.Sharpness(image).enhance(sharpness)
    return image

# --- Sidebar UI ---
st.sidebar.header("Standard Settings")
brightness = st.sidebar.slider("Brightness", 0.5, 3.0, 1.0)
contrast = st.sidebar.slider("Contrast", 0.5, 3.0, 1.0)
sharpness = st.sidebar.slider("Sharpness", 0.5, 5.0, 1.0)

st.sidebar.header("Color Channels (RGB)")
r_weight = st.sidebar.slider("Red Channel", 0.0, 2.0, 1.0)
g_weight = st.sidebar.slider("Green Channel", 0.0, 2.0, 1.0)
b_weight = st.sidebar.slider("Blue Channel", 0.0, 2.0, 1.0)

# --- Main UI ---
st.title("✨ Image Enhancement Lab")

uploaded_file = st.sidebar.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    original_img = Image.open(uploaded_file).convert("RGB")
    
    # Process
    enhanced_img = enhance_image(original_img, brightness, contrast, sharpness, r_weight, g_weight, b_weight)

    # Side-by-Side
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original")
        st.image(original_img, use_container_width=True)
    with col2:
        st.subheader("Enhanced")
        st.image(enhanced_img, use_container_width=True)

    # Download
    import io
    buf = io.BytesIO()
    enhanced_img.save(buf, format="JPEG")
    st.download_button("Download Image", buf.getvalue(), "enhanced.jpg", "image/jpeg")
else:
    st.info("Upload an image in the sidebar to start adjusting colors.")
