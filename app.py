import streamlit as st
from PIL import Image, ImageEnhance
import numpy as np
import cv2  # Added for efficient HSL manipulation

# --- Page Configuration ---
st.set_page_config(page_title="Pro Image Enhancer", layout="wide")

def adjust_hsl(image, hue, sat, light):
    """Adjusts Hue, Saturation, and Lightness using OpenCV."""
    img_array = np.array(image)
    # Convert RGB to HLS (Hue, Lightness, Saturation)
    hls = cv2.cvtColor(img_array, cv2.COLOR_RGB2HLS).astype(np.float32)
    
    # Apply adjustments
    hls[:, :, 0] = (hls[:, :, 0] + hue) % 180  # Hue is 0-179 in OpenCV
    hls[:, :, 1] = np.clip(hls[:, :, 1] * light, 0, 255)
    hls[:, :, 2] = np.clip(hls[:, :, 2] * sat, 0, 255)
    
    # Convert back to RGB
    final_img = cv2.cvtColor(hls.astype(np.uint8), cv2.COLOR_HLS2RGB)
    return Image.fromarray(final_img)

def adjust_rgb(image, r_w, g_w, b_w):
    img_array = np.array(image).astype(np.float32)
    img_array[:, :, 0] *= r_w
    img_array[:, :, 1] *= g_w
    img_array[:, :, 2] *= b_w
    return Image.fromarray(np.clip(img_array, 0, 255).astype(np.uint8))

# --- Sidebar UI ---
st.sidebar.title("🎨 Image Controls")
uploaded_file = st.sidebar.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

# Use Expanders to keep the sidebar height "low" and organized
with st.sidebar.expander("✨ Basic Enhancement", expanded=True):
    brightness = st.slider("Brightness", 0.5, 3.0, 1.0, 0.1)
    contrast = st.slider("Contrast", 0.5, 3.0, 1.0, 0.1)
    sharpness = st.slider("Sharpness", 0.0, 5.0, 1.0, 0.1)

with st.sidebar.expander("🌈 Color & HSL"):
    hue = st.slider("Hue Shift", -90, 90, 0, 1)
    saturation = st.slider("Saturation", 0.0, 3.0, 1.0, 0.1)
    lightness = st.slider("Lightness", 0.0, 3.0, 1.0, 0.1)

with st.sidebar.expander("🔴 RGB Channels"):
    r_weight = st.slider("Red", 0.0, 2.0, 1.0, 0.05)
    g_weight = st.slider("Green", 0.0, 2.0, 1.0, 0.05)
    b_weight = st.slider("Blue", 0.0, 2.0, 1.0, 0.05)

# --- Main UI ---
st.title("Pro Image Enhancement Lab")

if uploaded_file:
    original_img = Image.open(uploaded_file).convert("RGB")
    
    # Processing Chain
    # 1. RGB Tuning
    proc_img = adjust_rgb(original_img, r_weight, g_weight, b_weight)
    # 2. HSL Tuning
    proc_img = adjust_hsl(proc_img, hue, saturation, lightness)
    # 3. Final Polish
    proc_img = ImageEnhance.Brightness(proc_img).enhance(brightness)
    proc_img = ImageEnhance.Contrast(proc_img).enhance(contrast)
    proc_img = ImageEnhance.Sharpness(proc_img).enhance(sharpness)

    # Comparison Columns
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original")
        st.image(original_img, use_container_width=True)
    with col2:
        st.subheader("Enhanced")
        st.image(proc_img, use_container_width=True)

    # Download
    import io
    buf = io.BytesIO()
    proc_img.save(buf, format="JPEG")
    st.download_button("📩 Download Enhanced Image", buf.getvalue(), "enhanced.jpg", "image/jpeg")
else:
    st.info("Please upload an image to begin.")
