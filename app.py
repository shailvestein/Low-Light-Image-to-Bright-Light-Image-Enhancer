import streamlit as st
from PIL import Image, ImageEnhance
import numpy as np
import cv2

# --- Page Configuration ---
st.set_page_config(page_title="Pro AI Image Enhancer", layout="wide")

def auto_enhance(image):
    """Automatically adjusts brightness and contrast using Histogram Equalization."""
    img_array = np.array(image)
    img_yuv = cv2.cvtColor(img_array, cv2.COLOR_RGB2YUV)
    # Equalize the histogram of the Y channel (brightness)
    img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
    auto_img = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)
    return Image.fromarray(auto_img)

def denoise_image(image, strength):
    """Removes noise/grain from the image."""
    img_array = np.array(image)
    # Strength factor (h) usually stays between 3-10
    denoised = cv2.fastNlMeansDenoisingColored(img_array, None, strength, strength, 7, 21)
    return Image.fromarray(denoised)

def adjust_hsl(image, hue, sat, light):
    img_array = np.array(image)
    hls = cv2.cvtColor(img_array, cv2.COLOR_RGB2HLS).astype(np.float32)
    hls[:, :, 0] = (hls[:, :, 0] + hue) % 180
    hls[:, :, 1] = np.clip(hls[:, :, 1] * light, 0, 255)
    hls[:, :, 2] = np.clip(hls[:, :, 2] * sat, 0, 255)
    return Image.fromarray(cv2.cvtColor(hls.astype(np.uint8), cv2.COLOR_HLS2RGB))

# --- Sidebar UI ---
st.sidebar.title("🛠️ Pro Editor")
uploaded_file = st.sidebar.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

# Magic Buttons
st.sidebar.subheader("Quick Actions")
auto_on = st.sidebar.button("🪄 Auto-Enhance")
reset = st.sidebar.button("🔄 Reset All")

with st.sidebar.expander("✨ Basic & Denoise", expanded=True):
    brightness = st.slider("Brightness", 0.5, 3.0, 1.0)
    contrast = st.slider("Contrast", 0.5, 3.0, 1.0)
    denoise_strength = st.slider("Remove Noise (Denoise)", 0, 20, 0)

with st.sidebar.expander("🌈 Color & HSL"):
    saturation = st.slider("Saturation", 0.0, 3.0, 1.0)
    hue = st.slider("Hue Shift", -90, 90, 0)
    lightness = st.slider("Lightness", 0.0, 3.0, 1.0)

with st.sidebar.expander("🔴 RGB Channels"):
    r_w = st.slider("Red", 0.0, 2.0, 1.0)
    g_w = st.slider("Green", 0.0, 2.0, 1.0)
    b_w = st.slider("Blue", 0.0, 2.0, 1.0)

# --- Main UI ---
st.title("Pro Image Enhancement Lab")

if uploaded_file:
    original_img = Image.open(uploaded_file).convert("RGB")
    
    # 1. Start with Original or Auto-Enhanced
    if auto_on:
        proc_img = auto_enhance(original_img)
    else:
        proc_img = original_img.copy()

    # 2. Denoising (Dhaure kaam karta hai, isliye strength 0 pe off rahega)
    if denoise_strength > 0:
        proc_img = denoise_image(proc_img, denoise_strength)

    # 3. Apply Manual Adjustments
    # RGB
    arr = np.array(proc_img).astype(np.float32)
    arr[:,:,0] *= r_w; arr[:,:,1] *= g_w; arr[:,:,2] *= b_w
    proc_img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    
    # HSL & Others
    proc_img = adjust_hsl(proc_img, hue, saturation, lightness)
    proc_img = ImageEnhance.Brightness(proc_img).enhance(brightness)
    proc_img = ImageEnhance.Contrast(proc_img).enhance(contrast)

    # Comparison
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Before")
        st.image(original_img, use_container_width=True)
    with col2:
        st.subheader("After")
        st.image(proc_img, use_container_width=True)

    # Download
    import io
    buf = io.BytesIO()
    proc_img.save(buf, format="JPEG")
    st.download_button("📩 Download Result", buf.getvalue(), "enhanced.jpg")
else:
    st.info("Side menu se image upload karein!")
