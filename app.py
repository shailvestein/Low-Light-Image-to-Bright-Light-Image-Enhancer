import streamlit as st
# from PIL import Image, ImageEnhance
import numpy as np

# --- Page Configuration ---
st.set_page_config(page_title="AI Image Enhancer", layout="wide")

def enhance_image(image, brightness, contrast, sharpness):
    """Applies basic enhancements to the image."""
    enhancer = ImageEnhance.Brightness(image)
    image = enhancer.enhance(brightness)
    
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(contrast)
    
    enhancer = ImageEnhance.Sharpness(image)
    image = enhancer.enhance(sharpness)
    return image

# --- Sidebar UI ---
st.sidebar.header("Settings")
st.sidebar.info("Adjust the sliders to enhance your photo.")

brightness = st.sidebar.slider("Brightness", 0.5, 3.0, 1.0)
contrast = st.sidebar.slider("Contrast", 0.5, 3.0, 1.0)
sharpness = st.sidebar.slider("Sharpness", 0.5, 5.0, 1.0)

# --- Main UI ---
st.title("✨ Image Enhancement Lab")
st.write("Upload a low-light or faded photo to see the transformation.")

uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load Image
    original_img = Image.open(uploaded_file).convert("RGB")
    
    # Process Image
    enhanced_img = enhance_image(original_img, brightness, contrast, sharpness)

    # --- Display Side-by-Side ---
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original")
        st.image(original_img, use_container_width=True, caption="Original Input")

    with col2:
        st.subheader("Enhanced")
        st.image(enhanced_img, use_container_width=True, caption="Enhanced Output")

    # --- Download Option ---
    st.divider()
    st.subheader("Download Results")
    # Convert PIL image to bytes for download
    import io
    buf = io.BytesIO()
    enhanced_img.save(buf, format="JPEG")
    byte_im = buf.getvalue()

    st.download_button(
        label="Download Enhanced Image",
        data=byte_im,
        file_name="enhanced_image.jpg",
        mime="image/jpeg"
    )
else:
    st.warning("Please upload an image file in the sidebar to get started.")

# --- Footer/Instructions ---
with st.expander("How it works"):
    st.write("""
        This tool uses the PIL (Python Imaging Library) to adjust pixel intensity and contrast. 
        For low-light images, increasing Brightness and Contrast usually yields the best results.
    """)

