import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import time

# --- Page Configuration ---
st.set_page_config(page_title="DeepSense AI Lab", layout="wide")

# --- Responsive Custom CSS ---
st.markdown("""
    <style>
    /* Global Background */
    .main { background-color: #0e1117; }
    
    /* Hero Section Responsive Text */
    .hero-text { 
        font-size: clamp(24px, 5vw, 48px) !important; 
        font-weight: 800; 
        color: #00d4ff; 
        text-align: center;
        padding: 0 10px;
    }
    .sub-text { 
        font-size: clamp(14px, 2vw, 18px) !important; 
        text-align: center; 
        color: #888; 
        margin-bottom: 30px;
    }

    /* Feature Cards Container */
    .feature-container {
        display: flex;
        flex-wrap: wrap;
        gap: 20px;
        justify-content: center;
    }

    .feature-card { 
        background-color: #1e2130; 
        padding: 20px; 
        border-radius: 15px; 
        border: 1px solid #333; 
        text-align: center;
        flex: 1 1 300px; /* Responsive basis */
        max-width: 350px;
    }

    /* Image Containers & Buttons */
    .stImage > img {
        border-radius: 12px;
        border: 1px solid #444;
        transition: transform 0.3s ease;
    }
    
    /* Responsive Download Button */
    div.stDownloadButton > button {
        width: 100% !important;
        background-color: #00c853 !important;
        color: white !important;
        padding: 12px !important;
        border-radius: 10px !important;
        font-size: 16px !important;
        font-weight: 600 !important;
        margin-top: 10px;
        border: none !important;
    }
    
    /* Remove padding for mobile */
    @media (max-width: 768px) {
        .main .block-container { padding: 1rem !important; }
    }
    </style>
    """, unsafe_allow_html=True)

# --- AI Core Logic ---
def run_ai_inference(img_np):
    # Placeholder: Replace with your actual model inference
    # Example: output = model(input)
    start_time = time.time()
    enhanced = cv2.detailEnhance(img_np, sigma_s=12, sigma_r=0.15)
    end_time = time.time()
    return enhanced, round((end_time - start_time), 3)

# --- Sidebar ---
st.sidebar.markdown("## 📥 Data Input")
uploaded_file = st.sidebar.file_uploader("Upload Image to Process", type=["jpg", "jpeg", "png"])

if uploaded_file is None:
    # === RESPONSIVE LANDING PAGE ===
    st.markdown('<p class="hero-text">Next-Gen AI Restoration</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-text">Give your low-light photos a new life with our trained Neural Network.</p>', unsafe_allow_html=True)
    
    # Using columns for features - Streamlit automatically stacks these on mobile
    f1, f2, f3 = st.columns(3)
    with f1:
        st.markdown('<div class="feature-card"><h3>🔍 Neural Resolution</h3><p>Intelligently reconstructs missing details.</p></div>', unsafe_allow_html=True)
    with f2:
        st.markdown('<div class="feature-card"><h3>💡 Low-Light AI</h3><p>Exposure recovery specifically tuned for dark sensors.</p></div>', unsafe_allow_html=True)
    with f3:
        st.markdown('<div class="feature-card"><h3>🛡️ Deep Denoise</h3><p>Removes digital grain while preserving edge sharpness.</p></div>', unsafe_allow_html=True)
    
    st.divider()
    st.info("👈 Upload an image from the sidebar to start the AI Engine.")

else:
    # === PROCESSING & RESPONSIVE DISPLAY ===
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, 1)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    with st.spinner("🧠 AI Model is analyzing pixels..."):
        ai_output, proc_time = run_ai_inference(img_rgb)

    st.markdown(f"### ⚡ AI Processed in `{proc_time}s`")
    
    # Streamlit columns are natively responsive (side-by-side on PC, stacked on Mobile)
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.markdown("#### 📸 ORIGINAL")
        st.image(img_rgb, use_container_width=True)

    with col_right:
        st.markdown("#### ✨ AI ENHANCED")
        st.image(ai_output, use_container_width=True)
        
        # Download Logic
        output_bgr = cv2.cvtColor(ai_output, cv2.COLOR_RGB2BGR)
        _, buffer = cv2.imencode('.jpg', output_bgr)
        
        st.download_button(
            label="📩 DOWNLOAD RESULT",
            data=buffer.tobytes(),
            file_name="ai_enhanced.jpg",
            mime="image/jpeg"
        )

    # Reset Option in Sidebar
    if st.sidebar.button("🔄 Clear & Restart"):
        st.rerun()

# --- Footer ---
st.divider()
st.caption("DeepSense AI Lab | Built with Python, OpenCV & Streamlit | Responsive Design v2.0")
