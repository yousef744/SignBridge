import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
import os

# Page config
st.set_page_config(
    page_title="SignBridge - ASL Translator",
    page_icon="🤟",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS
st.markdown("""
<style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 25px;
        padding: 12px 40px;
        border: none;
        font-weight: bold;
        font-size: 1.1em;
        width: 100%;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
    }
    .prediction-box {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        border-radius: 20px;
        padding: 30px;
        text-align: center;
        margin: 20px 0;
    }
    .big-letter {
        font-size: 150px;
        font-weight: bold;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        line-height: 1;
        margin: 20px 0;
    }
    .confidence-bar {
        background: #e0e0e0;
        border-radius: 15px;
        height: 25px;
        overflow: hidden;
        margin: 15px 0;
    }
    .confidence-fill {
        height: 100%;
        background: linear-gradient(90deg, #667eea, #764ba2);
        border-radius: 15px;
        transition: width 0.5s ease;
    }
    .team-card {
        background: white;
        border-radius: 15px;
        padding: 20px;
        margin-top: 30px;
        box-shadow: 0 5px 20px rgba(0,0,0,0.1);
    }
    .header-bg {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 20px;
        padding: 30px;
        margin-bottom: 30px;
        text-align: center;
        color: white;
    }
    .github-link {
        text-align: center;
        margin-top: 20px;
    }
    .github-link a {
        color: #667eea;
        text-decoration: none;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="header-bg">
    <h1 style="font-size: 3em; margin-bottom: 10px;">🤟 SignBridge</h1>
    <p style="font-size: 1.3em; opacity: 0.9;">
        Real-Time American Sign Language Translator
    </p>
    <p style="font-size: 0.9em; opacity: 0.7;">
        Powered by TensorFlow + MediaPipe
    </p>
</div>
""", unsafe_allow_html=True)

# Load model
@st.cache_resource
def load_model():
    """Load the trained ASL model"""
    model_paths = [
        'models/best_model.h5',
        'models/asl_model.keras',
        'models/asl_model_saved',
    ]

    for path in model_paths:
        if os.path.exists(path):
            try:
                if path.endswith('.h5') or path.endswith('.keras'):
                    return tf.keras.models.load_model(path)
                else:
                    return tf.keras.models.load_model(path)
            except Exception as e:
                st.warning(f"Failed to load {path}: {e}")
                continue

    return None

model = load_model()

# ASL labels
ASL_LABELS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
              'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

def preprocess_image(image):
    """Preprocess image for prediction"""
    img_array = np.array(image)

    # Convert to grayscale
    if len(img_array.shape) == 3:
        img = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        img = img_array

    # Resize to 28x28
    img = cv2.resize(img, (28, 28))

    # Normalize
    img = img.astype('float32') / 255.0

    # Reshape for model
    img = img.reshape(1, 28, 28, 1)

    return img

# Main content
st.markdown("<br>", unsafe_allow_html=True)

# Input method selection
option = st.radio(
    "📸 Choose input method:",
    ["Upload Image", "Use Camera"],
    horizontal=True,
    label_visibility="collapsed"
)

st.markdown("<br>", unsafe_allow_html=True)

image = None

if option == "Upload Image":
    uploaded_file = st.file_uploader(
        "Drop your hand sign image here",
        type=['jpg', 'jpeg', 'png'],
        help="Upload a clear image of an ASL hand sign"
    )
    if uploaded_file:
        image = Image.open(uploaded_file)
else:
    camera_image = st.camera_input("Take a photo of your hand sign")
    if camera_image:
        image = Image.open(camera_image)

if image:
    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("""
        <div class="prediction-box">
            <h3 style="color: #667eea; margin-bottom: 15px;">📷 Input Image</h3>
        </div>
        """, unsafe_allow_html=True)
        st.image(image, use_container_width=True)

    with col2:
        st.markdown("""
        <div class="prediction-box">
            <h3 style="color: #667eea; margin-bottom: 15px;">🎯 Prediction</h3>
        </div>
        """, unsafe_allow_html=True)

        if model:
            with st.spinner('Analyzing hand sign...'):
                # Preprocess
                processed = preprocess_image(image)

                # Predict
                predictions = model.predict(processed, verbose=0)
                predicted_class = int(np.argmax(predictions[0]))
                confidence = float(predictions[0][predicted_class])
                letter = ASL_LABELS[predicted_class]

            # Display result
            st.markdown(f"""
            <div style="text-align: center; padding: 20px;">
                <p style="font-size: 1.1em; color: #666; margin-bottom: 10px;">
                    Predicted Letter
                </p>
                <div class="big-letter">{letter}</div>
                <p style="font-size: 1.3em; color: #667eea; font-weight: bold;">
                    {letter} - {confidence*100:.1f}% Confidence
                </p>
                <div class="confidence-bar">
                    <div class="confidence-fill" style="width: {confidence*100}%"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Top 5 predictions
            st.markdown("<h4 style='text-align: center; color: #666; margin-top: 20px;'>Top Predictions</h4>", unsafe_allow_html=True)

            top5_idx = np.argsort(predictions[0])[-5:][::-1]
            for idx in top5_idx:
                prob = float(predictions[0][idx])
                letter_name = ASL_LABELS[idx]
                st.progress(prob, text=f"{letter_name}: {prob*100:.1f}%")

        else:
            st.error("❌ Model not found!")
            st.info("Please ensure the model file exists in the `models/` directory.")

            # Demo mode fallback
            st.markdown("""
            <div style="background: #fff3e0; border-radius: 15px; padding: 20px; margin: 20px 0;">
                <h4 style="color: #ef6c00; text-align: center;">⚡ Demo Mode</h4>
                <p style="text-align: center; color: #666;">Showing random prediction for demonstration.</p>
            </div>
            """, unsafe_allow_html=True)

            import random
            demo_letter = random.choice(ASL_LABELS)
            demo_conf = random.uniform(0.75, 0.99)

            st.markdown(f"""
            <div style="text-align: center; padding: 20px;">
                <div class="big-letter">{demo_letter}</div>
                <p style="font-size: 1.2em; color: #666;">
                    {demo_letter} - {demo_conf*100:.1f}% (Demo)
                </p>
            </div>
            """, unsafe_allow_html=True)

else:
    # Show instructions when no image
    st.markdown("""
    <div style="background: #f8f9fa; border-radius: 20px; padding: 40px; text-align: center; margin: 20px 0;">
        <h3 style="color: #667eea; margin-bottom: 20px;">👋 How to Use</h3>
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px; text-align: left;">
            <div style="background: white; padding: 20px; border-radius: 15px;">
                <h4 style="color: #667eea;">1. Upload Image</h4>
                <p style="color: #666;">Upload a clear photo of an ASL hand sign</p>
            </div>
            <div style="background: white; padding: 20px; border-radius: 15px;">
                <h4 style="color: #667eea;">2. Use Camera</h4>
                <p style="color: #666;">Take a photo directly using your webcam</p>
            </div>
        </div>
        <p style="margin-top: 20px; color: #999;">
            The AI will analyze the hand gesture and predict the corresponding letter (A-Z).
        </p>
    </div>
    """, unsafe_allow_html=True)

# Team info
st.markdown("""
<div class="team-card">
    <h3 style="text-align: center; color: #333; margin-bottom: 20px;">👥 Team SignBridge</h3>
    <table style="width: 100%; text-align: center; border-collapse: collapse;">
        <tr style="background: #f8f9fa;">
            <td style="padding: 12px; font-weight: bold; color: #667eea;">Team Leader</td>
            <td style="padding: 12px;">Yousef Osama Abd Elwarth</td>
        </tr>
        <tr>
            <td style="padding: 12px; font-weight: bold; color: #667eea;">Members</td>
            <td style="padding: 12px;">
                Mohamad Badran, Hassan Hamde, Yousef Elmahroke, 
                Yousef AlQai'i, Abdelrahman Allam
            </td>
        </tr>
    </table>
</div>
""", unsafe_allow_html=True)

# GitHub link
st.markdown("""
<div class="github-link">
    <p>📁 <a href="https://github.com/yousef744/SignBridge" target="_blank">
        View Source Code on GitHub
    </a></p>
</div>
""", unsafe_allow_html=True)

# Footer
st.markdown("""
<div style="text-align: center; padding: 30px 20px; color: #999; margin-top: 20px;">
    <p>Built with ❤️ using Streamlit + TensorFlow + OpenCV</p>
    <p style="font-size: 0.8em;">SignBridge Project © 2024</p>
</div>
""", unsafe_allow_html=True)
