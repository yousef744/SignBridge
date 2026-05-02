import streamlit as st
import numpy as np
from PIL import Image
import tflite_runtime.interpreter as tflite
import json
import os

# Page config
st.set_page_config(page_title="SignBridge - ASL Translator", page_icon="🤟", layout="centered")

# Title
st.title("🤟 SignBridge - ASL Translator")
st.markdown("### Real-time Sign Language Translation using AI")

# Load model
@st.cache_resource
def load_model():
    interpreter = tflite.Interpreter(model_path='models/asl_model.tflite')
    interpreter.allocate_tensors()
    return interpreter

@st.cache_resource
def load_labels():
    with open('models/label_mapping.json', 'r') as f:
        return json.load(f)

try:
    interpreter = load_model()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    labels = load_labels()
    model_loaded = True
except Exception as e:
    st.error(f"Error loading model: {e}")
    model_loaded = False

# Input method
st.markdown("---")
input_method = st.radio("Choose input method:", ["📁 Upload Image", "📷 Camera"])

prediction = None
confidence = None

if input_method == "📁 Upload Image":
    uploaded_file = st.file_uploader("Upload hand sign image", type=['jpg', 'jpeg', 'png'])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption='Uploaded Image', use_column_width=True)
        
        if model_loaded:
            # Preprocess
            img = image.resize((224, 224))
            img_array = np.array(img, dtype=np.float32) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            # Predict
            interpreter.set_tensor(input_details[0]['index'], img_array)
            interpreter.invoke()
            output = interpreter.get_tensor(output_details[0]['index'])
            
            prediction_idx = np.argmax(output[0])
            confidence = float(output[0][prediction_idx])
            
            # Get label
            labels_inv = {v: k for k, v in labels.items()}
            prediction = labels_inv.get(prediction_idx, str(prediction_idx))

elif input_method == "📷 Camera":
    camera_image = st.camera_input("Take a photo")
    if camera_image is not None:
        image = Image.open(camera_image).convert('RGB')
        
        if model_loaded:
            # Preprocess
            img = image.resize((224, 224))
            img_array = np.array(img, dtype=np.float32) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            # Predict
            interpreter.set_tensor(input_details[0]['index'], img_array)
            interpreter.invoke()
            output = interpreter.get_tensor(output_details[0]['index'])
            
            prediction_idx = np.argmax(output[0])
            confidence = float(output[0][prediction_idx])
            
            labels_inv = {v: k for k, v in labels.items()}
            prediction = labels_inv.get(prediction_idx, str(prediction_idx))

# Show result
if prediction:
    st.markdown("---")
    st.success(f"### Predicted Letter: **{prediction}**")
    st.progress(min(confidence, 1.0))
    st.info(f"Confidence: {confidence:.2%}")

# Footer
st.markdown("---")
st.caption("Powered by SignBridge Team | CNN + TFLite")