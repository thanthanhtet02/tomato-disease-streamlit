# Import required libraries
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json
import os
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2, EfficientNetB0

# Page setup
st.set_page_config(page_title="Tomato Disease Classifier", layout="wide")

# Custom CSS for green theme
st.markdown("""
    <style>
    .stApp {
        background-color: #E8F5E9 !important;
    }
    
    .main {
        background-color: #E8F5E9 !important;
    }
    
    [data-testid="stFileUploader"] {
        background-color: #C8E6C9 !important;
        padding: 20px;
        border-radius: 10px;
        border: 2px dashed #81C784 !important;
    }
    
    [data-testid="stFileUploadDropzone"] {
        background-color: #C8E6C9 !important;
    }
    </style>
""", unsafe_allow_html=True)

st.title("Tomato Leaf Disease Classifier")
st.write("Upload a tomato leaf image to detect diseases using a deep learning ensemble model")

# Sidebar
with st.sidebar:
    if os.path.exists("assets/icons8-plant-94.png"):
        st.image("assets/icons8-plant-94.png", width=150)
    
    st.header("How to Use")
    st.write("1. Upload a tomato leaf image")
    st.write("2. Wait for the prediction")
    st.write("3. View the results")
    
    if os.path.exists("assets/happy leave.JPG"):
        st.write("---")
        st.write("Example:")
        st.image("assets/happy leave.JPG", caption="Sample: Healthy Tomato Leaf", width=150)

# Load class names
@st.cache_resource
def load_class_names():
    with open("class_names.json", "r") as f:
        return json.load(f)

# Load ensemble weights
@st.cache_resource
def load_ensemble_weights():
    with open("ensemble_weights.json", "r") as f:
        return json.load(f)

class_names = load_class_names()
weights = load_ensemble_weights()

# Build model architecture
def build_model(base_model_class, preprocess_fn, model_name, num_classes=7):
    base = base_model_class(weights=None, include_top=False, input_shape=(128, 128, 3))
    base.trainable = False
    
    model = models.Sequential([
        layers.Input(shape=(128, 128, 3)),
        layers.Lambda(lambda x: preprocess_fn(x)),
        base,
        layers.GlobalAveragePooling2D(),
        layers.BatchNormalization(),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation="softmax")
    ], name=model_name)
    
    return model

# Load models
@st.cache_resource
def load_models():
    mobilenet = build_model(
        MobileNetV2,
        tf.keras.applications.mobilenet_v2.preprocess_input,
        "MobileNetV2",
        num_classes=len(class_names)
    )
    mobilenet.load_weights("mobilenetv2.weights.h5")
    
    efficientnet = build_model(
        EfficientNetB0,
        tf.keras.applications.efficientnet.preprocess_input,
        "EfficientNetB0",
        num_classes=len(class_names)
    )
    efficientnet.load_weights("efficientnetb0.weights.h5")
    
    return mobilenet, efficientnet

model1, model2 = load_models()

# Preprocess image
def preprocess_image(img):
    img = img.convert("RGB").resize((128, 128))
    arr = np.array(img).astype(np.float32)
    arr = np.expand_dims(arr, axis=0)
    return arr

# Ensemble prediction
def predict(image):
    pred1 = model1.predict(image, verbose=0)[0]
    pred2 = model2.predict(image, verbose=0)[0]
    
    w1 = float(weights.get("MobileNetV2", 0.5))
    w2 = float(weights.get("EfficientNetB0", 0.5))
    
    total = w1 + w2
    w1 = w1 / total
    w2 = w2 / total
    
    final_pred = (w1 * pred1) + (w2 * pred2)
    
    class_idx = int(np.argmax(final_pred))
    confidence = float(np.max(final_pred))
    
    return class_idx, confidence, final_pred

# File uploader
st.write("---")
uploaded_file = st.file_uploader("Choose a tomato leaf image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Uploaded Image")
        st.image(image, width=300)
    
    with col2:
        st.subheader("Prediction Results")
        
        processed_img = preprocess_image(image)
        
        with st.spinner("Analyzing image..."):
            pred_class, confidence, all_probs = predict(processed_img)
        
        disease_name = class_names[pred_class]
        st.success(f"**Prediction:** {disease_name}")
    
    st.write("---")
    st.subheader("All Class Probabilities")
    
    prob_data = []
    for i, cls in enumerate(class_names):
        prob_data.append({
            "Disease": cls,
            "Probability": f"{all_probs[i]*100:.2f}%"
        })
    
    prob_data = sorted(prob_data, key=lambda x: float(x["Probability"].replace("%", "")), reverse=True)
    
    for item in prob_data:
        st.write(f"{item['Disease']}: {item['Probability']}")

else:
    st.info("Please upload a tomato leaf image to get started")
    
    st.write("**Detectable Diseases:**")
    st.write("- Bacterial Spot")
    st.write("- Early Blight")
    st.write("- Healthy")
    st.write("- Late Blight")
    st.write("- Leaf Mold")
    st.write("- Septoria Leaf Spot")
    st.write("- Spider Mites (Two-spotted)")
