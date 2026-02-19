import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2

st.set_page_config(page_title="Tomato Disease Classifier", page_icon="🍅", layout="centered")
st.title("🍅 Tomato Leaf Disease Classifier (MobileNetV2)")
st.write("Upload a tomato leaf image to predict the disease class.")

@st.cache_resource
def load_class_names():
    with open("class_names.json", "r") as f:
        return json.load(f)

@st.cache_resource
def load_model():
    # Build the same architecture as your build_transfer_model()
    base_model = MobileNetV2(
        weights=None,
        include_top=False,
        input_shape=(128, 128, 3)
    )
    base_model.trainable = False

    # Same preprocessing used during training (expects pixel range 0..255)
    preprocess = layers.Lambda(lambda x: tf.keras.applications.mobilenet_v2.preprocess_input(x))

    model = models.Sequential([
        layers.Input(shape=(128, 128, 3)),
        preprocess,
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.BatchNormalization(),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(7, activation="softmax")
    ], name="MobileNetV2_Transfer")

    # IMPORTANT: match your GitHub file name exactly
    model.load_weights("mobilenetv2.weights.h5")
    return model

class_names = load_class_names()
model = load_model()

def preprocess_image(img: Image.Image):
    img = img.convert("RGB").resize((128, 128))
    arr = np.array(img).astype(np.float32)  # NO /255.0 (preprocess_input expects 0..255)
    arr = np.expand_dims(arr, axis=0)
    return arr

uploaded = st.file_uploader("Upload an image (jpg/png)", type=["jpg", "jpeg", "png"])

if uploaded:
    img = Image.open(uploaded)
    st.image(img, caption="Uploaded image", use_container_width=True)

    x = preprocess_image(img)

    with st.spinner("Predicting..."):
        probs = model.predict(x, verbose=0)[0]
        pred_idx = int(np.argmax(probs))
        conf = float(np.max(probs))

    st.subheader("Prediction")
    st.success(f"**{class_names[pred_idx]}**")
    st.write(f"Confidence: **{conf:.3f}**")

    st.subheader("Class Probabilities")
    for name, p in sorted(zip(class_names, probs), key=lambda t: t[1], reverse=True):
        st.write(f"{name}: {float(p):.3f}")
