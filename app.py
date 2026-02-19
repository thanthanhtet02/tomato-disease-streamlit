import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2, EfficientNetB0

# -------------------------------------------------
# Page Config
# -------------------------------------------------
st.set_page_config(page_title="Tomato Disease Classifier", page_icon="🍅", layout="centered")
st.title("🍅 Tomato Leaf Disease Classifier (2-Model Ensemble)")
st.write("This app uses **soft voting** to combine MobileNetV2 and EfficientNetB0.")

# -------------------------------------------------
# Load JSON files
# -------------------------------------------------
@st.cache_resource
def load_class_names():
    with open("class_names.json", "r") as f:
        return json.load(f)

@st.cache_resource
def load_ensemble_weights():
    with open("ensemble_weights.json", "r") as f:
        return json.load(f)

class_names = load_class_names()
ens_w = load_ensemble_weights()

# -------------------------------------------------
# Build Transfer Architecture (matches your training architecture)
# -------------------------------------------------
def build_transfer_arch(base_model_class, preprocess_fn, model_name, num_classes=7):
    base_model = base_model_class(
        weights=None,
        include_top=False,
        input_shape=(128, 128, 3)
    )
    base_model.trainable = False

    preprocess = layers.Lambda(lambda x: preprocess_fn(x))

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
        layers.Dense(num_classes, activation="softmax")
    ], name=model_name)

    return model

# -------------------------------------------------
# Load 2 models (cached)
# -------------------------------------------------
@st.cache_resource
def load_models():
    mobilenet = build_transfer_arch(
        MobileNetV2,
        tf.keras.applications.mobilenet_v2.preprocess_input,
        "MobileNetV2_Transfer",
        num_classes=len(class_names)
    )

    effnet = build_transfer_arch(
        EfficientNetB0,
        tf.keras.applications.efficientnet.preprocess_input,
        "EfficientNetB0_Transfer",
        num_classes=len(class_names)
    )

    # filenames must match your repo exactly
    mobilenet.load_weights("mobilenetv2.weights.h5")
    effnet.load_weights("efficientnetb0.weights.h5")

    return mobilenet, effnet

mobilenet, effnet = load_models()

# -------------------------------------------------
# Image preprocessing
# IMPORTANT: keep 0..255 range (do NOT /255)
# -------------------------------------------------
def preprocess_image(img: Image.Image):
    img = img.convert("RGB").resize((128, 128))
    arr = np.array(img).astype(np.float32)  # 0..255
    arr = np.expand_dims(arr, axis=0)
    return arr

# -------------------------------------------------
# Ensemble prediction (soft voting)
# -------------------------------------------------
def ensemble_predict(x):
    p_m = mobilenet.predict(x, verbose=0)[0]
    p_e = effnet.predict(x, verbose=0)[0]

    w_m = float(ens_w.get("MobileNetV2", 0.5))
    w_e = float(ens_w.get("EfficientNetB0", 0.5))
    s = w_m + w_e
    if s == 0:
        w_m, w_e, s = 0.5, 0.5, 1.0

    # normalize weights (just in case)
    w_m /= s
    w_e /= s

    p = (w_m * p_m) + (w_e * p_e)

    pred_idx = int(np.argmax(p))
    conf = float(np.max(p))
    return pred_idx, conf, p

# -------------------------------------------------
# UI
# -------------------------------------------------
uploaded = st.file_uploader("Upload an image (jpg/png)", type=["jpg", "jpeg", "png"])

if uploaded:
    img = Image.open(uploaded)
    st.image(img, caption="Uploaded image", use_container_width=True)

    x = preprocess_image(img)

    with st.spinner("Running ensemble inference (2 models)..."):
        pred_idx, conf, probs = ensemble_predict(x)

    st.subheader("Prediction")
    st.success(f"**{class_names[pred_idx]}**")
    st.write(f"Confidence: **{conf:.3f}**")

    st.subheader("Class Probabilities")
    for name, p in sorted(zip(class_names, probs), key=lambda t: t[1], reverse=True):
        st.write(f"{name}: {float(p):.3f}")
