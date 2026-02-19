import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json

# -------------------------------------------------
# Page Configuration
# -------------------------------------------------
st.set_page_config(
    page_title="Tomato Disease Classifier",
    page_icon="🍅",
    layout="centered"
)

st.title("🍅 Tomato Leaf Disease Classifier")
st.write("Upload a tomato leaf image to predict the disease.")

# -------------------------------------------------
# Load Model (cached to avoid reloading)
# -------------------------------------------------
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("mobilenetv2.keras")
    return model

@st.cache_resource
def load_class_names():
    with open("class_names.json", "r") as f:
        class_names = json.load(f)
    return class_names

model = load_model()
class_names = load_class_names()

# -------------------------------------------------
# Image Preprocessing
# -------------------------------------------------
def preprocess_image(image):
    image = image.convert("RGB")
    image = image.resize((128, 128))
    img_array = np.array(image).astype(np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# -------------------------------------------------
# File Upload
# -------------------------------------------------
uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    with st.spinner("Predicting..."):
        processed_image = preprocess_image(image)
        predictions = model.predict(processed_image, verbose=0)

        predicted_index = np.argmax(predictions)
        confidence = float(np.max(predictions))

        st.subheader("Prediction Result")
        st.success(f"**Class:** {class_names[predicted_index]}")
        st.write(f"Confidence: **{confidence:.3f}**")

        st.subheader("All Class Probabilities")
        probs = predictions[0]
        for i, prob in enumerate(probs):
            st.write(f"{class_names[i]}: {prob:.3f}")
