import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load Model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("e_waste_classifier_final.h5")
    return model

model = load_model()

# Classes (change according to your model)
CLASS_NAMES = ["Battery", "Mobile", "Laptop", "Charger", "Others"]


# Preprocess Image
def preprocess_image(image):
    image = image.resize((224, 224))  # Change size if different
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image


# UI
st.title("♻️ E-Waste Image Classifier")
st.write("Upload an image of an e-waste item to classify it")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Show image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Predict
    img_array = preprocess_image(image)
    predictions = model.predict(img_array)
    score = predictions[0]
    pred_class = CLASS_NAMES[np.argmax(score)]
    confidence = np.max(score) * 100

    st.subheader(f"Prediction: **{pred_class}**")
    st.write(f"Confidence: **{confidence:.2f}%**")

