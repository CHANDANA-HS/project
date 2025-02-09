import streamlit as st
import numpy as np
import tensorflow as tf
import cv2
from PIL import Image

# Load the trained model
model = tf.keras.models.load_model("sign_language_model.h5")

# Function to preprocess the image
def preprocess_image(image):
    image = image.convert("L")  # Convert to grayscale
    image = image.resize((28, 28))  # Resize to 28x28 pixels
    image = np.array(image) / 255.0  # Normalize (0 to 1)
    image = image.reshape(1, 28, 28, 1)  # Reshape for model input
    return image

# Streamlit UI
st.title("🖐️ Sign Language Detection")
st.write("Upload an image of a sign language letter to predict.")

# Upload Image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)  # Open the image
    st.image(image, caption="Uploaded Image", use_column_width=True)  # Show image

    # Preprocess image
    processed_image = preprocess_image(image)

    # Predict using the model
    prediction = model.predict(processed_image)
    predicted_label = np.argmax(prediction)  # Get the predicted label

    # Show result
    st.write(f"**Predicted Sign: {chr(65 + predicted_label)}**")  # Convert to A-Z
