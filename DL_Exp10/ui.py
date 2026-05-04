import streamlit as st
import onnxruntime as ort
import numpy as np
from PIL import Image

st.title("🧠 Digit Classifier (ONNX Model)")

# Load model
session = ort.InferenceSession("model.onnx")

def preprocess(image):
    image = image.convert("L").resize((28, 28))
    image = np.array(image).astype(np.float32)

    # invert colors
    image = 255 - image

    # normalize
    image = image / 255.0
    image = (image - 0.1307) / 0.3081

    image = np.expand_dims(image, axis=(0,1))
    return image

# Upload file
uploaded_file = st.file_uploader("Upload a digit image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    st.image(image, caption="Uploaded Image", use_column_width=True)

    input_data = preprocess(image)
    output = session.run(None, {"input": input_data})

    prediction = int(np.argmax(output[0]))

    st.success(f"Predicted Digit: {prediction}")