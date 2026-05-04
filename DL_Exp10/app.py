from fastapi import FastAPI, File, UploadFile
import onnxruntime as ort
import numpy as np
from PIL import Image
import io

app = FastAPI()

session = ort.InferenceSession("model.onnx")

def preprocess(image):
    image = image.convert("L").resize((28, 28))
    image = np.array(image).astype(np.float32)

    # 🔥 FIX: invert (MNIST style)
    image = 255 - image

    # normalize
    image = image / 255.0
    image = (image - 0.1307) / 0.3081

    image = np.expand_dims(image, axis=(0,1))
    return image

@app.get("/")
def home():
    return {"message": "ONNX Digit Classifier Running"}

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(await file.read()))
    input_data = preprocess(image)

    output = session.run(None, {"input": input_data})
    prediction = int(np.argmax(output[0]))

    return {"digit": prediction}