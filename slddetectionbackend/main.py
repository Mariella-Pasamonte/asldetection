from fastapi import FastAPI, HTTPException, UploadFile, File
import mediapipe as mp
import numpy as np
import pickle
import os
import requests
from starlette.concurrency import run_in_threadpool as RIT
from PIL import Image
import io
from preprocess import predictASL, detect_hand_landmarks2D
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://asldetection-frontend.vercel.app"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_URL = f"https://mariella-asl-model-bucket.s3.ap-southeast-2.amazonaws.com/aslModel.pkl"
MODEL_PATH = "/tmp/aslModel.pkl"
model = None


def download_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading model from S3...")
        response = requests.get(MODEL_URL)
        with open(MODEL_PATH, "wb") as f:
            f.write(response.content)
        print("Model downloaded.")


def get_model():
    global model
    if model is None:
        download_model()  # Download model if not loaded already
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
    return model

def PredFunc(contents: bytes):
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    image = np.array(image)
    landmarks = detect_hand_landmarks2D(image)
    model = get_model()
    prediction = predictASL(model, landmarks)
    print("prediction:",prediction[0])
    if prediction:
        return(prediction[0])
    else: 
        raise HTTPException(status_code=500, detail="Bad image process.")

@app.post("/predict")
async def predict_landmarks(file: UploadFile = File(...)):
    try:
        contents = await file.read()

        result = await RIT(PredFunc, contents)

        return {"result": result}
    except Exception as e:
        raise HTTPException(status_code=504, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(app, host="192.168.1.11", port=8000)
