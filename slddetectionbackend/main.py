from fastapi import FastAPI, HTTPException
import pickle
import os
import requests
from starlette.concurrency import run_in_threadpool as RIT
from preprocess import predictASL
from fastapi.middleware.cors import CORSMiddleware
from typing import List

IMAGEDIR = "images/"

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_URL63 = f"https://mariella-asl-model-bucket.s3.ap-southeast-2.amazonaws.com/aslModel63(complete).pkl"
MODEL_PATH63 = "/tmp/aslModel63(complete).pkl"
MODEL_URL126 = f"https://mariella-asl-model-bucket.s3.ap-southeast-2.amazonaws.com/aslModel126(complete).pkl"
MODEL_PATH126 = "/tmp/aslModel126(complete).pkl"
model63 = None
model126 = None


def download_model(MODEL_PATH,MODEL_URL):
    if not os.path.exists(MODEL_PATH):
        print("Downloading model from S3...")
        response = requests.get(MODEL_URL)
        with open(MODEL_PATH, "wb") as f:
            f.write(response.content)
        print("Model downloaded.")

def get_model(MODEL_PATH63, MODEL_URL63, MODEL_PATH126, MODEL_URL126):
    global model63, model126
    if model63 is None or model126 is None:
        download_model(MODEL_PATH63, MODEL_URL63)  # Download model if not loaded already
        download_model(MODEL_PATH126, MODEL_URL126)
        with open(MODEL_PATH63, "rb") as f:
            model63 = pickle.load(f)
        with open(MODEL_PATH126, "rb") as f:
            model63 = pickle.load(f)
    return model63, model126
    
def PredFunc(landmark: List[float]):
    model63, model126 = get_model(MODEL_PATH63, MODEL_URL63, MODEL_PATH126, MODEL_URL126)
    prediction = predictASL(model63, model126, landmark)
    if prediction:
        return(prediction[0])
    else: 
        raise HTTPException(status_code=500, detail="Bad image process.")

@app.post("/predict")
async def predict_landmarks(landmark:List[float]):
    try:
        result = await RIT(PredFunc, landmark)

        return {"result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(app, host="0.0.0.0", port=8000)

