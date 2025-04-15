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

# model_path = os.path.join(os.path.dirname(__file__), "model", "aslModel(a-cAndKira)150(80-20-100).pkl")

# with open(model_path, "rb") as f:
#     model=pickle.load(f)
    
def PredFunc(landmark: List[float]):
    model = get_model()
    prediction = predictASL(model, landmark)
    print("prediction:",prediction[0])
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
    # uvicorn.run(app, host="192.168.1.11", port=8000)

