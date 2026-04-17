from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from .processor import run_tb_ensemble
import cv2
import base64

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://chexit.vercel.app"], # For MVP, allow all; restrict to your Vercel URL later
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    img_bytes = await file.read()
    
    # Run the Averaging Ensemble
    prob, heatmap = run_tb_ensemble(img_bytes)
    
    # Encode heatmap to Base64
    _, buffer = cv2.imencode('.jpg', (heatmap * 255).astype('uint8'))
    heatmap_base64 = base64.b64encode(buffer).decode('utf-8')

    return {
        "probability": round(prob, 4),
        "label": "Tuberculosis" if prob > 0.5 else "Normal",
        "heatmap": f"data:image/jpeg;base64,{heatmap_base64}"
    }