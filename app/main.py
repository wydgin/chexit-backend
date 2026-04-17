from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from .processor import run_tb_ensemble
import io

app = FastAPI()

# IMPORTANT: Allow your Vercel frontend to talk to this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://your-frontend.vercel.app"], 
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    return {"status": "Backend is running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read file
    img_bytes = await file.read()
    
    # Run the "Plug-and-Play" function
    prob, heatmap = run_tb_ensemble(img_bytes)
    
    # For now, just return the probability
    return {
        "probability": prob,
        "label": "Positive" if prob > 0.5 else "Negative"
    }