from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import joblib
import numpy as np
from PIL import Image
import io
import os

app = FastAPI()

# Enable CORS for frontend development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Set specific origins if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static folder
app.mount("/frontend", StaticFiles(directory="frontend"), name="frontend")

# Load the trained model
model = joblib.load("backend/model/digit_model.pkl")

# Serve upload.html at root URL
@app.get("/", response_class=HTMLResponse)
async def serve_upload_form():
    return FileResponse("frontend/upload.html")

# Handle image prediction
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(await file.read())).convert("L")
    image = image.resize((8, 8))  # Resize to match model input
    image = np.array(image) / 16.0  # Normalize to match training
    image = image.reshape(1, -1)

    prediction = model.predict(image)[0]
    return JSONResponse(content={"prediction": int(prediction)})
