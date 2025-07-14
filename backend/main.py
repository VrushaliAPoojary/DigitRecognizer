from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import joblib
import numpy as np
from PIL import Image
import io
import os
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Prevent GUI backend error

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static HTML
app.mount("/frontend", StaticFiles(directory="frontend"), name="frontend")

# Load the trained model
model_path = "backend/model/digit_model.pkl"
model = joblib.load(model_path)

# Log file
log_file = "prediction_history.txt"
if not os.path.exists(log_file):
    with open(log_file, "w") as f:
        f.write("timestamp,prediction\n")

@app.get("/", response_class=HTMLResponse)
async def serve_upload_form():
    return FileResponse("frontend/upload.html")

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # Load and preprocess image
    image = Image.open(io.BytesIO(await file.read())).convert("L")
    image = image.resize((8, 8))
    image = np.array(image) / 255.0 * 16
    image = image.reshape(1, -1)

    # Predict
    prediction = model.predict(image)[0]

    # Log prediction
    with open(log_file, "a") as f:
        f.write(f"{datetime.now().isoformat(timespec='seconds')},{int(prediction)}\n")

    # Save debug image for inspection
    plt.imshow(image.reshape(8, 8), cmap='gray')
    plt.title(f"Predicted: {prediction}")
    plt.axis('off')
    plt.savefig("debug_image.png")  # Saves to root project folder

    return JSONResponse(content={"prediction": int(prediction)})
