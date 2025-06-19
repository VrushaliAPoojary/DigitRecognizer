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
    image = Image.open(io.BytesIO(await file.read())).convert("L")
    image = image.resize((28, 28))
    image = np.array(image) / 255.0
    image = image.reshape(1, -1)
    


    prediction = model.predict(image)[0]

    with open(log_file, "a") as f:
        f.write(f"{datetime.now().isoformat(timespec='seconds')},{int(prediction)}\n")

    return JSONResponse(content={"prediction": int(prediction)})
