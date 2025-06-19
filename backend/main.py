from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles  # Import this

import joblib
import numpy as np
from PIL import Image
import io


app = FastAPI()  # ← FIRST define `app`

# Now it's safe to mount frontend
app.mount("/frontend", StaticFiles(directory="frontend", html=True), name="frontend")

# Prediction API
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("L").resize((8, 8))
    image = np.array(image)
    image = 16 - (image / 16).astype(int)
    image = image.reshape(1, -1)

    model = joblib.load("backend/model/digit_model.pkl")
    prediction = model.predict(image)[0]
    return JSONResponse(content={"prediction": int(prediction)})
