from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import numpy as np
import io
import os
import tensorflow as tf
from tensorflow.keras.applications.densenet import preprocess_input

app = FastAPI(title="Alzheimer MRI Classifier API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "DenseNet121_model1.h5")

print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH, compile=False)
print("Model loaded successfully!")

CLASS_NAMES = [
    "MildDemented",
    "ModerateDemented",
    "NonDemented",
    "VeryMildDemented",
]

CLASS_TRANSLATIONS = {
    "MildDemented": {"ar": "خرف خفيف", "fr": "Démence légère", "en": "Mild Demented"},
    "ModerateDemented": {"ar": "خرف متوسط", "fr": "Démence modérée", "en": "Moderate Demented"},
    "NonDemented": {"ar": "بدون خرف", "fr": "Non dément", "en": "Non Demented"},
    "VeryMildDemented": {"ar": "خرف خفيف جداً", "fr": "Démence très légère", "en": "Very Mild Demented"},
}

@app.get("/")
def root():
    return {
        "status": "ok",
        "service": "NeuroTest Pro - Alzheimer MRI Classifier",
        "model": "DenseNet121",
        "classes": CLASS_NAMES,
        "disclaimer": "Outils éducatif uniquement. Ne remplace pas un diagnostic médical.",
    }

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        if not file.content_type or not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image")

        contents = await file.read()
        if len(contents) == 0:
            raise HTTPException(status_code=400, detail="Empty file")
        if len(contents) > 10 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="File too large (max 10MB)")

        img = Image.open(io.BytesIO(contents)).convert("RGB")
        img = img.resize((224, 224))

        img_array = np.array(img, dtype=np.float32)
        img_array = preprocess_input(img_array)
        img_array = np.expand_dims(img_array, axis=0)

        preds = model.predict(img_array, verbose=0)[0]

        predicted_index = int(np.argmax(preds))
        predicted_class = CLASS_NAMES[predicted_index]
        confidence = float(preds[predicted_index]) * 100.0

        all_probs = {}
        for i, name in enumerate(CLASS_NAMES):
            all_probs[name] = round(float(preds[i]) * 100.0, 2)

        return {
            "success": True,
            "predicted_class": predicted_class,
            "confidence": round(confidence, 2),
            "all_probabilities": all_probs,
            "translations": CLASS_TRANSLATIONS[predicted_class],
            "disclaimer_ar": "هذا التحليل تعليمي فقط ولا يُعد تشخيصاً طبياً.",
            "disclaimer_fr": "Ce résultat est éducatif uniquement.",
            "disclaimer_en": "This result is educational only.",
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")