from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import numpy as np
import io
import os
from ai_edge_litert.interpreter import Interpreter

app = FastAPI(title="Alzheimer MRI Classifier API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model_v2.tflite")

print("Loading TFLite model...")
interpreter = Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
print("TFLite model loaded!")

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

CLASS_NAMES = ["MildDemented", "ModerateDemented", "NonDemented", "VeryMildDemented"]

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
        "model": "MobileNetV2 TFLite",
        "classes": CLASS_NAMES,
    }

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        if len(contents) == 0:
            raise HTTPException(status_code=400, detail="Empty file")
        if len(contents) > 10 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="File too large (max 10MB)")

        img = Image.open(io.BytesIO(contents)).convert("RGB")
        img = img.resize((128, 128))

        img_array = np.array(img, dtype=np.float32)
        img_array = (img_array / 127.5) - 1.0
        img_array = np.expand_dims(img_array, axis=0)

        interpreter.set_tensor(input_details[0]['index'], img_array)
        interpreter.invoke()
        preds = interpreter.get_tensor(output_details[0]['index'])[0]

        predicted_index = int(np.argmax(preds))
        predicted_class = CLASS_NAMES[predicted_index]
        confidence = float(preds[predicted_index]) * 100.0

        all_probs = {name: round(float(preds[i]) * 100.0, 2) for i, name in enumerate(CLASS_NAMES)}

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

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
