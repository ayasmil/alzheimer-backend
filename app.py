from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import numpy as np
import io
import os
from ai_edge_litert.interpreter import Interpreter

# ✅ Rate Limiting
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

app = FastAPI(title="Alzheimer MRI Classifier API")

# ✅ Rate Limiter — 15 طلب / دقيقة لكل IP
# يستخدم default_limits + Middleware ليعمل قبل FastAPI validation
limiter = Limiter(
    key_func=get_remote_address,
    default_limits=["15/minute"],
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "DenseNet121_model1.tflite")

print("Loading TFLite model...")
interpreter = Interpreter(model_path=MODEL_PATH, num_threads=2)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print("TFLite model loaded!")

IMG_SIZE = 224

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


def preprocess_densenet(img):
    x = np.array(img, dtype=np.float32)
    x = x / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    x = (x - mean) / std
    return x


@app.get("/")
def root():
    return {
        "status": "ok",
        "service": "NeuroTest Pro - Alzheimer MRI Classifier",
        "model": "DenseNet121",
        "classes": CLASS_NAMES,
        "disclaimer": "Outil éducatif uniquement. Ne remplace pas un diagnostic médical.",
    }


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/predict")
async def predict(request: Request, file: UploadFile = File(...)):
    try:
        if not file.content_type or not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image")

        contents = await file.read()
        if len(contents) == 0:
            raise HTTPException(status_code=400, detail="Empty file")
        if len(contents) > 10 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="File too large (max 10MB)")

        img = Image.open(io.BytesIO(contents)).convert("RGB")
        img = img.resize((IMG_SIZE, IMG_SIZE))

        img_array = preprocess_densenet(img)
        img_array = np.expand_dims(img_array, axis=0).astype(np.float32)

        interpreter.set_tensor(input_details[0]["index"], img_array)
        interpreter.invoke()
        preds = interpreter.get_tensor(output_details[0]["index"])[0]

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
