import os
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential

# Initialize FastAPI application
app = FastAPI(
    title="Text Classification MLOps API",
    description="REST API serving the Azure ML registered Logistic Regression text classifier.",
    version="1.0.0"
)

# Alphabetically sorted target categories from 20 Newsgroups dataset
CATEGORIES = [
    "comp.graphics",
    "rec.autos",
    "sci.space",
    "talk.politics.mideast"
]

LOCAL_MODEL_PATH = os.path.join(
    "downloaded_model", "text_classifier_logistic_regression", "model.pkl"
)

def load_model():
    """Load model locally if cached, otherwise download latest version from Azure ML Registry."""
    if not os.path.exists(LOCAL_MODEL_PATH):
        print("Local model not found. Downloading from Azure ML Model Registry...")
        credential = DefaultAzureCredential()
        ml_client = MLClient(
            credential=credential,
            subscription_id="8b30bea2-1d02-4686-b8e5-8ffc9c995c66",
            resource_group_name="rg-mlops-project",
            workspace_name="mlw-text-classification"
        )
        os.makedirs("downloaded_model", exist_ok=True)
        # Download version 2 (created by GitHub Actions CT pipeline)
        ml_client.models.download(
            name="text_classifier_logistic_regression",
            version="2",
            download_path="downloaded_model"
        )
    print(f"Loading model from: {LOCAL_MODEL_PATH}")
    return joblib.load(LOCAL_MODEL_PATH)

# Load model once at server startup
model_pipeline = load_model()

# Define request and response schemas using Pydantic
class TextRequest(BaseModel):
    text: str

class PredictionResponse(BaseModel):
    text: str
    prediction: str
    confidence: float
    probabilities: dict[str, float]

@app.get("/")
def health_check():
    """Health check endpoint to verify that the API is running."""
    return {"status": "healthy", "model": "text_classifier_logistic_regression", "version": "2"}

@app.post("/predict", response_model=PredictionResponse)
def predict(request: TextRequest):
    """Classify an incoming text string and return predicted category with confidence scores."""
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Input text must not be empty.")

    pred_idx = model_pipeline.predict([request.text])[0]
    probs = model_pipeline.predict_proba([request.text])[0]

    predicted_label = CATEGORIES[pred_idx]
    confidence = round(float(probs[pred_idx]) * 100, 2)
    prob_dict = {
        category: round(float(prob) * 100, 2)
        for category, prob in zip(CATEGORIES, probs)
    }

    return PredictionResponse(
        text=request.text,
        prediction=predicted_label,
        confidence=confidence,
        probabilities=prob_dict
    )