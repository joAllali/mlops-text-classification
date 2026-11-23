import os
import joblib
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential

def main():
    print("1. Connecting to Azure ML Workspace...")
    credential = DefaultAzureCredential()
    subscription_id = "8b30bea2-1d02-4686-b8e5-8ffc9c995c66"

    ml_client = MLClient(
        credential=credential,
        subscription_id=subscription_id,
        resource_group_name="rg-mlops-project",
        workspace_name="mlw-text-classification"
    )

    print("2. Downloading registered model (v1) from Azure Model Registry...")
    download_dir = "downloaded_model"
    os.makedirs(download_dir, exist_ok=True)

    # Download version 1 of the registered model into a local folder
    ml_client.models.download(
        name="text_classifier_logistic_regression",
        version="1",
        download_path=download_dir
    )

    # Locate the downloaded model.pkl file
    model_path = os.path.join(download_dir, "text_classifier_logistic_regression", "model.pkl")
    print(f"3. Loading model from: {model_path}")
    model_pipeline = joblib.load(model_path)

    # Scikit-Learn's fetch_20newsgroups sorts target categories alphabetically
    categories = [
        "comp.graphics",
        "rec.autos",
        "sci.space",
        "talk.politics.mideast"
    ]

    # Sample unseen texts to test the classifier
    sample_texts = [
        "NASA announced a new rocket launch to explore the outer planets and deep space.",
        "The new V8 engine delivers incredible horsepower and improved fuel efficiency on the highway.",
        "OpenGL rendering pipelines require fast GPU shaders for real-time 3D graphics.",
        "Diplomatic negotiations in the Middle East continue to focus on regional stability."
    ]

    print("4. Running predictions on unseen text samples...\n")
    predictions = model_pipeline.predict(sample_texts)
    probabilities = model_pipeline.predict_proba(sample_texts)

    for text, pred_idx, probs in zip(sample_texts, predictions, probabilities):
        predicted_label = categories[pred_idx]
        confidence = probs[pred_idx] * 100
        print(f"Text:       \"{text}\"")
        print(f"Prediction: {predicted_label} (Confidence: {confidence:.2f}%)\n")

if __name__ == "__main__":
    main()