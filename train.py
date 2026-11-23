import joblib
import os
import mlflow
from azure.ai.ml import MLClient
from azure.ai.ml.entities import Model
from azure.ai.ml.constants import AssetTypes
from azure.identity import DefaultAzureCredential
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score

def main():
    print("1. Connecting to Azure ML Workspace...")
    credential = DefaultAzureCredential()
    
    # Subscription ID for rg-mlops-project
    subscription_id = "8b30bea2-1d02-4686-b8e5-8ffc9c995c66"

    # Initialize Azure ML Client
    ml_client = MLClient(
        credential=credential,
        subscription_id=subscription_id,
        resource_group_name="rg-mlops-project",
        workspace_name="mlw-text-classification"
    )

    # Configure MLflow tracking URI for Azure ML Workspace
    ws = ml_client.workspaces.get("mlw-text-classification")
    mlflow.set_tracking_uri(ws.mlflow_tracking_uri)
    mlflow.set_experiment("text-classification-baseline")

    print("2. Loading dataset...")
    categories = ['sci.space', 'rec.autos', 'comp.graphics', 'talk.politics.mideast']
    dataset = fetch_20newsgroups(subset='all', categories=categories, remove=('headers', 'footers', 'quotes'))
    X_train, X_test, y_train, y_test = train_test_split(dataset.data, dataset.target, test_size=0.2, random_state=42)

    with mlflow.start_run():
        print("3. Training model and logging parameters...")
        max_features = 5000
        max_iter = 1000

        # Log parameters via MLflow
        mlflow.log_param("max_features", max_features)
        mlflow.log_param("max_iter", max_iter)
        mlflow.log_param("model_type", "LogisticRegression")

        # Build and train pipeline
        model_pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=max_features, stop_words='english')),
            ('classifier', LogisticRegression(max_iter=max_iter))
        ])
        model_pipeline.fit(X_train, y_train)

        print("4. Evaluating model...")
        y_pred = model_pipeline.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')

        print(f"-> Accuracy: {acc:.4f}")
        print(f"-> F1-Score: {f1:.4f}")

        # Log metrics via MLflow
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)

        print("5. Registering model in Azure Cloud Model Registry...")
        # Save model pipeline artifact locally
        os.makedirs("outputs", exist_ok=True)
        model_file_path = "outputs/model.pkl"
        joblib.dump(model_pipeline, model_file_path)

        # Register model directly into Azure ML Model Registry via Azure SDK
        azure_model = Model(
            path=model_file_path,
            type=AssetTypes.CUSTOM_MODEL,
            name="text_classifier_logistic_regression",
            description="Logistic Regression baseline text classification model"
        )
        ml_client.models.create_or_update(azure_model)
        
        print("Successfully completed!")

if __name__ == "__main__":
    main()