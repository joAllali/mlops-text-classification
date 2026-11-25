# End-to-End MLOps Text Classification Pipeline (Azure ML, MLflow & FastAPI)

An industry-style, cost-optimized MLOps pipeline for multi-class text classification using **Scikit-Learn**, **MLflow**, **Azure Machine Learning**, **FastAPI**, **Docker**, and **GitHub Actions**.

## Architecture Overview

This project implements a **hybrid MLOps architecture** designed to eliminate cloud compute costs while maintaining enterprise-grade experiment tracking, model versioning, and automated Continuous Training (CT):

```text
+-------------------+       git push        +-----------------------------------+
|  Local Dev / IDE  | --------------------> |          GitHub Actions           |
+-------------------+                       |  1. Run train.py (CT Pipeline)    |
                                            |  2. Build & Validate Docker Image |
                                            +-----------------+-----------------+
                                                              |
                                      Logs Metrics &          | Registers Model
                                      Parameters (MLflow)     | Artifact (Azure SDK)
                                                              v
+-------------------+   Downloads Model     +-----------------------------------+
| FastAPI + Docker  | <-------------------- |      Azure Machine Learning       |
|  (REST Inference) |     (Versioned)       |  - Experiment Tracking (MLflow)   |
+-------------------+                       |  - Centralized Model Registry     |
                                            +-----------------------------------+
```

### Key Engineering Decisions
* **Zero Cloud Compute Cost:** Model training and CI/CD execution run locally or on GitHub-hosted Ubuntu runners, while Azure Machine Learning serves strictly as the centralized tracking backend and Model Registry.
* **Hybrid Azure SDK + MLflow Integration:** MLflow logs hyperparameters and evaluation metrics (`Accuracy`, `F1-Score`), while the native `azure-ai-ml` SDK handles reliable model artifact registration (`v1`, `v2`, `v3`, ...).
* **Automated Continuous Training (CT):** Every push to `main` triggers an automated retraining workflow via GitHub Actions authenticated securely via an Azure Service Principal, followed by an automated Docker container build check.

---

## Tech Stack

* **Machine Learning & Data:** Python 3.11, Scikit-Learn (TF-IDF Vectorizer + Logistic Regression), Pandas, Joblib
* **MLOps & Cloud:** Azure Machine Learning (`azure-ai-ml`, `azure-identity`), MLflow (`azureml-mlflow`)
* **Model Serving & Containerization:** FastAPI, Uvicorn, Pydantic, Docker
* **CI/CD & Automation:** GitHub Actions

---

## Project Structure

```text
mlops-text-classification/
├── .github/
│   └── workflows/
│       └── mlops-pipeline.yml   # CI/CD workflow for automated training & Docker build
├── app.py                       # FastAPI REST service for real-time inference
├── train.py                     # Training pipeline with MLflow tracking & Azure registration
├── score.py                     # Standalone batch inference script pulling from Azure Registry
├── Dockerfile                   # Production container definition for FastAPI serving
├── .dockerignore                # Excludes local environments and artifacts from image
├── requirements.txt             # Reproducible Python dependencies
└── README.md                    # Project documentation
```

---

## Model & Dataset

* **Dataset:** [20 Newsgroups](https://scikit-learn.org/stable/datasets/real_world.html#newsgroups-dataset) (4 target categories: `comp.graphics`, `rec.autos`, `sci.space`, `talk.politics.mideast`), split 80/20 into training and test sets with metadata headers/footers stripped to prevent data leakage.
* **Pipeline:** `TfidfVectorizer(max_features=5000, stop_words='english')` -> `LogisticRegression(max_iter=1000)`
* **Baseline Performance:**
  * **Accuracy:** `0.8830`
  * **Weighted F1-Score:** `0.8830`

---

## Getting Started (Local Setup)

### 1. Clone Repository & Create Virtual Environment
```bash
git clone [https://github.com/joAllali/mlops-text-classification.git](https://github.com/joAllali/mlops-text-classification.git)
cd mlops-text-classification

python -m venv venv
# Windows PowerShell:
.\venv\Scripts\Activate.ps1
# Linux/macOS:
# source venv/bin/activate

pip install -r requirements.txt
```

### 2. Authenticate with Azure
Ensure you have an active Azure ML Workspace (`mlw-text-classification` in resource group `rg-mlops-project`) and log in via Azure CLI:
```bash
az login
```

### 3. Train & Register Model
Run the training pipeline locally and stream metrics/artifacts to Azure ML:
```bash
python train.py
```

### 4. Run Batch Inference (`score.py`)
Pull the registered model from Azure ML Model Registry and classify unseen sample texts:
```bash
python score.py
```

### 5. Start FastAPI Inference Server
Launch the REST API locally:
```bash
uvicorn app:app --reload
```
Open the interactive Swagger UI at **`http://127.0.0.1:8000/docs`** and send a `POST` request to `/predict`:
```json
{
  "text": "NASA announced a new rocket launch to explore the outer planets and deep space."
}
```

---

## CI/CD Setup (GitHub Actions)

To enable automated retraining on push, configure the following repository secrets under **Settings > Secrets and variables > Actions**:
* `AZURE_CLIENT_ID` – Service Principal App ID
* `AZURE_CLIENT_SECRET` – Service Principal Password
* `AZURE_TENANT_ID` – Azure Active Directory Tenant ID