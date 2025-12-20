import os
import pickle
import mlflow
import pandas as pd
import pytest

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# -------------------- Fixtures --------------------

@pytest.fixture(scope="session")
def mlflow_setup():
    dagshub_token = os.getenv("DAGSHUB_TEST")
    if not dagshub_token:
        pytest.skip("DAGSHUB_TEST environment variable is not set")

    os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
    os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

    dagshub_url = "https://dagshub.com"
    repo_owner = "amaanrzv39"
    repo_name = "MLOps-EKS-Deployment"

    mlflow.set_tracking_uri(f"{dagshub_url}/{repo_owner}/{repo_name}.mlflow")


@pytest.fixture(scope="session")
def mlflow_client(mlflow_setup):
    return mlflow.tracking.MlflowClient()


@pytest.fixture(scope="session")
def model(mlflow_client):
    model_name = "sentiment_analysis_model"

    versions = mlflow_client.get_latest_versions(model_name)

    if not versions:
        pytest.skip("No model found in MLflow registry")

    model_uri = f"models:/{model_name}/{versions[0].version}"
    return mlflow.pyfunc.load_model(model_uri)


@pytest.fixture(scope="session")
def vectorizer():
    with open("models/vectorizer.pkl", "rb") as f:
        return pickle.load(f)


@pytest.fixture(scope="session")
def holdout_data():
    return pd.read_csv("data/processed/test_bow.csv")


# -------------------- Tests --------------------

def test_model_loaded(model):
    assert model is not None


def test_model_signature(model, vectorizer):
    input_text = "hi how are you"
    input_data = vectorizer.transform([input_text])

    input_df = pd.DataFrame(
        input_data.toarray(),
        columns=[str(i) for i in range(input_data.shape[1])]
    )

    prediction = model.predict(input_df)

    assert input_df.shape[1] == len(vectorizer.get_feature_names_out())
    assert len(prediction) == input_df.shape[0]
    assert prediction.ndim == 1


def test_model_performance(model, holdout_data):
    X_holdout = holdout_data.iloc[:, :-1]
    y_holdout = holdout_data.iloc[:, -1]

    y_pred = model.predict(X_holdout)

    accuracy = accuracy_score(y_holdout, y_pred)
    precision = precision_score(y_holdout, y_pred)
    recall = recall_score(y_holdout, y_pred)
    f1 = f1_score(y_holdout, y_pred)

    assert accuracy >= 0.40, "Accuracy below threshold"
    assert precision >= 0.40, "Precision below threshold"
    assert recall >= 0.40, "Recall below threshold"
    assert f1 >= 0.40, "F1 below threshold"
