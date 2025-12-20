import pytest
from app.main import app


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_home_page(client):
    response = client.get("/")
    assert response.status_code == 200
    assert b"<title>Sentiment Analysis</title>" in response.data


def test_predict_page(client):
    response = client.post("/predict", data={"text": "I love this!"})
    assert response.status_code == 200
    assert b"Positive" in response.data or b"Negative" in response.data
