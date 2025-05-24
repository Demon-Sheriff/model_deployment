import pytest 
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app import app  # now it should work

@pytest.fixture
def client():
    app.testing = True
    with app.test_client() as client:
        yield client

def test_predict_valid_input(client):
    pixels = [0]*784
    response = client.post('/predict', json={'pixels': pixels})

    assert response.status_code == 200
    data = response.get_json()

    assert 'prediction' in data