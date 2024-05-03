import requests
import json
import pytest

# Assuming your FastAPI server is running at this URL
BASE_URL = "http://127.0.0.1:8000"

# Define test data
input_data_ea = {"hours": 5.0}
input_data_rl_electrical = {"a": 20, "b": 30, "c": 40, "d": 10, "goal_consumption": 100}
input_data_rl_transport = {"a": 20, "b": 30, "c": 40, "d": 10, "goal_consumption": 100, "fuel_type": "Petrol"}

# Define test cases
@pytest.mark.parametrize("appliance", ["Tubelight", "Air Conditioner", "Ceiling Fan", "Washing Machine", "2 Wheeler", "4 Wheeler"])
def test_predict_endpoint(appliance):
    response = requests.post(f"{BASE_URL}/predict/{appliance}", json=input_data_ea)
    assert response.status_code == 200
    assert "prediction" in response.json()

@pytest.mark.parametrize("appliance", ["Tubelight", "Air Conditioner", "Ceiling Fan", "Washing Machine"])
def test_rl_electrical_endpoint(appliance):
    response = requests.post(f"{BASE_URL}/rl/electrical/{appliance}", json=input_data_rl_electrical)
    assert response.status_code == 200
    # Add assertions for expected response

@pytest.mark.parametrize("vehicle", ["2 Wheeler", "4 Wheeler"])
def test_rl_transport_endpoint(vehicle):
    response = requests.post(f"{BASE_URL}/rl/transport/{vehicle}", json=input_data_rl_transport)
    assert response.status_code == 200
    # Add assertions for expected response
