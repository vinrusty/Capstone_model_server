from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import joblib
from rl_electricity_washing_machine import run_rl_electricity_washing_machine
from rl_electricity_air_conditioner import ac_algorithm
from rl_electricity_ceiling_fan import fan_algorithm
from rl_electricity_tubelights import tubelights_algorithm
from rl_electricity_tubelights_final import run_tubelight_optimization
from rl_transport_2wheeler import rl_transport_2wheeler
from rl_transport_4wheeler import run_rl_transport_4wheeler
from rl_electricity_washing_machine_2 import optimize_washing_machine 
from rl_electricity_ceiling_fan_final import run_fan_optimization
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define request body model
class InputDataEA(BaseModel):
    hours: float

class RLInputData(BaseModel):
    ml_predict: float
# API endpoint for making predictions

class RLInputDataElectrical(BaseModel):
    a: float
    b: float
    c: float
    d: float
    goal_consumption: float

class RLInputDataTransport(BaseModel):
    a: float
    b: float
    c: float
    d: float
    goal_consumption: float
    fuel_type: str

@app.get("/")
def sayHello():
    return "Hello from the model server"

@app.post('/predict/{appliances}')
async def predict(appliances:str, input_data: InputDataEA):
    try:
        file = ''
        if appliances == "Tubelight":
            file = 'ML_models/linear_regression_model.joblib'
        elif appliances == "Air Conditioner":
            file = 'ML_models/AC_linear_regression_model.joblib'
        elif appliances == "Ceiling Fan":
            file = 'ML_models/CeilingFan_random_forest_regression_model.joblib'
        elif appliances == "Washing Machine":
            file = 'ML_models/WashingMachine_grad_boost_regression_model.joblib'
        elif appliances == "2 Wheeler":
            file = 'ML_models/2wheeler.joblib'
        elif appliances == "4 Wheeler":
            file = 'ML_models/4wheeler2.joblib'

        # Get input data from the frontend
        model = joblib.load(file)

        hours = input_data.hours
        print(hours)
        # Convert input data to numpy array and scale using the same scaler
        input_array = np.array([hours])

        # Make prediction using the loaded model
        prediction = model.predict([input_array])

        # Return the prediction as JSON response
        return {"prediction": prediction.flatten().tolist()}

    except Exception as e:
        print(str(e))
        raise HTTPException(status_code=500, detail=str(e))




@app.post("/rl/electrical/{appliances}")
async def rl_appliance_predict(appliances: str, input_data: RLInputDataElectrical):
    try:
        a = input_data.a/100
        b = input_data.b/100
        c = input_data.c/100
        d = input_data.d/100
        goal = input_data.goal_consumption
        if appliances == "Tubelight":
            # return tubelights_algorithm(a,b,c,d,goal)
            return run_tubelight_optimization(a,b,c,d,goal)
        elif appliances == "Air Conditioner":
            return ac_algorithm(a,b,c,d,goal)
        elif appliances == "Ceiling Fan":
            # return fan_algorithm(a,b,c,d,goal)
            return run_fan_optimization(a,b,c,d,goal_consumption=goal)
        elif appliances == "Washing Machine":
            return optimize_washing_machine([a, b, c, d], goal/10)
        
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))


        
@app.post("/rl/transport/{vehicle}")
async def rl_transport_predict(vehicle: str, input_data: RLInputDataTransport):
    try:
        a = input_data.a/100
        b = input_data.b/100
        c = input_data.c/100
        d = input_data.d/100
        goal = input_data.goal_consumption
        fuel = input_data.fuel_type

        print(input_data)

        if vehicle == "2 Wheeler":
            return rl_transport_2wheeler([a,b,c,d],goal)
            pass
        if vehicle == "4 Wheeler":
            return run_rl_transport_4wheeler([a,b,c,d],goal, fuel, 0.2)
    except Exception as e:
        
        raise HTTPException(status_code=500, detail=str(e))

