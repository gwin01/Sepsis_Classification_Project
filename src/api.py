from fastapi import FastAPI
import os, uvicorn
from typing import List, Literal
from pydantic import BaseModel
import joblib


#CONFIG
app = FastAPI(
title = "Sepsis Classification Web App",
version = "0.0.1",
description = "This App allows users to predict whether a patient would be diagnosed with Septsis or not"

)

#API INPUT
class Input(BaseModel):
    Plasma_Glucose: int
    Blood_Work_Result1: int
    Blood_Pressure: int
    Blood_Work_Result2: int
    Blood_Work_Result3: int
    Body_Mass_Index: float
    Blood_Work_Result4: float
    Age: int
    Insurance: int
    


#ENDPOI
@app.get("/Status")
async def root():
    return {"message": "Online"}

@app.post("/predict/")
def predict(input: Input):
    model = joblib.load("path_to_your_model.pkl")
    features = [input.Plasma_Glucose, input.Blood_Work_Result1, input.Blood_Pressure,input.Blood_Work_Result2,input.Blood_Work_Result2]
    prediction = model.predict([features])[0]
    return {"prediction": prediction

if __name__ == '__main__':
    uvicorn.run('api:app', reload =True)