
from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np

app = FastAPI()

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

@app.get("/")
def read_root():
    return {"message": "ML API is runnin????"}

@app.post("/predict")
def predict(input: IrisInput):
    data = np.array([[input.sepal_length, input.sepal_width,
                      input.petal_length, input.petal_width]])
    prediction = model.predict(data)[0]
    return {"prediction": int(prediction)}
