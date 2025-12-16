from pydantic import BaseModel
from typing import List


class CustomerData(BaseModel):
    features: List[float]


class Prediction(BaseModel):
    risk_probability: float
