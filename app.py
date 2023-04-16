from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import pickle

app = FastAPI()

class InputData(BaseModel):
    by_product: str
    similarity: float

def load_models():
    with open("raw_material_classifier.pkl", "rb") as f:
        raw_material_classifier = pickle.load(f)

    with open("company_classifier.pkl", "rb") as f:
        company_classifier = pickle.load(f)

    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    with open("columns.pkl", "rb") as f:
        columns = pickle.load(f)

    return raw_material_classifier, company_classifier, scaler, columns

raw_material_classifier, company_classifier, scaler, columns = load_models()

@app.post("/predict")
async def predict(input_data: InputData):
    new_data = pd.DataFrame({"By-Product": [input_data.by_product], "Similarity": [input_data.similarity]})
    new_data_encoded = pd.get_dummies(new_data, columns=["By-Product"])

    # Reorder the columns to match the order in the training data
    new_data_encoded = new_data_encoded.reindex(columns=columns, fill_value=0)
    new_data_encoded["Similarity"] = scaler.transform(new_data_encoded["Similarity"].values.reshape(-1, 1))

    raw_material_prediction = raw_material_classifier.predict(new_data_encoded)
    company_prediction = company_classifier.predict(new_data_encoded)

    return {
        "predicted_raw_material": raw_material_prediction[0],
        "predicted_company_name": company_prediction[0],
    }

