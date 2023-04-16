{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "5d9fb6b2",
   "metadata": {},
   "outputs": [],
   "source": [
    "from fastapi import FastAPI, HTTPException\n",
    "from pydantic import BaseModel\n",
    "import pandas as pd\n",
    "import pickle\n",
    "\n",
    "app = FastAPI()\n",
    "\n",
    "class InputData(BaseModel):\n",
    "    by_product: str\n",
    "    similarity: float\n",
    "\n",
    "def load_models():\n",
    "    with open(\"raw_material_classifier.pkl\", \"rb\") as f:\n",
    "        raw_material_classifier = pickle.load(f)\n",
    "\n",
    "    with open(\"company_classifier.pkl\", \"rb\") as f:\n",
    "        company_classifier = pickle.load(f)\n",
    "\n",
    "    with open(\"scaler.pkl\", \"rb\") as f:\n",
    "        scaler = pickle.load(f)\n",
    "\n",
    "    with open(\"columns.pkl\", \"rb\") as f:\n",
    "        columns = pickle.load(f)\n",
    "\n",
    "    return raw_material_classifier, company_classifier, scaler, columns\n",
    "\n",
    "raw_material_classifier, company_classifier, scaler, columns = load_models()\n",
    "\n",
    "@app.post(\"/predict\")\n",
    "async def predict(input_data: InputData):\n",
    "    new_data = pd.DataFrame({\"By-Product\": [input_data.by_product], \"Similarity\": [input_data.similarity]})\n",
    "    new_data_encoded = pd.get_dummies(new_data, columns=[\"By-Product\"])\n",
    "\n",
    "    # Reorder the columns to match the order in the training data\n",
    "    new_data_encoded = new_data_encoded.reindex(columns=columns, fill_value=0)\n",
    "    new_data_encoded[\"Similarity\"] = scaler.transform(new_data_encoded[\"Similarity\"].values.reshape(-1, 1))\n",
    "\n",
    "    raw_material_prediction = raw_material_classifier.predict(new_data_encoded)\n",
    "    company_prediction = company_classifier.predict(new_data_encoded)\n",
    "\n",
    "    return {\n",
    "        \"predicted_raw_material\": raw_material_prediction[0],\n",
    "        \"predicted_company_name\": company_prediction[0],\n",
    "    }\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "9cccf280",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10.10"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
