# Put the code for your API here.
from typing import Optional
from fastapi import FastAPI, Query
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel

import os
import joblib
import subprocess
from pandas import DataFrame

from starter.training.ml.model import inference
from starter.training.ml.data import process_data

app = FastAPI()

class Sample(BaseModel):
    age: Optional[int] = None
    workclass: Optional[str] = None
    fnlgt: Optional[int] = None
    education: Optional[str] = None
    education_num: Optional[str] = Query(None, alias="education-num")
    marital_status: Optional[str] = Query(None, alias="marital-status")
    occupation: Optional[str] = None
    relationship: Optional[str] = None
    race: Optional[str] = None
    sex: Optional[str] = None
    capital_gain: Optional[int] = Query(None, alias="capital-gain")
    capital_loss: Optional[int] = Query(None, alias="capital-loss")
    hours_per_week: Optional[int] = Query(None, alias="hours-per-week")
    native_country: Optional[str] = Query(None, alias="native-country")

    class Config:
        schema_extra = {
            "example": {
                "age":28,
                "workclass": "Private",
                "fnlgt": 105817,
                "education":"11th",
                "education-num":7,
                "marital-status": "Married-civ-spouse",
                "occupation": "Exec-managerial",
                "relationship":"Husband",
                "race":"White",
                "sex":"Male",
                "capital-gain":0,
                "capital-loss":0,
                "hours-per-week":50,
                "native-country":"United-States"}
            }

@app.get("/")
async def root():
    return {"message": "Welcome to the Census Income Prediction API"}

@app.post("/predict/{sample_id}")
async def create_item(sample_id: int, sample: Sample):

    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country"
    ]
    cwd = os.getcwd()
    model = joblib.load(os.path.join(cwd, "starter/model/census_clf.joblib"))
    encoder = joblib.load(os.path.join(cwd, "starter/model/census_encoder.joblib"))

    sample_dict = jsonable_encoder(sample)
    data = DataFrame(sample_dict, index=[0])

    X, _, _, _ = process_data(data,
                          categorical_features=cat_features,
                          training=False,
                          encoder=encoder)

    pred = inference(model, X)
    return {"sample_id": sample_id, "sample": sample, "pred": str(pred)}


if "DYNO" in os.environ and os.path.isdir(".dvc"):
    os.system("aws configure --profile 'udacity'")
    os.system("dvc remote add remotestores3 ${{env.DVC_REMOTE_REPOSITORY}}")
    os.system("dvc config core.no_scm true core.remote remotestores3")

    # if os.system("dvc pull") != 0:
    #     exit("dvc pull failed")
    dvc_output = subprocess.run(["dvc", "pull"], capture_output=True, text=True)
    print(dvc_output.stdout)
    print(dvc_output.stderr)
    if dvc_output.returncode != 0:
        print("dvc pull failed")
    # os.system("rm -r .dvc .apt/usr/lib/dvc")