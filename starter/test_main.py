from fastapi.testclient import TestClient
from starter.main import app
import requests
import json
import ast

client = TestClient(app)


def test_root_get():
    r = client.get("/")
    assert r.status_code == 200, f"Failed root get with status code {r.status_code}"
    assert r.json()["message"] == "Welcome to the Census Income Prediction API"

def test_predict_negative_label():
    """
    Negative label test. Expected result for sample class <=50K is label 0.
    """
    sample_id = 0
    sample = {"age": 39,
              "workclass": "State-gov",
              "fnlgt": 77516,
              "education": "Bachelors",
              "education-num": 13,
              "marital-status": "Never-married",
              "occupation": "Adm-clerical",
              "relationship": "Not-in-family",
              "race": "White",
              "sex": "Male",
              "capital-gain": 2174,
              "capital-loss": 0,
              "hours-per-week": 40,
              "native-country": "United-States"}

    r = client.post(f"/predict/{sample_id}", json=sample)
    assert r.status_code == 200
    assert ast.literal_eval(r.json()["pred"])[0] == 0 #expect negative case, salary <=50K


def test_predict_positive_label():
    """
    Positive label test. Expected result for sample class >50K is label 1.
    """
    sample_id = 7
    sample = {'age': 31,
              'workclass': 'Private',
              'fnlgt': 45781,
              'education': 'Masters',
              'education-num': 14,
              'marital-status': 'Never-married',
              'occupation': 'Prof-specialty',
              'relationship': 'Not-in-family',
              'race': 'White',
              'sex': 'Female',
              'capital-gain': 14084,
              'capital-loss': 0,
              'hours-per-week': 50,
              'native-country': 'United-States'}

    r = client.post(f"/predict/{sample_id}", json=sample)
    assert r.status_code == 200
    assert ast.literal_eval(r.json()["pred"])[0] == 1 #expect negative case, salary <=50K