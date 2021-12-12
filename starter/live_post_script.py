"""
Script to test query the deployed heroku app
"""
import json
import requests

def request_live_app():
    """
    Negative label test. Expected result for sample class <=50K is label 0.
    """
    sample_id = 0
    base_url = f"https://ml-census-api.herokuapp.com/predict/{sample_id}"

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

    r = requests.post(base_url, data=json.dumps(sample))
    print(f"Status Code: {r.status_code}")
    print("Api response:")
    print(r.json())

if __name__ == "__main__":
    request_live_app()