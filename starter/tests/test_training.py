import os
import pytest
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from starter.training.ml import data, model

@pytest.fixture(scope="session")
def data_fixture(request):
    cwd = os.getcwd()
    df = pd.read_csv(os.path.join(cwd, "starter/data/census_cleaned.csv"))


    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]

    sample_X, sample_y, encoder, lb = data.process_data(df,
                                                        categorical_features=cat_features,
                                                        label="salary",
                                                        training=True
                                                        )
    clf = joblib.load(os.path.join(cwd,"starter/model/census_clf.joblib"))
    return (sample_X, sample_y, encoder, lb), clf


def test_load_data():

    required_columns = {
        "age": pd.api.types.is_integer_dtype,
        "workclass": pd.api.types.is_string_dtype,
        "fnlgt": pd.api.types.is_integer_dtype,
        "education": pd.api.types.is_string_dtype,
        "education-num": pd.api.types.is_integer_dtype,
        "marital-status": pd.api.types.is_string_dtype,
        "occupation": pd.api.types.is_string_dtype,
        "relationship": pd.api.types.is_string_dtype,
        "race": pd.api.types.is_string_dtype,
        "sex": pd.api.types.is_string_dtype,
        "capital-gain": pd.api.types.is_integer_dtype,
        "capital-loss": pd.api.types.is_integer_dtype,
        "hours-per-week": pd.api.types.is_integer_dtype,
        "native-country": pd.api.types.is_string_dtype,
        "salary": pd.api.types.is_string_dtype
    }

    dataset = data.load_data()
    assert set(dataset.columns).issuperset(set(required_columns.keys()))

    for col, format_func in required_columns.items():
        assert format_func(dataset[col]), f"Column {col} failed test {format_func(dataset[col])}"

    assert data.load_data().shape[1] == 15, "Number of columns different than expected"
    assert data.load_data().shape[0] > 1

def test_save_model_artifact():

    reg = RandomForestClassifier()
    name = "saving_artifact_test.joblib"
    path = "./starter/tests"
    model.save_model_artifact(reg, name, path=path)
    assert os.path.exists(path), f"Artifact not saved at {path}"

def test_compute_model_metrics(data_fixture):

    dfs, clf = data_fixture
    assert dfs[0].shape[1] == 108
    preds = clf.predict(dfs[0])
    precision, recall, fbeta = model.compute_model_metrics(dfs[1], preds)

    assert isinstance(precision, float), f"Recall metric is wrong type {precision.type}"
    assert isinstance(recall, float), f"Recall metric is wrong type {recall.type}"
    assert isinstance(fbeta, float), f"Recall metric is wrong type {fbeta.type}"


def test_inferecne(data_fixture):
    dfs, clf = data_fixture
    preds = clf.predict(dfs[0])
    assert isinstance(preds, np.ndarray), f"Predictions are wrong type {type(preds)}"
    assert preds.shape[0] == dfs[0].shape[0], "Number of input samples is different from number of predictions"