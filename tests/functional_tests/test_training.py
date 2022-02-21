import os.path as op
import sys

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import r2_score

HERE = op.dirname(op.abspath("__file__"))
test_path = op.join(HERE, "..", "..", "src", "housing")
sys.path.append(test_path)


@pytest.fixture
def test_df():
    data = pd.read_csv("../../data/raw/housing.csv").sample(20)
    return data


@pytest.fixture
def full_pipeline():
    pipeline_used = joblib.load("../../artifacts/pipeline.pkl")
    return pipeline_used


@pytest.fixture
def model():
    model_used = joblib.load("../../artifacts/model.pkl")
    return model_used


def test_training(test_df, full_pipeline, model):
    test_x = test_df.drop("median_house_value", axis=1)
    test_y = test_df["median_house_value"].copy()

    test_x_prepared = full_pipeline.transform(test_x)
    test_y_pred = model.predict(test_x_prepared)

    assert isinstance(test_x_prepared, np.ndarray)
    assert isinstance(test_y_pred, np.ndarray)
    assert r2_score(test_y, test_y_pred) > 0.85
