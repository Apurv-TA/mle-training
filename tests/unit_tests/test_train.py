import os.path as op
import sys
import numpy as np

import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor


HERE = op.dirname(op.abspath("__file__"))
test_path = op.join(HERE, "..", "..", "src", "housing")
sys.path.append(test_path)
import train


@pytest.fixture
def train_df():
    train_data = pd.read_csv("../../data/processed/train.csv").sample(2000)
    return train_data


@pytest.fixture
def model_param(train_df):
    models = {
        "Linear_regres": LinearRegression(),
        "Decision_tree": DecisionTreeRegressor(),
        "Random_forest": RandomForestRegressor(),
    }
    x, y, pipe, results = train.basic_modeling(train_df, models)

    return x, y, pipe, models, results


def test_basic_modeling(model_param):
    x, y, _, models, results = model_param

    assert isinstance(x, np.ndarray)
    assert isinstance(y, pd.Series)
    assert isinstance(results, dict)

    assert models.keys() == results.keys()
    assert (set(map(type, results.values())) == {np.float64})


def test_model_search(model_param):
    x, y, _, models, results = model_param
    model = models[max(results)]

    grid_result = train.model_search(x, y, model)

    assert isinstance(grid_result.best_score_, float)
    assert grid_result.best_score_ >= max(results.values())
    assert isinstance(grid_result.best_params_, dict)
