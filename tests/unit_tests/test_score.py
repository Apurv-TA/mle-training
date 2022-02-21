import os.path as op
import sys

import joblib
import pandas as pd
import pytest

HERE = op.dirname(op.abspath("__file__"))
test_path = op.join(HERE, "..", "..", "src", "housing")
sys.path.append(test_path)
import score
import utils


@pytest.fixture
def test_df():
    data = pd.read_csv("../../data/processed/test.csv").sample(200)
    return data


@pytest.fixture
def model():
    return joblib.load("../../artifacts/model.pkl")


@pytest.fixture
def pipe():
    return joblib.load("../../artifacts/pipeline.pkl")


def test_score(test_df, model, pipe):
    score_model = score.score_test(
        test_df,
        model,
        pipe,
        utils.CombinedAttributesAdder()
    )
    assert isinstance(score_model, float)
