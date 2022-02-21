import os.path as op
import sys

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal, assert_series_equal

HERE = op.dirname(op.abspath("__file__"))
test_path = op.join(HERE, "..", "..", "src", "housing")
sys.path.append(test_path)
import ingest_data


@pytest.fixture
def data():
    df = pd.read_csv("../../data/raw/housing.csv")
    return df


@pytest.fixture
def train_df():
    train = pd.read_csv("../../data/processed/train.csv")
    return train


@pytest.fixture
def test_df():
    test = pd.read_csv("../../data/processed/test.csv")
    return test


def test_load_raw_data(data):
    load_df = ingest_data.load_raw_data()
    assert isinstance(load_df, pd.DataFrame)
    assert load_df.shape == data.shape
    assert load_df.equals(data)


def test_train_test(data, train_df, test_df):
    load_train, load_test = ingest_data.train_test(data)

    assert isinstance(load_train, pd.DataFrame)
    assert "income_cat" not in load_train.columns
    assert_series_equal(
        pd.Series(load_train.columns), pd.Series(train_df.columns)
    )
    assert load_train.shape == train_df.shape

    assert isinstance(load_test, pd.DataFrame)
    assert "income_cat" not in load_test.columns
    assert_series_equal(
        pd.Series(load_test.columns), pd.Series(test_df.columns)
    )
    assert load_test.shape == test_df.shape

    assert_frame_equal(
        pd.concat([load_train, load_test], ignore_index=True),
        pd.concat([train_df, test_df], ignore_index=True),
    )
