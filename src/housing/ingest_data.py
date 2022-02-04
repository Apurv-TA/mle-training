"""Module created to download and create training and validation datasets"""
# importing libraries

import argparse
import os
import tarfile
import urllib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import StratifiedShuffleSplit

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--data", nargs="?", const="../../data")
location = parser.parse_args().data

# DATA
# LOADING DATA
def load_raw_data(
    housing_url="https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.tgz",
    housing_path=os.path.join(location, "raw"),
):
    """function to read and load raw data"""

    # fetching housing data
    os.makedirs(housing_path, exist_ok=True)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()

    # loading the data as a Dataframe
    csv_path = os.path.join(housing_path, "housing.csv")
    df = pd.read_csv(csv_path)
    return df


# TRAIN TEST SPLIT


def train_test(df, housing_path=os.path.join(location, "processed")):
    # creating training and test set
    df["income_cat"] = pd.cut(
        df["median_income"],
        bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf],
        labels=[1, 2, 3, 4, 5],
    )

    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(df, df["income_cat"]):
        strat_train_set = df.loc[train_index]
        strat_test_set = df.loc[test_index]

        for set_ in (strat_train_set, strat_test_set):
            set_.drop("income_cat", axis=1, inplace=True)

    os.makedirs(housing_path, exist_ok=True)
    strat_train_set.to_csv(housing_path + "/train.csv", index=False)
    strat_test_set.to_csv(housing_path + "/test.csv", index=False)

    return strat_train_set, strat_test_set


# FINAL FUNCTION


def data_loading():
    df = load_raw_data()
    train, test = train_test(df)

    return train, test


if __name__ == "__main__":
    data_loading()
