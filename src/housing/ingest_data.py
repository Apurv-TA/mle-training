# -*- coding: utf-8 -*-
"""ingest_data.py docstring

This module was created to download the data on which we will
be working on and to split the data and create training and
validation datasets.

"""

# importing libraries

import os
import tarfile
import urllib.request

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

# DATA
# LOADING DATA


def load_raw_data(
    loc="../../data",
    housing_url="""
    https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.tgz
    """
):
    """Function to read and load raw data.

    The default arguments of the function can be overwritten when supplied by
    the user

    Parameters
    ----------
        loc: StrOrBytesPath
            location of the path where the file can be saved
        housing_url: str
            url from which the data can be taken

    Returns
    ----------
    The dataframe
    """

    housing_path = os.path.join(loc, "raw")

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


def train_test(df, loc):
    """Function to split the data into train and test

    The default arguments of the function can be overwritten
    when supplied by the user

    Parameters
    ----------
        df: pd.Dataframe
            The dataframe on which we are working on
        loc: StrOrBytesPath
            location of the path where the file can be saved

    Returns
    ----------
    The dataframes
    """
    housing_path = os.path.join(loc, "processed")

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


def data_loading(data_url, location):
    """This function combines all the things that are done in this module

    Parameters
    ----------
        data_url: str
            URL of the web location from where data can be loaded
        location: str
            Location of the folder where the data will be stored

    Returns
    ----------
    The Training and testing data
    """

    df = load_raw_data(loc=location, housing_url=data_url)
    train_test(df, loc=location)
