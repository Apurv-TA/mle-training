# -*- coding: utf-8 -*-
"""ingest_data.py docstring

This module was created to download the data on which we will
be working on and to split the data and create training and
validation datasets.

Attributes
----------
args: argument namespace
    It is created in order to define the argument namespace.
"""

# importing libraries

import os
import tarfile
import urllib.request

import numpy as np
import pandas as pd
import get_argument
import logging_setup
from sklearn.model_selection import StratifiedShuffleSplit

# DATA
# LOADING DATA

args = get_argument.argument()


def load_raw_data(
    housing_url="""
    https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.tgz
    """,
    housing_path=os.path.join(args.data, "raw"),
):
    """Function to read and load raw data.

    The default arguments of the function can be overwritten when supplied by
    the user

    Parameters
    ----------
            housing_url: str
                url from which the data can be taken
            housing_path: StrOrBytesPath
                location of the path where the file can be saved

    Returns
    ----------
    The dataframe
    """

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


def train_test(df, housing_path=os.path.join(args.data, "processed")):
    """Function to split the data into train and test

    The default arguments of the function can be overwritten
    when supplied by the user

    Parameters
    ----------
            df: pd.Dataframe
                The dataframe on which we are working on
            housing_path: StrOrBytesPath
                location of the path where the file can be saved

    Returns
    ----------
    The dataframes
    """

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
    """This function combines all the things that are done in this module

    Returns
    ----------
    The Training and testing data
    """

    df = load_raw_data()
    train, test = train_test(df)

    return train, test


if __name__ == "__main__":
    if args.log_path:
        LOG_FILE = os.path.join(args.log_path, "custom_configure.log")
    else:
        LOG_FILE = None
    logger = logging_setup.configure_logger(
        log_file=LOG_FILE,
        console=args.no_console_log,
        log_level=args.log_level
    )

    logger.info("Starting the run of ingest_data.py")
    data_loading()
    logger.debug(f"Data saved in {args.data}")
    logger.info("Run ended")
