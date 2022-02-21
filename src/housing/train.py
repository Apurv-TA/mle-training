# -*- coding: utf-8 -*-
"""train.py docstring

This module was created in order to create the pipeline and to
train the model both of which will be used by us in other modules.
"""

# IMPORTING LIBRARIES
import os
import warnings

import get_argument
import joblib
import logging_setup
import numpy as np
import pandas as pd
import utils
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor

# MODELLING
# BASIC MODEL


def eval_matrics(model, x, y):
    """function to get r2 score using cross_val_score

    The default arguments of the function can be overwritten when
    supplied by the user

    Parameters
    ----------
            model:
                The model for which evaluatio is done
            x: array-like of shape (n_samples, n_features)
                The data to fit. Can be for example a list, or an array.
            y : array-like of shape (n_samples,) or (n_samples, n_outputs),
                default=None
                The target variable to try to predict in the case of
                supervised learning.

    Returns
    ----------
    The mean of the cross validation scores
    """

    scores = cross_val_score(model, x, y, scoring="r2", cv=10)
    return scores.mean()


def basic_modeling(train, models):
    """Function created to create the pipeline and do some basic modelling
    on the data

    The default arguments of the function can be overwritten when supplied
    by the user

    Parameters
    ----------
            train: pd.Dataframe
                The training data for our problem
            models: dict
                Dictionary of model names and the models we will be using
                for basic testing on the data.

    Returns
    ----------
    The final input and output data along with the model showing best result
    as well as the full pipeline i.e. housing_prepared, housing_labels,
    full_pipeline and evaluation result on basic tests.
    """

    housing = train.drop("median_house_value", axis=1)
    housing_labels = train["median_house_value"].copy()

    num_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("attribs_adder", utils.CombinedAttributesAdder()),
            ("std_scaler", StandardScaler()),
        ]
    )

    full_pipeline = ColumnTransformer(
        [
            (
                "num", num_pipeline,
                list(housing.drop("ocean_proximity", axis=1))
            ),
            ("cat", OneHotEncoder(), ["ocean_proximity"]),
        ]
    )

    housing_prepared = full_pipeline.fit_transform(housing)

    eval_dict = {}
    for model in models:
        score = eval_matrics(models[model], housing_prepared, housing_labels)
        eval_dict[model] = score

    return housing_prepared, housing_labels, full_pipeline, eval_dict


# FINE TUNING
def model_search(train_x, train_y, model, v=0):
    """Function to perform hyperparameter tuning on the model

    The default arguments of the function can be overwritten when
    supplied by the user

    Parameters
    ----------
            train_x: pd.Dataframe
                The training input data
            train_y: pd.Dataframe
                The training output data
            model:
                Final model on which we will be working on

    Returns
    ----------
    Best search determined on the basis of parameter grid provided
    and the number of cv.
    """

    param_grid = [
        {
            "n_estimators": [3, 10, 30, 100, 300],
            "max_features": [2, 4, 6, 8, 10]
        },
        {
            "bootstrap": [0],
            "n_estimators": [3, 10, 30, 100],
            "max_features": [2, 3, 4, 6],
        },
    ]

    grid_search = GridSearchCV(
        model,
        param_grid=param_grid,
        cv=5,
        scoring="r2",
        return_train_score=True,
        verbose=v,
    )
    grid_search.fit(train_x, train_y)

    return grid_search


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    args = get_argument.argument()
    train = pd.read_csv(args.data + "/processed/train.csv")

    models = {
        "Linear_regres": LinearRegression(),
        "Decision_tree": DecisionTreeRegressor(),
        "Random_forest": RandomForestRegressor(),
    }

    # defining the logger
    if args.log_path:
        LOG_FILE = os.path.join(args.log_path, "custom_configure.log")
    else:
        LOG_FILE = None

    logger = logging_setup.configure_logger(
        log_file=LOG_FILE,
        console=args.no_console_log,
        log_level=args.log_level
    )

    # starting the run
    logger.info("Starting the run of train.py")

    housing_x, housing_y, final_pipeline, results = basic_modeling(
        train, models)
    for model in results:
        logger.debug(
            f"{model}_R2_Score: \t{results[model]}"
        )

    model_selected = max(results)
    final_model = models[model_selected]

    logger.info(f"\nModel Selected: \t{model_selected}")
    logger.info(f"Full pipeline used: \t{final_pipeline}")

    logger.debug("Starting hyperparameter tuning using GridSearchCV:")
    tuning_result = model_search(
        train_x=housing_x,
        train_y=housing_y,
        model=final_model,
        v=args.verbosity,
    )

    logger.info(
        f"{model_selected} hyperparameters found: {tuning_result.best_params_}"
    )
    logger.debug(f"Best score is: {tuning_result.best_score_}")

    joblib.dump(tuning_result.best_estimator_, args.save + "model.pkl")
    joblib.dump(final_pipeline, args.save + "pipeline.pkl")

    logger.info(f"model and pipeline saved in {args.save}")
    logger.info("Run ended")
