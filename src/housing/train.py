"""Module created to train the model"""
# IMPORTING LIBRARIES
import os
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor

from get_argument import argument
from logging_setup import configure_logger

# MODELLING
# BASIC MODEL
rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6


class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    """Class to add some extra features to the dataframe

    The features created are 'rooms_per_household' and
    'population_per_household'
    """

    def __init__(self, add_bedrooms_per_room=True):  # no *args or **kargs
        self.add_bedrooms_per_room = add_bedrooms_per_room

    def fit(self, X, y=None):
        return self  # nothing else to do

    def transform(self, X):
        """Function to transform the dataframe"""
        rooms_per_household = X[:, rooms_ix] / X[:, households_ix]
        population_per_household = X[:, population_ix] / X[:, households_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[
                X,
                rooms_per_household,
                population_per_household,
                bedrooms_per_room
            ]
        return np.c_[X, rooms_per_household, population_per_household]


def eval_matrics(model, x, y):
    """function to get r2 score using cross_val_score

    The default arguments of the function can be overwritten when
    supplied by the user

    Parameters
    ----------
            model:
                The model for which evaluatio is done
            x:
                Input values(Features)
            y:
                Output values(Label data)
    Returns
    ----------
    The mean of the cross validation scores
    """
    scores = cross_val_score(model, x, y, scoring="r2", cv=10)
    return scores.mean()


def basic_modeling(train):
    """Function created to create the pipeline and do some basic modelling
    on the data

    The default arguments of the function can be overwritten when supplied
    by the user

    Parameters
    ----------
            train:
                The training data for our problem
    Returns
    ----------
    The final input and output data along with the model showing best result
    as well as the full pipeline i.e. housing_prepared, housing_labels,
    models[model_selected] and full_pipeline
    """
    housing = train.drop("median_house_value", axis=1)
    housing_labels = train["median_house_value"].copy()

    num_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("attribs_adder", CombinedAttributesAdder()),
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
    models = {
        "Linear_regres": LinearRegression(),
        "Decision_tree": DecisionTreeRegressor(),
        "Random_forest": RandomForestRegressor(),
    }

    eval_dict = {}
    for model in models:
        score = eval_matrics(models[model], housing_prepared, housing_labels)
        eval_dict[model] = score
        logger.debug(f"{model}_R2_Score: \t{score}")

    model_selected = max(eval_dict)
    logger.info(f"\nModel Selected: \t{model_selected}")
    logger.info(f"Full pipeline used: \t{full_pipeline}")

    final_model = models[model_selected]

    return housing_prepared, housing_labels, final_model, full_pipeline


# FINE TUNING
def model_search(train_x, train_y, params_grid, model, cv):
    """Function to perform hyperparameter tuning on the model

    The default arguments of the function can be overwritten when
    supplied by the user

    Parameters
    ----------
            train_x:
                The training input data
            train_y:
                The training output data
            params_grid:
                Parameters grid for Hyperparameter training
            model:
                Final model on which we will be working on
            cv:
                Number of cross validations for Hyperparameter tuning
    Returns
    ----------
    Best estimator that can be created
    """

    grid_search = GridSearchCV(
        model,
        params_grid,
        cv=cv,
        scoring="r2",
        return_train_score=True,
        verbose=args.verbosity,
    )
    logger.debug("Starting hyperparameter tuning using GridSearchCV:")
    grid_search.fit(train_x, train_y)

    logger.info(
        f"Hyperparameters of {model} found: {grid_search.best_params_}."
    )
    logger.debug(f"Best score is: {grid_search.best_score_}")

    return grid_search.best_estimator_


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    args = argument()

    if args.log_path:
        log_f = os.path.join(args.log_path, "custom_configure.log")
    else:
        log_f = None

    logger = configure_logger(
        log_file=log_f, console=args.no_console_log, log_level=args.log_level
    )

    logger.info("Starting the run of train.py")
    train = pd.read_csv(args.data + "/processed/train.csv")
    housing_x, housing_y, final_model, final_pipeline = basic_modeling(train)

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

    final_model = model_search(
        train_x=housing_x,
        train_y=housing_y,
        params_grid=param_grid,
        model=final_model,
        cv=5,
    )

    joblib.dump(final_model, args.save + "model.pkl")
    joblib.dump(final_pipeline, args.save + "pipeline.pkl")
    logger.info(f"model and pipeline saved in {args.save}")
    logger.info("Run ended")
