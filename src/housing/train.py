# -*- coding: utf-8 -*-
"""train.py docstring

This module was created in order to create the pipeline and to
train the model both of which will be used by us in other modules.
"""

# IMPORTING LIBRARIES
import warnings

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

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


def basic_modeling(train, models, custom_class):
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
        custom_class: class
            custom class needed for creating the pipeline

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
            ("attribs_adder", custom_class),
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
        v: int, default 0
            value to put as verbosity during GridSearchCV

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


def training(data_loc, artifacts_loc, models, verbosity, custom_class):
    """
    Final function used to train the best estimator, and to save
    the model and pipeline to a given location

    Parameters
    ----------
        data_loc: str or path
            The path of the location where data will be stored
        artifacts_lod: str or path
            The path of the location where the artifacts will be stored
        models: dict
            Dictionary of the form {model_name: model} which will be used
            for basic modeling
        verbosity: int
            numeric value supplied by the user used to control the output
            of GridSearchCV
        custom_class: class
            custom class needed for the pipeline to work

    Returns
    ----------
        'results', 'model_selected', 'final_pipeline', 'tuning_result' where
        results is a dictionary of model name and the cross validation score
        model_selected is the model selected by the module, final_pipeline is
        the final pipeline used and tuning_result is the result of
        Hyperparameter tuning.
    """

    warnings.filterwarnings("ignore")
    train = pd.read_csv(data_loc + "/processed/train.csv")

    housing_x, housing_y, final_pipeline, results = basic_modeling(
        train,
        models,
        custom_class=custom_class()
    )

    model_selected = max(results)
    final_model = models[model_selected]

    tuning_result = model_search(
        train_x=housing_x,
        train_y=housing_y,
        model=final_model,
        v=verbosity,
    )

    joblib.dump(tuning_result.best_estimator_, artifacts_loc + "model.pkl")
    joblib.dump(final_pipeline, artifacts_loc + "pipeline.pkl")

    return results, model_selected, final_pipeline, tuning_result
