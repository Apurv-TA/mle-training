# -*- coding: utf-8 -*-
"""score.py docstring

Script to get the model and determine its performance on the test set.
It is evaluated with the help of r2_Score
"""

import joblib
import pandas as pd
from sklearn.metrics import r2_score


def score_test(df_test, model, full_pipeline, custom_class):
    """Function created to calculate the score of the model on the
    basis of model selected and pipeline used.

    The default arguments of the function can be overwritten
    when supplied by the user

    Parameters
    ----------
        df_test: pd.Dataframe
            The dataframe(containing test data) on which we are working on
        model:
            The model used i.e. final model on which we are calculating
            score on
        full_pipeline:
            Final pipeline generated which is to be used for processing
            data
        custom_class: class
            custom class needed for the pipeline to work

    Returns
    ----------
    The R2 score is printed and returned
    """

    x_test = df_test.drop("median_house_value", axis=1)
    y_test = df_test["median_house_value"].copy()

    x_test_prepared = full_pipeline.transform(x_test)
    y_final_pred = model.predict(x_test_prepared)

    model_r2_score = r2_score(y_test, y_final_pred)
    return model_r2_score


def result(data_loc, artifacts_loc, custom_class):
    """Function created to return the final score on the model.

    Parameters
    ----------
        data_loc: str
            path of the location where the data is stored
        artifacts_loc: str
            path of the location where the artifacts i.e. pipeline
            and the model is saved.
        custom_class: class
            custom class needed for the pipeline to work

    Returns
    ----------
    The final score caluclated is returned.
    """
    test = pd.read_csv(data_loc + "/processed/test.csv")

    final_model = joblib.load(artifacts_loc + "model.pkl")
    housing_pipeline = joblib.load(artifacts_loc + "pipeline.pkl")

    score = score_test(
        df_test=test,
        model=final_model,
        full_pipeline=housing_pipeline,
        custom_class=custom_class()
    )

    return score
