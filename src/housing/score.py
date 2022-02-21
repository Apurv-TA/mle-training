# -*- coding: utf-8 -*-
"""score.py docstring

Script to get the model and determine its performance on the test set.
It is evaluated with the help of r2_Score
"""
import os

import get_argument
import joblib
import logging_setup
import pandas as pd
from sklearn.metrics import r2_score
from utils import CombinedAttributesAdder


def score_test(df_test, model, full_pipeline):
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


if __name__ == "__main__":
    args = get_argument.argument()
    if args.log_path:
        LOG_FILE = os.path.join(args.log_path, "custom_configure.log")
    else:
        LOG_FILE = None

    logger = logging_setup.configure_logger(
        log_file=LOG_FILE,
        console=args.no_console_log,
        log_level=args.log_level
    )

    logger.info("Starting the run of score.py")
    test = pd.read_csv(args.data + "/processed/test.csv")

    final_model = joblib.load(args.save + "model.pkl")
    housing_pipeline = joblib.load(args.save + "pipeline.pkl")

    score = score_test(
        df_test=test,
        model=final_model,
        full_pipeline=housing_pipeline
    )

    logger.debug(
        f"The R2 score of the model on test set is: {score}"
    )
    logger.info("Run ended")
