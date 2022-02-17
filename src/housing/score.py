"""Script to get the model and its score on the test set"""
import os

import joblib
import pandas as pd
from sklearn.metrics import r2_score

from get_argument import argument
from logging_setup import configure_logger
from train import CombinedAttributesAdder


def score_test(df, model, full_pipeline):
    """Function created to calculate the score of the model on the
    basis of model selected and pipeline used.

    The default arguments of the function can be overwritten
    when supplied by the user

    Parameters
    ----------
            df:
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
    x_test = df.drop("median_house_value", axis=1)
    y_test = df["median_house_value"].copy()

    x_test_prepared = full_pipeline.transform(x_test)
    y_final_pred = model.predict(x_test_prepared)

    model_r2_score = r2_score(y_test, y_final_pred)
    logger.debug(
        f"The R2 score of the model on test set is: {model_r2_score.round(5)}"
    )
    return model_r2_score


if __name__ == "__main__":
    args = argument()
    if args.log_path:
        log_f = os.path.join(args.log_path, "custom_configure.log")
    else:
        log_f = None

    logger = configure_logger(
        log_file=log_f, console=args.no_console_log, log_level=args.log_level
    )

    logger.info("Starting the run of score.py")
    test = pd.read_csv(args.data + "/processed/test.csv")

    final_model = joblib.load(args.save + "model.pkl")
    housing_pipeline = joblib.load(args.save + "pipeline.pkl")

    score_test(df=test, model=final_model, full_pipeline=housing_pipeline)
    logger.info("Run ended")
