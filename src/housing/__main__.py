import os

import utils
from get_argument import argument
from ingest_data import data_loading
from logging_setup import configure_logger
from score import result
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from train import training

if __name__ == "__main__":
    args = argument()

    if args.log_path:
        LOG_FILE = os.path.join(args.log_path, "custom_configure.log")
    else:
        LOG_FILE = None

    logger = configure_logger(
        log_file=LOG_FILE,
        console=args.no_console_log,
        log_level=args.log_level
    )

    logger.info("Starting the file run.")
    # ----------

    data_loading(
        data_url="""
        https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.tgz
        """,
        location=args.data,
    )
    logger.debug(f"Data saved in {args.data}")

    # ----------

    models = {
        "Linear_regres": LinearRegression(),
        "Decision_tree": DecisionTreeRegressor(),
        "Random_forest": RandomForestRegressor(),
    }
    results, model_selected, final_pipeline, tuning_result = training(
        data_loc=args.data,
        artifacts_loc=args.save,
        models=models,
        verbosity=args.verbosity,
        custom_class=utils.CombinedAttributesAdder
    )
    for model in results:
        logger.debug(
            f"{model}_R2_Score: \t{results[model]}"
        )
    logger.info(f"\nModel Selected: \t{model_selected}")
    logger.info(f"Full pipeline used: \t{final_pipeline}")
    logger.info(
        f"{model_selected} hyperparameters found:{tuning_result.best_params_}"
    )
    logger.debug(f"Best score is: {tuning_result.best_score_}")
    logger.info(f"model and pipeline saved in {args.save}")

    # ----------

    outcome = result(
        data_loc=args.data,
        artifacts_loc=args.save,
        custom_class=utils.CombinedAttributesAdder
    )
    logger.debug(
        f"The R2 score of the model on test set is: {outcome}"
    )
    # ----------
    logger.info("Run ended")
