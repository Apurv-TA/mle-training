import argparse

def argument():
    """Function to get argument if given by the user

    Returns
    ----------
    The argument namespace
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log-level",
        help="Specify the log level(Default is DEBUG)",
        default="DEBUG",
    )
    parser.add_argument(
        "--log-path",
        help="Specify the file to which log need to be saved: Default: No file created",
        default=None
    )
    parser.add_argument(
        "--no-console-log",
        help="whether or not to write logs to the console: Default: write logs to console",
        action="store_false",
    )
    parser.add_argument(
        "--data",
        help="The location where data is to be saved",
        default="../../data/"
    )
    parser.add_argument(
        "--save",
        help="Location where artifacts are to be stored",
        default="../../artifacts/"
    )
    parser.add_argument(
        "-v", "--verbosity", help="Show the output of Grid Search",
        choices=[ 1, 2, 3], type=int, default=0
    )
    return parser.parse_args()
