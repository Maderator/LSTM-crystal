import argparse
import logging
import os
import pickle

import basic_parameters
import data_processing.data_normalization as normalization
import optimization.optimization as optimization
import sklearn.preprocessing
import utils

### CHANGE THIS ###

def identity(X, y, window_size, inputs_lag, outputs_lag):
    return X, y

def scaler_identity(X):
    return X

identity_scaler = sklearn.preprocessing.FunctionTransformer(func=scaler_identity, inverse_func=scaler_identity)

PARAM_GRID = {
    "dataset-scaler" : [
        identity_scaler,
        normalization.GlobalMinMaxScaler(),
        normalization.GlobalMaxAbsScaler(),
        normalization.GlobalStandardScaler(),
        normalization.GlobalRobustScaler(),
        sklearn.preprocessing.MinMaxScaler(),
        sklearn.preprocessing.MaxAbsScaler(),
        sklearn.preprocessing.StandardScaler(),
        sklearn.preprocessing.RobustScaler(),
    ],
    "training-epochs": [1]
}

script_name = os.path.basename(__file__).split(".")[0]
LOGS_FOLDER = os.path.join("logs", script_name)
USE_MULTIPROCESSING = False

ALWAYS_VERBOSE = True

### CHANGE THIS ###

if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(message)s", level=logging.INFO
    )
    argparser = argparse.ArgumentParser()
    argparser.add_argument("-v", "--verbose", action="store_true", help="Prints debug and info messages to the console", )
    argparser.add_argument("--max_minutes", type=int, default=60 * 20, help="Maximum number of minutes for optimization", )
    argparser.add_argument("--optimization_mode", type=str, default="gridsearch")
    argparser.add_argument("--init_points", type=int, default=5, help="Number of initial points for Bayesian optimization", )
    argparser.add_argument("--old_logs", action="store_true", help="Load logs from previous optimization", )
    argparser.add_argument("--basic_parameters_path", type=str, help="Path to basic parameters json file", default=None, ) 
    argparser.add_argument("--logs_first_n_letters", type=int, default=1, help="Number of letters to use for the log file name from each word in the parameter name",)
    argparser.add_argument("--logs_decimals", type=int, default=2, help="Number of decimals to use for the log file name from the parameter value",)
    args = argparser.parse_args()
    
    if args.verbose or ALWAYS_VERBOSE:
        logging.getLogger().setLevel(logging.DEBUG)
    else:
        logging.getLogger().setLevel(logging.WARNING)

    cur_path = os.path.dirname(os.path.abspath(__file__))
    root_path = utils.find_root_path("LSTM-crystal-growth")

    params = basic_parameters.crystal_basic_parameters

    experiment_path = os.path.dirname(os.path.abspath(__file__))

    utils.dump_basic_params_and_changed_params(params, PARAM_GRID, experiment_path, LOGS_FOLDER)

    optimization.search(
        experiment_path=experiment_path,
        basic_parameters = params,
        max_minutes=args.max_minutes,
        optimization_mode=args.optimization_mode,
        gs_csv_log_path=os.path.join(cur_path, "logs", f"{script_name}_comparison.csv"),
        param_grid=PARAM_GRID,
        folds_first_n_letters=args.logs_first_n_letters,
        folds_decimals=args.logs_decimals,    
        logs_folder=LOGS_FOLDER,
        use_multiprocessing=USE_MULTIPROCESSING,
    )
