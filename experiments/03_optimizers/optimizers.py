# Compare different optimizers

# As described in the diploma thesis, there are many different optimizers for neural network training.
# We will test the following optimizers which are implemented by the Tensorflow Keras framework:
# - SGD
# - SGD with momentum
# - SGD with momentum and nesterov accelerated gradient
# - RMSProp (with momentum)
# - Adam
# - Nadam
# - AdamW

import argparse
import logging
import os

import basic_parameters
import data_processing.data_preprocessing
import optimization.optimization as optimization
import sklearn.preprocessing
import utils

### CHANGE THIS ###
params = basic_parameters.crystal_basic_parameters
params["dataset"]["scaler"] = sklearn.preprocessing.StandardScaler()
params["dataset"]["preprocessing_function"] = data_processing.data_preprocessing.windowed_preprocessing
params["dataset"]["return_sequences"] = False

#CHOSEN_PARAM_GRID = "rmsprop" # "sgd", "rmsprop", "adam", "other"

PARAM_GRID_SGD = {
    "training-optimizer": ["sgd", "sgd_nesterov",],
    "training-optimizer_params-learning_rate": [1e-2, 5e-3, 1e-3, 5e-4, 1e-4],
    "training-optimizer_params-momentum": [None, 0.9, 0.99],
    #"training-epochs": [1]
}

PARAM_GRID_RMSPROP = {
    "training-optimizer": ["rmsprop",],
    "training-optimizer_params-learning_rate": [1e-2, 5e-3, 1e-3, 5e-4, 1e-4],
    "training-optimizer_params-momentum": [None, 0.9, 0.99],
    #"training-epochs": [1]
}

PARAM_GRID_ADAM = {
    "training-optimizer": ["adam", "nadam", "adamw",],
    "training-optimizer_params-learning_rate": [1e-2, 1e-3, 1e-4, 1e-5],
    "training-optimizer_params-momentum": [None],
}

PARAM_GRID_OTHER = {
    "training-optimizer": ["adadelta","adafactor","adagrad","adamax","ftrl","lion",], # Other optimizers not specified in the thesis
    "training-optimizer_params-learning_rate": [1e-2, 5e-3, 1e-3, 5e-4, 1e-4],
    "training-optimizer_params-momentum": [None],
}

#PARAM_GRID_DICT = {"sgd": PARAM_GRID_SGD, "rmsprop":PARAM_GRID_RMSPROP, "adam": PARAM_GRID_ADAM,"other": PARAM_GRID_OTHER,}
PARAM_GRID_DICT = {"other": PARAM_GRID_OTHER,}
#PARAM_GRID = PARAM_GRID_DICT[CHOSEN_PARAM_GRID]

#script_name = os.path.basename(__file__).split(".")[0]
#LOGS_FOLDER = os.path.join("logs", script_name, CHOSEN_PARAM_GRID)
USE_MULTIPROCESSING = True

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
    argparser.add_argument("--logs_decimals", type=int, default=5, help="Number of decimals to use for the log file name from the parameter value",)
    args = argparser.parse_args()
    
    if args.verbose or ALWAYS_VERBOSE:
        logging.getLogger().setLevel(logging.DEBUG)
    else:
        logging.getLogger().setLevel(logging.WARNING)

    for CHOSEN_PARAM_GRID in PARAM_GRID_DICT.keys():
        logging.info(f"Starting optimization for optimizers in group: {CHOSEN_PARAM_GRID}")
        cur_path = os.path.dirname(os.path.abspath(__file__))

        script_name = os.path.basename(__file__).split(".")[0]
        LOGS_FOLDER = os.path.join("logs", script_name, CHOSEN_PARAM_GRID)
        PARAM_GRID = PARAM_GRID_DICT[CHOSEN_PARAM_GRID]

        params = basic_parameters.crystal_basic_parameters

        experiment_path = os.path.dirname(os.path.abspath(__file__))

        utils.dump_basic_params_and_changed_params(params, PARAM_GRID, experiment_path, LOGS_FOLDER)

        optimization.search(
            experiment_path=experiment_path,
            basic_parameters = params,
            max_minutes=args.max_minutes,
            optimization_mode=args.optimization_mode,
            gs_csv_log_path=os.path.join(cur_path, "logs", f"{script_name}_{CHOSEN_PARAM_GRID}_comparison.csv"),
            param_grid=PARAM_GRID,
            folds_first_n_letters=args.logs_first_n_letters,
            folds_decimals=args.logs_decimals,    
            logs_folder=LOGS_FOLDER,
            use_multiprocessing=USE_MULTIPROCESSING,
        )
