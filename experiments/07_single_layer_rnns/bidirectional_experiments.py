import argparse
import logging
import os

import basic_parameters
import data_processing.data_preprocessing as data_preprocessing
import optimization.optimization as optimization
import sklearn.preprocessing
import utils

### CHANGE THIS ###
params = basic_parameters.crystal_basic_parameters
params["dataset"]["scaler"] = sklearn.preprocessing.StandardScaler()
params["dataset"]["preprocessing_function"] = data_preprocessing.lag_features
params["dataset"]["return_sequences"] = True
params["training"]["optimizer_params"]["learning_rate"] = 1e-4
params["training"]["optimizer_params"]["clipnorm"] = 5.0

#params["model"]["cell_type"] = "gru"
# remove rnn_cell_params other than units
if params["model"]["cell_type"] == "gru":
    units = params["model"]["rnn_cell_params"]["units"]
    params["model"]["rnn_cell_params"] = {"units": units}

OPTIMIZATION_MODE = "gridsearch" # "bayesopt" # "gridsearch"
BAYESOPT_HAS_OLD_LOGS = True

PARAM_GRID = {
    "model-cell_type": ["lstm"],
    "model-rnn_cell_params-units": [5, 10, 20, 25, 50, 100, 200,],
    "model-bidirectional":[True],
    #"training-epochs": [1],
}
PBOUNDS = {
    "model-rnn_cell_params-units": (10, 200),
}

PBOUNDS_TYPES = {
    "model-rnn_cell_params-units": int,

}

script_name = os.path.basename(__file__).split(".")[0]
exp_path = os.path.dirname(os.path.abspath(__file__))
LOGS_FOLDER = os.path.join(exp_path, "logs", script_name)
USE_MULTIPROCESSING = True

ALWAYS_VERBOSE = True

### CHANGE THIS ###

if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(message)s", level=logging.INFO
    )
    argparser = argparse.ArgumentParser()
    argparser.add_argument("-v", "--verbose", action="store_true", help="Prints debug and info messages to the console", )
    argparser.add_argument("--max_minutes", type=int, default=7*60, help="Maximum number of minutes for optimization", )
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
    bayes_log_path = os.path.join(LOGS_FOLDER, "bayesopt_logs.json")

    utils.dump_basic_params_and_changed_params(params, PARAM_GRID, experiment_path, LOGS_FOLDER)

    optimization.search(
        experiment_path=experiment_path,
        basic_parameters = params,
        bayesopt_json_log_path=bayes_log_path, # TODO: add path
        max_minutes=args.max_minutes,
        optimization_mode=OPTIMIZATION_MODE,
        bayesopt_init_points=args.init_points,
        bayesopt_old_logs=BAYESOPT_HAS_OLD_LOGS,
        bayesopt_pbounds=PBOUNDS,
        bayesopt_pbounds_types=PBOUNDS_TYPES,
        gs_csv_log_path=os.path.join(cur_path, "logs", "peephole_experiments.csv"),
        param_grid=PARAM_GRID,
        folds_first_n_letters=args.logs_first_n_letters,
        folds_decimals=args.logs_decimals,    
        logs_folder=LOGS_FOLDER,
        use_multiprocessing=USE_MULTIPROCESSING,
    )
