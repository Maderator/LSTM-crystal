import argparse
import logging
import os
import pickle

import basic_parameters
import data_processing.data_preprocessing
import optimization.optimization as optimization
import sklearn.preprocessing
import utils
from functools import partial


def identity(X, y, window_size, inputs_lag, outputs_lag):
    return X, y

def initialize_results(results_path):
    """Initializes the results file with an empty list if the file does not exists."""
    if not os.path.exists(results_path):
        os.makedirs(os.path.dirname(results_path), exist_ok=True)
        with open(results_path, "wb") as f:
            pickle.dump([], f)

def main():

    ### CHANGE THIS ###
    
    
    params = basic_parameters.crystal_basic_parameters
    params["dataset"]["scaler"] = sklearn.preprocessing.StandardScaler()
    in_lags = [0, 2, 3, 4, 6, 8, 10]
    out_lags = [0, 2, 3, 4, 6, 8, 10]
    lag_functions = []
    for in_lag in in_lags:
        for out_lag in out_lags:
            partial_func = partial(data_processing.data_preprocessing.lag_features, inputs_lag=in_lag, outputs_lag=out_lag)
            partial_func.__name__ = f"lag_features_{in_lag}_{out_lag}"
            lag_functions.append(
                (
                    in_lag, 
                    out_lag, 
                    partial_func,
                )
            )
    
    experiments = []
    experiments.append(
        {
            "dataset-preprocessing_functions" : identity,
            "input_lag" : 0,
            "output_lag" : 0,
        }
    )
    for in_lag, out_lag, lag_function in lag_functions:
        exp = {
            "dataset-preprocessing_functions" : lag_function,
            "input_lag" : in_lag,
            "output_lag" : out_lag,
        }
        experiments.append(exp)
    
    
    #PARAM_GRID = {
    #    "dataset-preprocessing_functions" : [
    #        identity,
    #        data_processing.data_preprocessing.lag_features,
    #        data_processing.data_preprocessing.windowed_preprocessing,
    #    ],
    #    #"dataset-return_sequences" : [True],
    #}
    
    script_name = os.path.basename(__file__).split(".")[0]
    script_path = os.path.dirname(os.path.abspath(__file__))
    LOGS_FOLDER = os.path.join(script_path, "logs", script_name)
    USE_MULTIPROCESSING = True
    
    ALWAYS_VERBOSE = True
    
    ### CHANGE THIS ###

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

    experiment_path = os.path.dirname(os.path.abspath(__file__))

    #utils.dump_basic_params_and_changed_params(params, PARAM_GRID, experiment_path, LOGS_FOLDER)

    results_path = os.path.join(LOGS_FOLDER, "gridsearch_results.pkl")

    initialize_results(results_path)

    for exp in experiments:
        with open(results_path, "rb") as f:
            res_exps = pickle.load(f)
        not_computed = True
        for res_exp in res_exps:
            if res_exp["input_lag"] == exp["input_lag"] and res_exp["output_lag"] == exp["output_lag"]:
                exp["metrics_per_fold"] = res_exp["metrics_per_fold"]
                not_computed = False
                break
        if not_computed:
            results = optimization.do_gridsearch(
                param_grid={"no_params": [1,2]},
                experiment_path=os.path.dirname(os.path.abspath(__file__)),
                experiments=[exp],
                logs_folder=LOGS_FOLDER,
                use_multiprocessing=USE_MULTIPROCESSING,
                max_minutes=60,
                basic_parameters=params,
                save_experiments=False,
            ) 
            all_results = res_exps + results
            with open(os.path.join(LOGS_FOLDER, "gridsearch_results.pkl"), "wb") as f:
                pickle.dump(all_results, f)

if __name__ == "__main__":
    main()