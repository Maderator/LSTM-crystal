import argparse
import logging
import os
import pickle

import basic_parameters
import data_processing.data_preprocessing as data_preprocessing
import optimization.optimization as optimization
import sklearn.preprocessing
import utils

params = basic_parameters.crystal_basic_parameters
params["dataset"]["scaler"] = sklearn.preprocessing.StandardScaler()
params["dataset"]["preprocessing_function"] = data_preprocessing.lag_features
params["dataset"]["return_sequences"] = True
params["training"]["optimizer_params"]["learning_rate"] = 1e-4
params["training"]["optimizer_params"]["clipnorm"] = 5.0

#params["model"]["cell_type"] = "gru"
# remove rnn_cell_params other than units
#units = params["model"]["rnn_cell_params"]["units"]
#params["model"]["rnn_cell_params"] = {"units": units}

OPTIMIZATION_MODE = "gridsearch" # "bayesopt" # "gridsearch"
BAYESOPT_HAS_OLD_LOGS = True

script_name = os.path.basename(__file__).split(".")[0]
exp_path = os.path.dirname(os.path.abspath(__file__))
LOGS_FOLDER = os.path.join(exp_path, "logs", script_name)
USE_MULTIPROCESSING = True

ALWAYS_VERBOSE = True

if __name__ == "__main__":
    #logging.getLogger().setLevel(logging.DEBUG)
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.DEBUG) 
    
    while True:
        with open(os.path.join(LOGS_FOLDER, "gridsearch_results.pkl"), "rb") as f:
            experiments = pickle.load(f)
        pending_experiments = []
        finished_experiments = []
        for experiment in experiments:
            if experiment["metrics_per_fold"] is None:
                experiment.pop("metrics_per_fold")
                if "model-cell_type" in experiment and experiment["model-cell_type"] == "gru":
                    #params["model"]["cell_type"] = "gru"22
                    # remove rnn_cell_params other than units
                    units = params["model"]["rnn_cell_params"]["units"]
                    params["model"]["rnn_cell_params"] = {"units": units}
                
                pending_experiments.append(experiment)
            else:
                finished_experiments.append(experiment)
        print(len(pending_experiments))
        print(len(finished_experiments))
        #print(pending_experiments)
        
        if len(pending_experiments) == 0:
            break
        
        results = optimization.do_gridsearch(
            param_grid={"no_params": [1,2]},
            experiment_path=os.path.dirname(os.path.abspath(__file__)),
            experiments=pending_experiments,
            logs_folder=LOGS_FOLDER,
            use_multiprocessing=USE_MULTIPROCESSING,
            max_minutes=2,
            basic_parameters=params,
            save_experiments=False,
        )
        
        all_results = finished_experiments + results
        
        with open(os.path.join(LOGS_FOLDER, "gridsearch_results.pkl"), "wb") as f:
            pickle.dump(all_results, f)
        
    
    