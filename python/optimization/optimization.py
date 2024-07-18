import copy
import itertools
import logging
import os
import pickle
import time
from functools import partial

import bayes_opt
import multiprocess
import numpy as np
import utils
from optimization.data_classes import KFoldResults
from optimization.training import split_training
from sklearn.model_selection import KFold

def initialize_class(params: dict, root_path: str, class_name):
    params = copy.deepcopy(params)
    params["data_path"] = os.path.join(root_path, params["data_path"])
    init_params = {k: params[k] for k in class_name.__init__.__code__.co_varnames if k in params}
    return class_name(**init_params)

def optimization_function(
    changed_params,
    basic_parameters,
    experiment_path,
    first_n_letters=1,
    decimals=2,
    logs_folder="logs/main",
    use_multiprocessing = True,
) -> KFoldResults:
    cur_path = experiment_path 
    params = basic_parameters

    params = utils.change_dict_values(params, changed_params)
    dataset_params = params["dataset"]
    model_params = params["model"]
    training_params = params["training"]

    root_path = utils.find_root_path("LSTM-crystal-growth")
    #dataset = initialize_dataset(dataset_params, root_path)
    dataset = initialize_class(dataset_params, root_path, params["dataset_class"])

    input_shape, output_shape = dataset.get_model_input_output_shapes()

    # TRAINING AND EVALUATION
    if "k_folds" not in dataset_params or dataset_params["k_folds"] in [None, 0, 1]:
        # no splitting
        dataset_size = len(dataset.raw_train_data["Inputs"])
        splits = [(np.arange(dataset_size), np.arange(dataset_size))]
    else:
        # split dataset to k folds
        k_fold = KFold(n_splits=dataset_params["k_folds"]) #shuffle=False
        splits = k_fold.split(
            dataset.raw_train_data["Inputs"]
        )  # Whole dataset is in train_data field
        splits = list(splits)
    try:

        if use_multiprocessing:
            manager = multiprocess.Manager()
            process_results = manager.dict()

            processes=[]
            for i, (train_index, test_index) in enumerate(splits):
                cur_ds = dataset.reassign_and_scale_dataset(train_index, test_index)
                process = multiprocess.Process(target=split_training, args=(changed_params, model_params, training_params, dataset_params, cur_ds, input_shape, output_shape, i, cur_path, logs_folder, first_n_letters, decimals, True, process_results))
                processes.append(process)

            for i, process in enumerate(processes):
                logging.info(f"Starting process {i}")
                process.start()

            for i, process in enumerate(processes):
                process.join()
                logging.info(f"Joined process {i}")

            all_results = list(process_results.values())
            if len(all_results) == 0:
                logging.warning(f"Experiment did not return results. Returning np.nan as result.")
                return np.nan, [np.nan]*dataset.raw_train_data["Outputs"].shape[-1], np.nan, [np.nan]*dataset.raw_train_data["Outputs"].shape[-1]
            
            return KFoldResults(all_results)

        # Not multiprocessing
        split_training_partial = partial(
            split_training,
            changed_params=changed_params,
            model_params=model_params,
            training_params=training_params,
            dataset_params=dataset_params,
            input_shape=input_shape,
            output_shape=output_shape,
            cur_path=cur_path,
            logs_folder=logs_folder,
            first_n_letters=first_n_letters,
            decimals=decimals,
        )
        all_results = []
        for i, (train_index, test_index) in enumerate(splits):
            cur_ds = dataset.reassign_and_scale_dataset(train_index, test_index)
            results = split_training_partial(dataset=cur_ds, i=i)
            all_results.append(results)
        return KFoldResults(all_results)

    except (utils.NesterovMissingError, utils.MomentumNotSupportedError) as e:
        logging.error(f"Experiment with training_params:{training_params} did not run: {e} Returning results of this experiment as np.nan")
        num_features = dataset.raw_train_data["Outputs"].shape[-1]
        return np.nan, [np.nan]*num_features, np.nan, [np.nan]*num_features

def save_results(
    results: KFoldResults,
    experiments, exp_idx, start_time, save_to_file=False, experiment_path=None, logs_folder="logs/main",
) -> None:
    elapsed_time = utils.get_delta_minutes(start_time)
    experiment_changed_params = copy.deepcopy(experiments[exp_idx])
    if "training-additional_callbacks" in experiment_changed_params:
        experiment_changed_params["training-additional_callbacks"] = [utils.serialize_callback(callback) for callback in experiment_changed_params["training-additional_callbacks"]]

    if not results:
        logging.warning("No results to save")
        kfold_res = None
    else:
        kfold_res = results.get_list_of_results()
    experiments[exp_idx]["metrics_per_fold"] = kfold_res

    if save_to_file:
        params_shortcut = utils.get_dictionary_short_string(
            experiment_changed_params, first_n_letters=2, decimals=5
        )
        dict_results = experiment_changed_params
        dict_results["results"] = {
            "kfolds": kfold_res,
        }
        exp_path = os.path.join(experiment_path, logs_folder, "partial_results", f"{params_shortcut}.pkl")
        os.makedirs(os.path.dirname(exp_path), exist_ok=True)
        with open(exp_path, "wb") as f:
            pickle.dump(dict_results, f)

    logging.info(f"Folds RMSE unscaled results: {[fold_res['root_mean_squared_error']['unscaled'].metric for fold_res in kfold_res] if kfold_res else None}")
    logging.info(f"Elapsed time (experiment): {elapsed_time} minutes")
    
def check_max_time_reached(experiments, exp_idx, start_time, max_minutes=60*20):
    if utils.get_delta_minutes(start_time) > max_minutes:
        # Stop the optimization and set the rest of the experiments results to np.nan
        logging.info(f"Max time reached, stopping")
        logging.info(f"Pending experimnets: {len(experiments) - exp_idx - 1}")
        logging.info(f"{experiments[exp_idx+1:]}")
        for n in range(exp_idx + 1, len(experiments)):
            experiments[n]["metrics_per_fold"] = None
        return True
    return False

def do_bayes_optimization(
    pbounds,
    pbounds_types,
    basic_parameters,
    experiment_path,
    json_log_path,
    max_minutes=60 * 20,
    init_points=5,
    have_logs=False,
    first_n_letters=1,
    decimals=2,
    logs_folder="logs/main",
    use_multiprocessing=True,
):
    if json_log_path is None:
        raise ValueError(
            "json_log_path must be specified for do_bayes_optimization function. Current value: None."
        )

    start_time = time.time()

    optimizer = bayes_opt.BayesianOptimization(
        f=None,
        pbounds=pbounds,
        verbose=2,
        allow_duplicate_points=True,
        random_state=1,
    )
    optimizer.set_gp_params(alpha=1e-4)  # default alpha=1e-6

    utility = bayes_opt.UtilityFunction(
        kind="ei", xi=1e-1, kappa_decay=0.99, kappa_decay_delay=5
    )

    logger = bayes_opt.logger.JSONLogger(path=json_log_path, reset=False)
    logging.info(f"Bayesian optimization JSONLogger path:{json_log_path}")
    if have_logs and os.path.exists(json_log_path):
        bayes_opt.util.load_logs(optimizer, logs=[json_log_path])
    else:
        os.makedirs(os.path.dirname(json_log_path), exist_ok=True)

    optimizer.subscribe(bayes_opt.event.Events.OPTIMIZATION_STEP, logger)

    experiments = []
    i = 0
    while utils.get_delta_minutes(start_time) < max_minutes:
        logging.info(
            f"Elapsed time: {utils.get_delta_minutes(start_time)} minutes"
        )
        experiment_start_time = time.time()
        
        next_point_to_probe = optimizer.suggest(utility)
        casted_next_point_to_probe = utils.cast_dict_values(next_point_to_probe, pbounds_types)
        experiments.append(casted_next_point_to_probe)

        logging.info(f"Next points to probe: {casted_next_point_to_probe}")

        #target_scaled, target_per_feature_scaled, target_unscaled, target_per_feature_unscaled 
        results = optimization_function(
            changed_params=casted_next_point_to_probe,
            basic_parameters=basic_parameters,
            experiment_path=experiment_path,
            first_n_letters=first_n_letters,
            decimals=decimals,
            logs_folder=logs_folder,
            use_multiprocessing=use_multiprocessing,
        )
        #target_scaled = results.rmse_scaled
        list_of_res = results.get_list_of_results()
        rmses_unscaled = [res["root_mean_squared_error"]["unscaled"].metric for res in list_of_res]
        target_unscaled = np.mean(rmses_unscaled)
        
        negative_target = -target_unscaled
        optimizer.register(
            params=next_point_to_probe,
            target=negative_target,
        )
    
        save_results(results, experiments, i, experiment_start_time, save_to_file=True, experiment_path=experiment_path, logs_folder=logs_folder)
        logging.info(f"Experiment {i+1}/{len(experiments)} finished")
        logging.info(f"Elapsed time: {utils.get_delta_minutes(start_time)} minutes")
        i += 1
    with open(os.path.join(experiment_path, logs_folder, "bayesopt_results.pkl"), "wb") as f:
        pickle.dump(experiments, f)

def do_gridsearch(
    param_grid,
    basic_parameters,
    experiment_path,
    max_minutes=60 * 20,
    first_n_letters=1,
    decimals=2,
    logs_folder="logs/main",
    use_multiprocessing=True,
    experiments = None,
    save_experiments: bool = True,
):
    start_time = time.time()

    keys, values = zip(*param_grid.items())
    # If experiments are not given, Generate them by creating all the possible combination of the given param_grid values
    experiments = experiments or [dict(zip(keys, v)) for v in itertools.product(*values)]

    for i, experiment in enumerate(experiments):
        logging.info(f"Experiment {i+1}/{len(experiments)} started")
        logging.info(f"Current experiment: {experiment}")
        experiment_start_time = time.time()

        # Run experiment
        #mean_rmse_scaled, mean_rmse_per_feature_scaled, mean_rmse_unscaled, mean_rmse_per_feature_unscaled
        results = optimization_function(
            changed_params=experiment,
            basic_parameters=basic_parameters,
            experiment_path=experiment_path,
            first_n_letters=first_n_letters,
            decimals=decimals,
            logs_folder=logs_folder,
            use_multiprocessing=use_multiprocessing,
        )

        save_results(results, experiments, i, experiment_start_time, save_to_file=True, experiment_path=experiment_path, logs_folder=logs_folder)
        if check_max_time_reached(experiments, i, start_time, max_minutes):
            break
        logging.info(f"Experiment {i+1}/{len(experiments)} finished")
        logging.info(f"Elapsed time: {utils.get_delta_minutes(start_time)} minutes")
    if save_experiments:
        for exp in experiments:
            if "training-additional_callbacks" in exp:
                exp["training-additional_callbacks"] = [utils.serialize_callback(callback) for callback in exp["training-additional_callbacks"]]
        with open(os.path.join(experiment_path, logs_folder, "gridsearch_results.pkl"), "wb") as f:
            pickle.dump(experiments, f)
    return experiments

def search(
    experiment_path,
    basic_parameters,
    bayesopt_json_log_path=None,
    max_minutes=60 * 20,
    optimization_mode="gridsearch",
    bayesopt_init_points=5,
    bayesopt_old_logs=False,
    bayesopt_pbounds=None,
    bayesopt_pbounds_types=None,
    gs_csv_log_path=None,
    param_grid=None,
    folds_first_n_letters=1,
    folds_decimals=2,
    logs_folder="logs/main",
    use_multiprocessing=False,
):
    if optimization_mode == "bayesopt":
        if bayesopt_json_log_path is None:
            raise ValueError(
                "bayesopt_json_log_path must be specified for bayesopt optimization mode"
            )

        if bayesopt_pbounds is None:
            raise ValueError(
                "bayesopt_pbounds must be specified for bayesopt optimization mode"
            )
        do_bayes_optimization(
            pbounds = bayesopt_pbounds,
            pbounds_types = bayesopt_pbounds_types,
            basic_parameters = basic_parameters,
            experiment_path = experiment_path,
            json_log_path=bayesopt_json_log_path,
            max_minutes = max_minutes,
            init_points=bayesopt_init_points,
            have_logs=bayesopt_old_logs,
            first_n_letters=folds_first_n_letters,
            decimals=folds_decimals,
            logs_folder=logs_folder,
            use_multiprocessing=use_multiprocessing,
        )
    elif optimization_mode == "gridsearch":
        if param_grid is None:
            raise ValueError("param_grid must be specified for gridsearch")
        do_gridsearch(
            param_grid=param_grid,
            basic_parameters=basic_parameters,
            experiment_path=experiment_path,
            max_minutes=max_minutes,
            first_n_letters=folds_first_n_letters,
            decimals=folds_decimals,
            logs_folder=logs_folder,
            use_multiprocessing=use_multiprocessing,
        )