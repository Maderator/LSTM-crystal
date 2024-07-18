import copy
import logging
import os
import pickle

import basic_parameters
import data_processing.data_preprocessing as data_preprocessing
import keras
import optimization.optimization as optimization
import optimization.training as training
import sklearn.preprocessing
import tqdm
import utils
from experiments_utils import get_experiment_label, load_final_testing_experiments

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

def load_crystal_dataset():
    params = basic_parameters.crystal_basic_parameters
    params["dataset"]["scaler"] = sklearn.preprocessing.StandardScaler()
    params["dataset"]["preprocessing_function"] = data_preprocessing.lag_features
    params["dataset"]["return_sequences"] = True
    params["dataset"]["final_testing"] = True
    root_path = utils.find_root_path("LSTM-crystal-growth")
    crystal_dataset = optimization.initialize_class(params["dataset"], root_path, params["dataset_class"])
    return crystal_dataset

def load_models_per_folds(id: str, models_path, folds_num: int=10):
    models = []
    for i in range(folds_num):
        path = os.path.join(models_path, f"{id}_fold{i}.keras")
        model = keras.models.load_model(path)
        #print(model.summary())
        models.append(model)
    return models

def load_all_experiment_models(
    exp_log_path: str,
    results_filename:str="gridsearch_results.pkl"
):
    grid_path = os.path.join(exp_log_path, results_filename)
    models_path = os.path.join(exp_log_path, "models")

    with open(grid_path, "rb") as f:
        results = pickle.load(f)

    models = []

    logger.info(f"loading experiments")
    for res in tqdm.tqdm(results):
        cur_params = copy.deepcopy(res)
        cur_params.pop("metrics_per_fold")
        params_shortcut = utils.get_dictionary_short_string(cur_params, first_n_letters=1, decimals=2)

        model_data = {
            "label": get_experiment_label(res),
            "training_results" : res,
            "id": params_shortcut,
            "fold_models": load_models_per_folds(
                                id=params_shortcut,
                                models_path=models_path
                            )
        }
        models.append(model_data)
    return models

def test_model(model, dataset):
    testing_results = training.evaluate_model(
        model=model,
        test_x = dataset.test_data["Inputs"],
        test_y = dataset.test_data["Outputs"],
        reverse_scale_outputs_func = dataset.reverse_scale_outputs,
        prediction_window=1,
    )
    return testing_results

def test_models(models_data, dataset):
    for model_data in models_data:
        logger.info(f"testing model {model_data["label"]}")
        test_results = []
        for model in tqdm.tqdm(model_data["fold_models"]):
            res = test_model(model, dataset)
            test_results.append(res)
        model_data["test_results"] = test_results
    return models_data

def prepare_pickleable_data(models_data):
    data = copy.deepcopy(models_data)
    for model_data in data:
        model_data.pop("fold_models")
    return data

def main(
    cur_exp_path, 
    results_filename:str="gridsearch_results.pkl", 
    testing_output_filename:str="testing_results.pkl"
):
    models_data = load_all_experiment_models(cur_exp_path, results_filename=results_filename)
    dataset = load_crystal_dataset()
    models_data = test_models(models_data, dataset)
    pickleable_data = prepare_pickleable_data(models_data)

    results_path = os.path.join(cur_exp_path, testing_output_filename)
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    with open(results_path, "wb") as f:
        pickle.dump(pickleable_data, f)
    
def look_at_testing_results():
    path_1l_300 = r"C:\Users\janma\Programovani\diplomova_prace\LSTM-crystal-growth\experiments\09_test_dataset_eval\logs\multi_layer_dynamic_1l_300"

    cur_exp_path = path_1l_300
    results_path = os.path.join(cur_exp_path, "testing_results.pkl")
    with open(results_path, "rb") as f:
        results = pickle.load(f)
    logger.info(results)

if __name__ == "__main__":
    path_dyn_10l = r"C:\Users\janma\Programovani\diplomova_prace\LSTM-crystal-growth\experiments\09_test_dataset_eval\logs\multi_layer_dynamic_10l"
    path_1l_300 = r"C:\Users\janma\Programovani\diplomova_prace\LSTM-crystal-growth\experiments\09_test_dataset_eval\logs\multi_layer_dynamic_1l_300"
    multi_layer_static = r"C:\Users\janma\Programovani\diplomova_prace\LSTM-crystal-growth\experiments\09_test_dataset_eval\logs\multi_layer_static"

    #main(path_dyn_10l)
    #main(path_1l_300)
    #main(multi_layer_static, results_filename="gridsearch_results.pkl", testing_output_filename="testing_results_static.pkl")
    main(multi_layer_static, results_filename="gridsearch_results_1l.pkl", testing_output_filename="testing_results_1l.pkl")
    #look_at_testing_results()