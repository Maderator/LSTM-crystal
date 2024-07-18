import copy
import functools
import json
import logging
import os
import pickle
import re
import time

import keras.callbacks as keras_callbacks
import keras.optimizers as keras_optimizers
import sklearn.preprocessing
from tensorflow import keras

#################### General ####################

def find_root_path(
    root_directory_name: str="LSTM-crystal-growth", start_path: str = None, case_sensitive=False
) -> str:
    if start_path is None:
        # Set the start path to the directory of the current script
        start_path = os.path.dirname(os.path.abspath(__file__))

    while True:
        base = os.path.basename(start_path)
        dir = os.path.dirname(start_path)

        if (case_sensitive and root_directory_name in os.listdir(start_path)) or (
            not case_sensitive and base.lower() == root_directory_name.lower()
        ):
            return start_path
        elif dir == start_path:
            # Reached root which has no basename (e.g. C:\ == start_path (C:\))
            return None
        else:
            start_path = dir

def get_delta_minutes(start_time: float) -> float:
    return (time.time() - start_time) / 60.0

def dump_basic_params_and_changed_params(basic_parameters, param_grid, experiment_path, logs_folder):
    logs_path = os.path.join(experiment_path, logs_folder)
    os.makedirs(logs_path, exist_ok=True)
    with open(os.path.join(logs_path, "basic_parameters.pickle"), "wb") as f:
        pickle.dump(basic_parameters, f)
    with open(os.path.join(logs_path, "changed_params.pickle"), "wb") as f:
        pickle.dump(param_grid, f)

def serialize_callback(callback):
    """Can serialize only keras.callbacks.ReduceLROnPlateau callbacks for now.

    Args:
        callback (function): _description_

    Returns:
        Any(dictionary): Returns a dictionary with the callback configuration if the callback is supported.
    
    Raises:
        NotImplementedError: If the callback is not supported.
    """
    if isinstance(callback, keras_callbacks.ReduceLROnPlateau):
        return {
            "class_name": "ReduceLROnPlateau",
            "config": {
                "monitor": callback.monitor,
                "factor": callback.factor,
                "patience": callback.patience,
                "verbose": callback.verbose,
                "mode": callback.mode,
                "min_delta": callback.min_delta,
                "cooldown": callback.cooldown,
                "min_lr": callback.min_lr,
            }
        }
    raise NotImplementedError(f"Callback {callback} not supported.")

#################### Dictionaries ####################

def cast_dict_values(dict_values, dict_types):
    dict_values = copy.deepcopy(dict_values)
    for key in dict_values:
        if key in dict_types:
            cast_func = dict_types[key]
            if cast_func == bool:
                # use float range [0,2). Therefore, the values between 0 and 1 will be converted to 0 (False) and values between 1 and 2 will be converted to 1 (True)
                dict_values[key] = bool(int(dict_values[key]))
            else:
                dict_values[key] = cast_func(dict_values[key])
    return dict_values

def change_dict_values(
    original_dict: dict, changed_params: dict, separator: str = "-"
) -> dict:
    new_dict = copy.deepcopy(original_dict)
    for key, value in changed_params.items():
        separated_keys = key.split(separator)
        cur_dict = new_dict
        for i, sub_key in enumerate(separated_keys):
            if i == len(separated_keys) - 1:
                cur_dict[sub_key] = value

            elif sub_key in cur_dict:
                cur_dict = cur_dict[sub_key]  # reference, not copy
            else:
                logging.warning(f"Key {key} not found in dictionary")
                # raise KeyError(f"Key {key} not found in dictionary")
    return new_dict

def get_dictionary_short_string(
    dictionary: dict, first_n_letters: int = 2, decimals: int = 2, remove_non_filename_chars: bool = True
) -> str:
    """For each key get the first n letters of each word separated by underscores or hyphens and concatenate with the value rounded to m decimals"""

    def get_first_n_letters(string: str, n: int) -> str:
        return "".join([word[:n] for word in string.replace("-", "_").split("_")])

    key_vlaue_pairs = []
    for key, value in dictionary.items():
        letters = get_first_n_letters(key, first_n_letters)
        if isinstance(value, float):
            val = f"{round(value, decimals):.{decimals}f}"
        elif callable(value):
            if isinstance(value, functools.partial):
                val = value.func.__name__
            else:
                val = value.__name__
        elif isinstance(value, sklearn.preprocessing.FunctionTransformer):
            val = value.func.__name__
        elif isinstance(value, list):
            list_name = "["
            for i, v in enumerate(value):
                if i > 0:
                    list_name += ","
                list_name += v.__class__.__name__
            list_name += "]"
        else:
            val = str(value)
        key_vlaue_pairs.append(f"{letters}_{val}")
    short_string = "-".join(key_vlaue_pairs)
    
    if remove_non_filename_chars:
        short_string = re.sub(r'[\\/:*?"<>|]', "", short_string)
        # On Unix systems, you can use the following to remove all non-alphanumeric characters:
        # short_string = re.sub(r"[^a-zA-Z0-9_-]", "", short_string)
    return short_string


#################### Optimizers ####################

class NesterovMissingError(Exception):
    pass

class MomentumNotSupportedError(Exception):
    pass

def initialize_optimizer(optimizer_name="adam", **kwargs):
    """Initializes an optimizer from keras.optimizers.

    Args:
        optimizer_name (str, optional): Name of the optimizer. Defaults to "adam".
        **kwargs: Keyword arguments for the optimizer. See keras.optimizers (https://www.tensorflow.org/api_docs/python/tf/keras/optimizers) for supported optimizers and their arguments.

    Raises:
        ValueError: If the optimizer_name is not supported.

    Returns:
        keras.optimizers.Optimizer: Optimizer object.
    """
    supported_optimizers = {
        "adam": keras_optimizers.Adam,
        "sgd": keras_optimizers.SGD,
        "sgd_nesterov": keras_optimizers.SGD,
        "rmsprop": keras_optimizers.RMSprop,
        "nadam": keras_optimizers.Nadam,
        "adamw": keras_optimizers.AdamW,
        # Other optimizers not specified in the thesis
        "adadelta": keras_optimizers.Adadelta,
        "adafactor": keras_optimizers.Adafactor,
        "adagrad": keras_optimizers.Adagrad,
        "adamax": keras_optimizers.Adamax,
        "ftrl": keras_optimizers.Ftrl,
        "lion": keras_optimizers.Lion,
    }

    if optimizer_name not in supported_optimizers.keys():
        raise ValueError(
            f"Optimizer {optimizer_name} not supported. Supported optimizers are {list(supported_optimizers.keys())}."
        )

    if optimizer_name not in ["sgd", "sgd_nesterov"] and "nesterov" in kwargs and kwargs["nesterov"] is True:
        raise NesterovMissingError(
            f"Optimizer {optimizer_name} does not support the nesterov momentum."
        )

    if "momentum" in kwargs:
        if kwargs["momentum"] is None:
            # Remove None momentum from the kwargs
            kwargs.pop("momentum")
        elif optimizer_name not in ["sgd", "sgd_nesterov", "rmsprop"]:
            # Raise an error if the optimizer does not support the momentum
            raise MomentumNotSupportedError(
                f"Optimizer {optimizer_name} does not support the momentum."
            )

    if optimizer_name == "sgd_nesterov":
        kwargs["nesterov"] = True
        return supported_optimizers["sgd"](**kwargs)

    optimizer = supported_optimizers[optimizer_name](**kwargs)
    return optimizer

#################### Data processing ####################


def load_parameters(parameters_path: str):
    with open(parameters_path, "r") as file:
        params_dict = json.load(file)
    params = params_dict["parameters"]
    return params, params["dataset"], params["model"], params["training"]


def log_name_from_dicts(dicts: dict) -> str:
    log_name = ""
    for key, value in dicts.items():
        log_name += f"{key}_{value}_"
    return log_name[:-1]


if __name__ == "__main__":
    old_dict = {
        "a": {"b": {"c": 1}},
        "d": {"e": 2},
        "f": {"g": {"h": {"i": 3}}},
    }
    new_dict = change_dict_values(
        old_dict,
        {"a-b-c": 3, "d-e": 4},
    )
    print(old_dict)
    print(new_dict)

#################### Model training ####################

def compile_and_fit(
    model,
    train,
    test,
    optimizer=keras_optimizers.Adam(),
    batch_size=16,
    epochs=10,
    verbose=0,
    loss = keras.losses.MeanSquaredError(),
    metric = keras.metrics.RootMeanSquaredError(),
    callbacks = []
):
    loss = keras.losses.MeanSquaredError()
    metric = keras.metrics.RootMeanSquaredError()
    model.compile(optimizer=optimizer, loss=loss, metrics=[metric])
    model.fit(
        train["Inputs"],
        train["Outputs"],
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(test["Inputs"], test["Outputs"]),
        shuffle=True,
        verbose=verbose,
        callbacks=callbacks,
    ) 