import os

import data_processing.data_preprocessing
import keras
import sklearn.preprocessing
import utils
from data_processing.dataset import CrystalDataset

crystal_basic_parameters = {
    "dataset": {
        "data_path": os.path.join(utils.find_root_path(), "python", "data"),
        "scaler": sklearn.preprocessing.MinMaxScaler(),
        "preprocessing_function": data_processing.data_preprocessing.lag_features,
        "shuffle": True,
        "final_testing": False,
        "test_size": 0.0,
        "outputs_lag": 3,
        "inputs_lag": 0,
        "window_size": 3,
        "return_sequences": True,
        "seed": 42,
        "scaler_partitions":[list(range(2)), list(range(2, 2+5)), list(range(2+5, 2+5+1))], # 2 inputs, 5 fluxes, 1 solid/liquid interface position
        "scaler_kwargs":{},
        "k_folds": 10,
    },
    "dataset_class": CrystalDataset,
    "model": {
        "bidirectional": False,
        "cell_type": "lstm",
        "rnn_cell_params": {
            "units": 20,
            "peephole_connections": False,
            "layer_normalization": False,
            "ln_epsilon": 1e-3,
            "ln_center": True,
            "ln_scale": True,
            "ln_beta_initializer": "zeros",
            "ln_gamma_initializer": "ones",
            "dropout": 0.0,
            "recurrent_dropout": 0.0
        },
        "num_layers": 1,
        "rnn_layer_params":{
            "return_sequences": True
        },
        "dilation_base": None,
        "residual_block_size": None,
        "smyl_std": False,
        "smyl_residual": False
    },
    "training": {
        "batch_size": 32,
        "epochs": 200,
        "optimizer": "rmsprop",
        "early_stopping": False,
        "model_checkpoint": True,
        "verbose": 0,
        "shuffle": True,
        "optimizer_params": {
            "learning_rate": 1e-4,
            "momentum": 0.99,
            "clipnorm": 2.0
        },
        "early_stopping_params": {
            "monitor": "val_loss",
            "patience": 10,
            "min_delta": 0,
            "mode": "min",
            "restore_best_weights": True
        },
        "model_checkpoint_params": {
            # Do not specify filepath here, it will be done in the training function
            "monitor": "val_loss",
            "verbose": 0,
            "save_best_only": True,
            "save_weights_only": False,
            "mode": "auto",
            "save_freq": "epoch",
            "initial_value_threshold": None,
        },
        "additional_callbacks": None
    }   
}

