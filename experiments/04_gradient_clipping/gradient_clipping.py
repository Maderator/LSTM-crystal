import argparse
import json
import logging
import os
import time

import numpy as np
import tensorflow as tf
import utils
from bayes_opt import BayesianOptimization
from bayes_opt.event import Events
from bayes_opt.logger import JSONLogger
from bayes_opt.util import load_logs
from data_processing.dataset import CrystalDatasetV1
from predictor.models.predefined_networks import get_rnn_model_parameters
from predictor.models.rnn_constructor import RNNConstructor
from sklearn.model_selection import KFold
from utils import find_root_path, initialize_optimizer

logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s", level=logging.INFO)


def initialize_dataset(params, root_path):
    data_path = params[
        "data_path"
    ]  # Path to the data folder from the root project directory
    data_directory = os.path.join(root_path, data_path)
    dataset = CrystalDatasetV1(
        training=params["training"],
        shuffled=params["shuffled"],
        test_size=0.0,
        seed=params["seed"],
        data_path=data_directory,
        preprocessing_type=params["preprocessing_type"],
        window_size=params["window_size"],
        scaling_type=params["scaling_type"],
        return_sequences=True,
    )
    return dataset


def get_input_output_shapes(dataset):
    input_shape = dataset.train_data["Inputs"].shape[1:]  # (timesteps, features)
    output_shape = dataset.train_data["Outputs"].shape[-1]  # (predicted_outputs)
    return input_shape, output_shape


def compile_and_fit(
    model,
    train,
    test,
    optimizer=tf.keras.optimizers.Adam(),
    batch_size=16,
    epochs=10,
    verbose=0,
    callbacks=None,
):
    loss = tf.keras.losses.MeanSquaredError()
    metric = tf.keras.metrics.RootMeanSquaredError()
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
    )  # callbacks=[tensorboard_callback])

def prepare_model_parameters(model_params, output_shape):
    return get_rnn_model_parameters(
        output_units=output_shape,
        cell_type=model_params["cell_type"],
        lstm_units=model_params["rnn_cell_params"]["units"],
        num_layers=model_params["num_layers"],
        return_sequences=model_params["rnn_layer_params"]["return_sequences"],
    )

def initialize_callbacks(
    training_params,
    csv_log_file="logs/training_log.csv",
):
    def create_log_dir(log_file):
        log_dir = os.path.dirname(log_file)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

    create_log_dir(csv_log_file)
    csv_logger = tf.keras.callbacks.CSVLogger(csv_log_file)
    if training_params["early_stopping"]:
        early_stop_params = training_params["early_stopping_params"]
    else:
        early_stop_params = {"monitor": "val_loss", "patience": np.inf}
    early_stopping = tf.keras.callbacks.EarlyStopping(**early_stop_params)
    return csv_logger, early_stopping

def compile_model(
    model_params,
    training_params,
    input_shape,
    output_shape,
    loss=tf.keras.losses.MeanSquaredError(),
    metric=tf.keras.metrics.RootMeanSquaredError(),
):
    out_layer, lstm_layers, dense_layers, residual_size = prepare_model_parameters(
        model_params, output_shape
    )
    model = RNNConstructor(
        input_shape=input_shape,
        output_layer_params=out_layer,
        rnn_layers_params=lstm_layers,
        dense_layers_parmas=dense_layers,
        residual_block_size=residual_size,
    )
    optimizer = initialize_optimizer(
        training_params["optimizer"], **training_params["optimizer_params"]
    )
    model.compile(optimizer=optimizer, loss=loss, metrics=[metric])
    return model


def optimization_function(clipnorm=2.0):
    root_path = find_root_path("LSTM-crystal-growth")
    cur_path = os.path.dirname(os.path.abspath(__file__))

    # Load dataset, model, and training parameters
    basic_parameters_path = os.path.join(
        root_path, "experiments", "basic_parameters.json"
    )

    
    params, dataset_params, model_params, training_params = utils.load_parameters(basic_parameters_path)
    training_params["clipnorm"] = clipnorm

    dataset = initialize_dataset(dataset_params, root_path)
    input_shape, output_shape = get_input_output_shapes(dataset)

    # TRAINING AND EVALUATION
    k_fold = KFold(n_splits=dataset_params["k_folds"])
    splits = k_fold.split(
        dataset.raw_train_data["Inputs"]
    )  # Whole dataset is in train_data field
    splits = list(splits)

    folds_rmse = []
    for i, (train_index, test_index) in enumerate(splits):
        logging.info(f"Fold {i+1}/{dataset_params['k_folds']}")

        cur_ds = dataset.reassign_and_scale_dataset(train_index, test_index)

        compiled_model = compile_model(
            model_params, training_params, input_shape, output_shape
        )

        log_file = f"{cur_path}/logs/clipnorm_{clipnorm}_fold_{i}.csv"
        callbacks = initialize_callbacks(training_params, log_file)

        compiled_model.fit(
            cur_ds.train_data["Inputs"],
            cur_ds.train_data["Outputs"],
            epochs=training_params["epochs"],
            batch_size=training_params["batch_size"],
            validation_data=(cur_ds.test_data["Inputs"], cur_ds.test_data["Outputs"]),
            shuffle=True,
            verbose=0,
            callbacks=callbacks,
        )

        mse, rmse = compiled_model.evaluate(
            cur_ds.test_data["Inputs"], cur_ds.test_data["Outputs"], verbose=0
        )
        logging.info(f"\tRMSE: {rmse}")

        folds_rmse.append(rmse)

    mean_rmse = np.mean(folds_rmse)
    logging.info(f"\tMean RMSE: {mean_rmse}")
    return mean_rmse


def negative_optimization_function(clipnorm=2.0):
    return -optimization_function(clipnorm)


def get_delta_minutes(start_time):
    return (time.time() - start_time) / 60.0


def main(max_minutes=60 * 20, init_points=5, old_logs=False):
    start_time = time.time()

    pbounds = {"clipnorm": (0.0, 100.0)}
    optimizer = BayesianOptimization(
        f=negative_optimization_function,
        pbounds=pbounds,
        verbose=2,
        allow_duplicate_points=False,
        random_state=1,
    )

    cur_path = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(cur_path, "logs", "gradient_clipping.json")

    logger = JSONLogger(path=config_path, reset=False)
    if old_logs and os.path.exists(config_path):
        load_logs(optimizer, logs=[config_path])

    optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)

    if len(optimizer.res) < init_points:
        init_points = init_points - logger._iterations
        logging.info(f"Starting optimization with {init_points} initial points")
        optimizer.maximize(
            init_points=init_points,
            n_iter=0,
        )

    while get_delta_minutes(start_time) < max_minutes:
        logging.info(f"Elapsed time: {get_delta_minutes(start_time)} minutes")
        optimizer.maximize(
            init_points=0,
            n_iter=1,
        )


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Prints debug and info messages to the console",
    )

    argparser.add_argument(
        "--max_minutes",
        type=int,
        default=60 * 20,
        help="Maximum number of minutes for optimization",
    )

    argparser.add_argument(
        "--init_points",
        type=int,
        default=5,
        help="Number of initial points for Bayesian optimization",
    )

    argparser.add_argument(
        "--old_logs",
        action="store_true",
        help="Load logs from previous optimization",
    )

    args = argparser.parse_args()
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    else:
        logging.getLogger().setLevel(logging.WARNING)

    main(args.max_minutes, args.init_points, args.old_logs)
