import logging
import os
from typing import List, Union

import keras
import numpy as np
import utils
from optimization import compilation
from optimization.data_classes import TrainingResults

def initialize_callbacks(
    training_params,
    csv_log_file="logs/training_log.csv",
    model_filepath="logs/model.keras",
) -> List:
    def create_log_dir(log_file):
        log_dir = os.path.dirname(log_file)
        os.makedirs(log_dir, exist_ok=True)

    loggers = []

    # Early stopping
    if training_params["early_stopping"]:
        early_stop_params = training_params["early_stopping_params"]
    else:
        early_stop_params = {"monitor": "val_loss", "patience": np.inf}
    loggers.append(keras.callbacks.EarlyStopping(**early_stop_params))

    # Learning rate scheduler
    #if "lr_epoch_decay" in training_params:
    #    def lr_scheduler(epoch, lr):
    #        if epoch > 0:
    #            # decay learning rate by lr_epoch_decay after every epoch
    #            return lr * training_params["lr_epoch_decay"]
    #        return lr
    if "lr_schedule" in training_params:
        lr_schedule = training_params["lr_schedule"]
    else:
        def lr_schedule(epoch, lr):
            return lr


    lr_callback = keras.callbacks.LearningRateScheduler(lr_schedule)
    loggers.append(lr_callback)

    #lr_logger = LearningRateLogger()
    #loggers.append(lr_logger)

    # Model checkpoint
    if training_params["model_checkpoint"]:
        model_checkpoint_params = training_params["model_checkpoint_params"]
        create_log_dir(os.path.dirname(model_filepath))
        model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=model_filepath, **model_checkpoint_params)
        loggers.append(model_checkpoint)

    # CSV Logger
    create_log_dir(csv_log_file)
    loggers.append(keras.callbacks.CSVLogger(csv_log_file))

    return loggers

def train_model(
    train_x, train_y, val_x, val_y,
    model_params,
    training_params,
    input_shape,
    output_shape,
    log_filepath,
    model_filepath,
) -> keras.Model:
    compiled_model = compilation.compile_model(
        model_params, training_params, input_shape, output_shape
    )
    callbacks = initialize_callbacks(training_params=training_params, csv_log_file=log_filepath, model_filepath=model_filepath)
    if "additional_callbacks" in training_params and training_params["additional_callbacks"]: #is not None
        assert isinstance(training_params["additional_callbacks"], list)
        callbacks += training_params["additional_callbacks"]


    compiled_model.fit(
        train_x,
        train_y,
        epochs=training_params["epochs"],
        batch_size=training_params["batch_size"],
        validation_data=(val_x, val_y),
        shuffle=training_params["shuffle"],
        verbose=training_params["verbose"],
        callbacks=callbacks,
    )
    return compiled_model

def evaluate_model(
    model: keras.Model,
    test_x: np.ndarray,
    test_y: np.ndarray,
    reverse_scale_outputs_func: callable,
    prediction_window: int = 1,
) -> TrainingResults:
    if prediction_window > 1:
        num_samples = test_x.shape[0]
        inputs, true_y = [], []
        for i in range(num_samples-prediction_window):
            sample = test_x[i]
            current_input = sample[np.newaxis, :, :]
            inputs.append(current_input)

            true_values = test_y[i:i+prediction_window]
            true_y.append(true_values)

        inputs = np.concatenate(inputs, axis=0)
        true_y = np.array(true_y)

        pred_y = []
        unscaled_pred_y, unscaled_true_y = [], []
        for i in range(prediction_window):
            pred = model.predict(inputs)
            pred_y.append(pred)
            unscaled_pred_y.append(reverse_scale_outputs_func(pred))
            unscaled_true_y.append(reverse_scale_outputs_func(true_y[:, i]))
            inputs = np.concatenate((inputs[:, 1:], pred[:, np.newaxis, :]), axis=1)

        pred_y = np.array(pred_y)
        unscaled_pred_y = np.array(unscaled_pred_y)
        unscaled_true_y = np.array(unscaled_true_y)
        pred_y = np.swapaxes(pred_y, 0, 1)
        unscaled_pred_y = np.swapaxes(unscaled_pred_y, 0, 1)
        unscaled_true_y = np.swapaxes(unscaled_true_y, 0, 1)

    else:
        pred_y = model.predict(test_x, verbose=0)
        true_y = test_y
        unscaled_pred_y = reverse_scale_outputs_func(pred_y)
        unscaled_true_y = reverse_scale_outputs_func(test_y)

    results = TrainingResults(
        model,
        pred_y,
        true_y,
        unscaled_pred_y,
        unscaled_true_y,
    )
    logging.info(results)

    return results

def split_training(
    changed_params,
    model_params,
    training_params,
    dataset_params,
    dataset,
    input_shape,
    output_shape,
    i,
    cur_path,
    logs_folder="logs/main",
    first_n_letters=1,
    decimals=2,
    parallel_training=False,
    manager_dict=None,
) -> TrainingResults:

    logging.info(f"Fold {i+1}/{dataset_params['k_folds']}")
    params_shortcut = utils.get_dictionary_short_string(
        changed_params, first_n_letters=first_n_letters, decimals=decimals
    )
    log_file = os.path.join(cur_path, logs_folder, "training_loss_metrics", f"{params_shortcut}_fold{i}.csv")
    model_file = os.path.join(cur_path, logs_folder, "models", f"{params_shortcut}_fold{i}.keras")

    train_x = dataset.train_data["Inputs"]
    train_y = dataset.train_data["Outputs"]
    val_x = dataset.test_data["Inputs"]
    val_y = dataset.test_data["Outputs"]

    trained_model = train_model(
        train_x=train_x,
        train_y=train_y,
        val_x=val_x,
        val_y=val_y,
        model_params=model_params,
        training_params=training_params,
        input_shape=input_shape,
        output_shape=output_shape,
        log_filepath=log_file,
        model_filepath=model_file,
    )

    results = evaluate_model(
        trained_model,
        val_x,
        val_y,
        dataset.reverse_scale_outputs,
        prediction_window=dataset_params.get("prediction_window", 1),
    )

    if parallel_training:
        manager_dict[i] = results
    else:
        return results