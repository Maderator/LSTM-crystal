import keras
import utils
from predictor.models.predefined_networks import get_rnn_model_parameters_generalized
from predictor.models.rnn_constructor import RNNConstructor


def prepare_model_parameters(model_params, output_shape):
    return get_rnn_model_parameters_generalized(
        output_units=output_shape,
        cell_type=model_params["cell_type"],
        rnn_cell_params=model_params["rnn_cell_params"],
        num_layers=model_params["num_layers"],
        rnn_layer_params=model_params["rnn_layer_params"],
        dilation_base=model_params["dilation_base"],
        residual_block_size=model_params["residual_block_size"],
        smyl_std=model_params["smyl_std"],
        smyl_residual=model_params["smyl_residual"],
    )

def compile_model(
    model_params,
    training_params,
    input_shape,
    output_shape,
    loss=keras.losses.MeanSquaredError(),
    metric=keras.metrics.RootMeanSquaredError(),
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
    optimizer = utils.initialize_optimizer(
        training_params["optimizer"], **training_params["optimizer_params"]
    )
    model.compile(optimizer=optimizer, loss=loss, metrics=[metric])
    return model