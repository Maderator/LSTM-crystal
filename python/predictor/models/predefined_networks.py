"""Functions that return dictionaries of network architectures parameters for Tensorflow 2.

Each function returns a dictionary with parameters for output layer, list of rnn
layers, list of dense layers, and residual block size. These parameters can then
be used to create a model using the rnn_constructor.py module.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np


def one_layered_lstm(
    output_units: int, lstm_units: int = 64, return_sequences: bool = False
) -> Tuple[Dict, List[Dict], List[Dict], Optional[int]]:
    """Basic one layered LSTM network

    Args:
        output_units (int): number of units in the last dense layer
        lstm_units (int): number of units in LSTM layer
        return_sequences (bool): whether to output whole sequences or only last output out of the model

    Returns:
        dict, list(dict), list(dict), int
        returns a dictionary with parameters for output layer, list of rnn layers, list out dense layers, and residual block size
    """
    lstm_layers = [
        {
            "cell_type": "lstm",
            "cell_kwargs": {"units": lstm_units},
            "layer_kwargs": {"return_sequences": return_sequences},
            "bidirectional": False,
        }
    ]
    output_layer_params = {
        "units": output_units,
    }
    return output_layer_params, lstm_layers, [], None


def smyl_std_lstm(
    output_units: int,
    cell_type: str = "lstm",
    lstm_units: int = 50,
    dilation_base: int = 2,
    residual_block_size: int = 2,
    num_layers: int = 4,
    return_sequences: bool = False,
) -> Tuple[Dict, List[Dict], List[Dict], Optional[int]]:
    """Standard LSTM architecture as defined by Smyl

    It consists of four layers of dillated lstm layers with residual connection between the input to third layer and the output of fourth layer.
    See A hybrid method of exponential smoothing and recurrent neural networks for time series forecasting by Smyl S. https://www.sciencedirect.com/science/article/pii/S0169207019301153 for more information

    Args:
        output_units (int): number of units in the last dense layer
        cell_type (string): either "lstm" or "gru" (type of rnn layer)
        lstm_units (int): number of units in rnn layer
        dilation_base (int): each layer has a dilation rate which is computed as dilation_base**layer_number
        residual_block_size (int or None): Size of rnn block which has a residual connection between its input and output.
        num_layers (int): number of rnn layers
        return_sequences (bool): whether to output whole sequences or only last output out of the model

    Returns:
        dict, list(dict), list(dict), int
        returns a dictionary with parameters for output layer, list of rnn layers, list out dense layers, and residual block size
    """
    lstm_layers_params = []

    for i in range(num_layers):
        return_sequences = True
        if i == num_layers - 1:
            return_sequences = return_sequences
        layer_kwargs = {
            "cell_type": cell_type,
            "cell_kwargs": {"units": lstm_units},
            "layer_kwargs": {"return_sequences": return_sequences},
            "dilation_kwargs": {"dilation_rate": dilation_base**i},
        }
        lstm_layers_params.append(layer_kwargs)

    output_layer_params = {
        "units": output_units,
        "activation": None,  # linear adapter (adaptor)
    }
    return output_layer_params, lstm_layers_params, [], residual_block_size


def smyl_residual_lstm(
    output_units,
    cell_type="lstm",
    lstm_units=50,
    dilation_base: Optional[int] = 3,
    residual_block_size: Optional[int] = None,
    num_layers: int = 4,
    return_sequences: bool = False,
) -> Tuple[Dict, List[Dict], List[Dict], Optional[int]]:
    """LSTM architecture as defined by Smyl with residual connections (Kim et al. style)

    It consists of four layers of dillated lstm layers with residual connections between the input of each layer and the output of a cell state.
    See A hybrid method of exponential smoothing and recurrent neural networks for time series forecasting by Smyl S. https://www.sciencedirect.com/science/article/pii/S0169207019301153 for more information.

    Args:
        output_units (int): number of units in the last dense layer
        cell_type (string): either "lstm" or "gru" (type of rnn layer)
        lstm_units (int): number of units in rnn layer
        dilation_base (int): each layer has a dilation rate which is computed as dilation_base**layer_number
        residual_block_size (int or None): Size of rnn block which has a residual connection between its input and output.
        num_layers (int): number of rnn layers
        return_sequences (bool): whether to output whole sequences or only last output out of the model

    Returns:
        dict, list(dict), list(dict), int
        returns a dictionary with parameters for output layer, list of rnn layers, list out dense layers, and residual block size
    """
    lstm_layers_params = []
    for i in range(num_layers):
        return_sequences = True
        if i == num_layers - 1:
            return_sequences = return_sequences
        layer_kwargs = {
            "cell_type": cell_type,
            "cell_kwargs": {"units": lstm_units, "residual_connection": True},
            "layer_kwargs": {"return_sequences": return_sequences},
        }
        if dilation_base is not None:
            layer_kwargs["dilation_kwargs"] = {"dilation_rate": dilation_base**i}
        lstm_layers_params.append(layer_kwargs)

    output_layer_params = {
        "units": output_units,
        "activation": None,  # linear adapter (adaptor)
    }
    return output_layer_params, lstm_layers_params, [], residual_block_size


def get_rnn_model_parameters(
    output_units: int,
    cell_type: str = "lstm",
    lstm_units: int = 50,
    num_layers: int = 1,
    return_sequences: bool = False,
    dilation_base: Optional[int] = None,
    residual_block_size: Optional[int] = None,
    smyl_std: bool = False,
    smyl_residual: bool = False,
) -> Tuple[Dict, List[Dict], List[Dict], Optional[int]]:
    """Create LSTM model parameters based on the specified architecture.

    Args:
        output_units (int): Number of units in the last dense layer.
        cell_type (str): Type of RNN layer ("lstm" or "gru").
        lstm_units (int): Number of units in RNN layer.
        num_layers (int): Number of RNN layers.
        return_sequences (bool): Whether to return sequences or only the last output.
        dilation_base (int): Base for calculating dilation rate.
        residual_block_size (int): Size of block with residual connections.
        smyl_std (bool): If True, use Smyl standard LSTM settings.
        smyl_residual (bool): If True, use Smyl residual LSTM settings.

    Returns:
        Tuple containing dictionaries for output layer, RNN layers, dense layers, and residual block size.
    """
    lstm_layers_params = []

    for i in range(num_layers):
        not_last_layer = i + 1 < num_layers
        layer_return_sequences = return_sequences or not_last_layer
        layer_kwargs = {
            "cell_type": cell_type,
            "cell_kwargs": {"units": lstm_units},
            "layer_kwargs": {"return_sequences": layer_return_sequences},
        }

        #if (smyl_std or smyl_residual) and dilation_base is not None:
        if dilation_base is not None:
            layer_kwargs["dilation_kwargs"] = {"dilation_rate": dilation_base**i}

        if smyl_residual:
            layer_kwargs["cell_kwargs"]["residual_connection"] = True

        lstm_layers_params.append(layer_kwargs)

    output_layer_params = {
        "units": output_units,
    }

    if smyl_residual:
        output_layer_params["activation"] = None  # linear adapter (adaptor)

    return output_layer_params, lstm_layers_params, [], residual_block_size


def get_rnn_model_parameters_generalized(
    output_units: int,
    cell_type: str = "lstm",
    rnn_cell_params: Dict = {"units": 50},
    num_layers: int = 1,
    rnn_layer_params: Dict = {"return_sequences": False},
    dilation_base: Optional[int] = None,
    residual_block_size: Optional[int] = None,
    smyl_std: bool = False,
    smyl_residual: bool = False,
) -> Tuple[Dict, List[Dict], List[Dict], Optional[int]]:
    """Create LSTM model parameters based on the specified architecture.

    Args:
        output_units (int): Number of units in the last dense layer.
        cell_type (str): Type of RNN layer ("lstm" or "gru").
        rnn_cell_params (Dict): Parameters for rnn cell as in tf.keras.layers.LSTMCell.
        num_layers (int): Number of RNN layers.
        rnn_layer_params (Dict): Parameters for rnn layer as in tf.keras.layers.LSTM.
        dilation_base (int): Base for calculating dilation rate.
        residual_block_size (int): Size of block with residual connections.
        smyl_std (bool): If True, use Smyl standard LSTM settings.
        smyl_residual (bool): If True, use Smyl residual LSTM settings.

    Returns:
        Tuple containing dictionaries for output layer, RNN layers, dense layers, and residual block size.
    """
    lstm_layers_params = []

    for i in range(num_layers):
        cur_layer_params = rnn_layer_params.copy()

        not_last_layer = i + 1 < num_layers
        cur_layer_params["return_sequences"] = (
            cur_layer_params["return_sequences"] or not_last_layer
        )  # If not last layer, always return sequences for proper connection with next layer
        layer_kwargs = {
            "cell_type": cell_type,
            "cell_kwargs": rnn_cell_params,
            "layer_kwargs": cur_layer_params,
        }

        if (smyl_std or smyl_residual) and dilation_base is not None:
            layer_kwargs["dilation_kwargs"] = {"dilation_rate": dilation_base**i}

        if smyl_residual:
            layer_kwargs["cell_kwargs"]["residual_connection"] = True

        lstm_layers_params.append(layer_kwargs)

    output_layer_params = {
        "units": output_units,
    }

    if smyl_std or smyl_residual:
        output_layer_params["activation"] = None  # linear adapter (adaptor)

    return output_layer_params, lstm_layers_params, [], residual_block_size
