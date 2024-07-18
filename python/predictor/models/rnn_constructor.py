import keras
import numpy as np

from . import custom_layers, rnn_cells

#import tensorflow as tf



class RNNConstructor(keras.Model):
    def __init__(
        self,
        input_shape,
        output_layer_params,
        rnn_layers_params=[],
        dense_layers_params=[],
        residual_block_size=None,
        **kwargs
    ):
        super().__init__()

        self.input_shape = input_shape
        self.output_layer_params = output_layer_params
        self.rnn_layers_params = rnn_layers_params
        self.dense_layers_params = dense_layers_params
        self.residual_block_size = residual_block_size

        self.supported_rnn_cells = {
            "gru": keras.layers.GRUCell,
            "lstm": rnn_cells.CustomLSTMCell,
        }

        self.rnn_layers = []
        self.dense_layers = []
        self.residual_block_size = residual_block_size

        #self.input_layer = keras.layers.InputLayer(input_shape=input_shape)

        for layer in rnn_layers_params:
            cell = self.construct_cell(layer["cell_type"], layer["cell_kwargs"])

            if "dilation_kwargs" in layer.keys():
                if "return_sequences" in layer["layer_kwargs"]:
                    # If we are using dilated RNNs, we need to set return_sequences to True
                    # as the dilation computation needs whole sequence.
                    # The dilation layer replaces the RNN layer return logic.
                    layer["dilation_kwargs"]["return_sequences"] = layer[
                        "layer_kwargs"
                    ]["return_sequences"]
                    layer["layer_kwargs"]["return_sequences"] = True
                if "return_state" in layer["layer_kwargs"]:
                    # If we are using dilated RNNs, we need to input return_state to the dilation layer kwargs
                    layer["dilation_kwargs"]["return_state"] = layer["layer_kwargs"][
                        "return_state"
                    ]
            rnn_layer = keras.layers.RNN(cell, **layer["layer_kwargs"])
            if "bidirectional" in layer.keys() and layer["bidirectional"]:
                rnn_layer = keras.layers.Bidirectional(rnn_layer)
            if "dilation_kwargs" in layer.keys():
                rnn_layer = custom_layers.DilatedRNNLayer(
                    rnn_layer, **layer["dilation_kwargs"]
                )
            self.rnn_layers.append(rnn_layer)

        for layer in dense_layers_params:
            dense_layer = keras.layers.Dense(**layer["layer_kwargs"])
            self.dense_layers.append(dense_layer)

        self.output_layer = keras.layers.Dense(**output_layer_params)

    def construct_cell(self, cell_type, cell_kwargs):
        assert cell_type in self.supported_rnn_cells.keys()
        cell = self.supported_rnn_cells[cell_type](**cell_kwargs)
        return cell

    def call(self, inputs, training=False):
        x = inputs
        #x = self.input_layer(inputs)
        res_in = None
        for i, layer in enumerate(self.rnn_layers):
            x = layer(x)

            # If we are using residual blocks, we need to add the residual connection whenever we reach the end of a block (i.e. i+1 % residual_block_size == 0)
            if (
                self.residual_block_size is not None
                and (i + 1) % self.residual_block_size == 0
            ):
                if res_in is not None:
                    if not layer.return_sequences:
                        res_in = res_in[:, -1, :]
                    x = x + res_in
                res_in = x
        for layer in self.dense_layers:
            x = layer(x)
        x = self.output_layer(x)
        return x

    def build_graph(self):
        """builds models graph so that it can be visualized"""
        x = keras.Input(shape=self.input_shape)
        return keras.Model(inputs=[x], outputs=self.call(x))

    def summary(self):
        self.build_graph().summary()

    def plot_model(self):
        return keras.utils.plot_model(
            self.build_graph(), expand_nested=True, show_shapes=True
        )

    def count_trainable_parameters(self):
        self.build_graph()
        return np.sum(
            [np.prod(v.get_shape().as_list()) for v in self.trainable_variables]
        )

    def get_config(self):
        config = super().get_config()
        config.update({
            "input_shape": self.input_shape,
            "output_layer_params": self.output_layer_params,
            "rnn_layers_params": self.rnn_layers_params,
            "dense_layers_params": self.dense_layers_params,
            "residual_block_size": self.residual_block_size
            })
        return config
    
    #@classmethod
    #def from_config(cls, config):
    #    return cls(**config)
