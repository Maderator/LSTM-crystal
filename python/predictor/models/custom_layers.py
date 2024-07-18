import tensorflow as tf


class DilatedRNNLayer(tf.keras.layers.Layer):
    """Dilated RNN layer for sequence data.

    This implementation was inspired by the implementation of the authors of paper introducing the Dilated RNN layer [Chang S. et al., 2017](http://arxiv.org/abs/1710.02224) which is publicly available on GitHub: https://github.com/code-terminator/DilatedRNN
    Unfortunately, authors used old version of Tensorflow 1 and Python 2. Therefore it is incompatible with our code so we reimplemented it using Tensorflow 2 and Python 3.

    This layer performs a sequence dilation on the input tensor and then processes the sequence using a RNN cell.

    See original paper [Chang S. et al., 2017](http://arxiv.org/abs/1710.02224) for more information about dilated RNNs.
    """

    def __init__(
        self,
        rnn_layer,
        dilation_rate: int = 2,
        return_sequences: bool = False,
        return_state: bool = False,
        **kwargs,
    ):
        """Initialize the DilatedRNNLayer.

        Args:
            rnn_layer: instance of rnn layer to use for processing the dilated sequence (e.g. tf.keras.layers.RNN, tf.keras.layers.LSTM, ...). This allows use of wrappers such as Bidirectional.
            dilation_rate: dilation rate of the layer
        """
        super().__init__(**kwargs)
        assert (
            dilation_rate >= 1
        ), f"Dilation rate must be greater or equal to 1. Got {dilation_rate} instead."
        assert (
            rnn_layer.return_sequences == True
        ), "DilatedRNNLayer must be used with rnn_layer which has parameter return_sequences=True"
        self.rnn_layer = rnn_layer
        self.dilation_rate = int(dilation_rate)
        self.return_sequences = bool(return_sequences)
        self.return_state = bool(return_state)

    def build(self, input_shape):
        super().build(input_shape)
        # # time_steps are usually of type None. Therefore we cannot assert if the dilation rate is smaller than number of time_steps
        # time_steps = input_shape[1]
        # # print(f"in_shape: {input_shape}")
        # assert (
        #    self.dilation_rate < time_steps
        # ), f"Dilation rate must be smaller than the length of the input sequence. Got dilation rate {self.dilation_rate} while input sequence length is {time_steps}."

    def call(self, inputs):
        # We assume that inputs have 3 dimensions (batch, time_steps, features)
        batch_size = tf.shape(inputs)[0]
        time_steps = tf.shape(inputs)[1]
        features = tf.shape(inputs)[2]

        # tf.math.mod needs to have same type for both arguments
        dilation_rate_same_type = tf.dtypes.cast(self.dilation_rate, dtype=time_steps.dtype)

        # 1. If the length of inputs is not divisible by rate, pad the inputs with zeros
        remainder = tf.math.mod(time_steps, dilation_rate_same_type)
        not_divisible = tf.math.not_equal(remainder, 0)

        def pad_inputs():
            # as depicted in figure 1 of the original paper we need to create
            # pairs of cells with the same color. Therefore for the last
            # self.dilation_rate-1 cells that are not divisible by rate we need
            # to pad the inputs so that each cell has someone to pair with.
            padding_size = self.dilation_rate - remainder
            # pad the end of the second dimension (time_steps)
            padding = [[0, 0], [0, padding_size], [0, 0]]
            return tf.pad(inputs, padding)

        padded_inputs = tf.cond(not_divisible, pad_inputs, lambda: inputs)

        floor_n_steps = tf.math.floordiv(time_steps, self.dilation_rate)
        dilated_n_steps = tf.cond(
            not_divisible, lambda: floor_n_steps + 1, lambda: floor_n_steps
        )

        # 2. Perform the dilation
        dilated_inputs = tf.reshape(
            padded_inputs,
            [batch_size * self.dilation_rate, dilated_n_steps, features],
        )
        dilated_outputs = self.rnn_layer(dilated_inputs)

        if self.return_state:
            states = dilated_outputs[1:]
            dilated_outputs = dilated_outputs[0]

        # 3. Unroll the dilated outputs (as in figure 1 of the original paper from the rightmost configuration to the unrolled output in the middle)
        padded_inputs_time_steps = tf.shape(padded_inputs)[1]
        out_features = tf.shape(dilated_outputs)[-1]
        unrolled_outputs = tf.reshape(
            dilated_outputs, [batch_size, padded_inputs_time_steps, out_features]
        )

        # 4. Remove padded zeros
        if self.return_sequences:
            outputs = unrolled_outputs[:, :time_steps, :]
        else:
            # return last time step output
            outputs = unrolled_outputs[:, time_steps - 1, :]

        if self.return_state:
            outputs = [outputs] + states

        return outputs
