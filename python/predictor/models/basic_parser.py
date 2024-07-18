def basic_parser(
    output_units=6,
    cell_type="lstm",
    lstm_units=30,
    dilation_base=2,
    residual_block_size=2,
    num_layers=4,
    return_sequences=False,
    peephole_connetions=False,
    residual_connections=False,
    layer_normalization=False,
    ln_epsilon=1e-3,
    ln_center=True,
    ln_scale=True,
    ln_beta_initializer="zeros",
    ln_gamma_initializer="ones",
    **kwargs,
):
    lstm_layers_params = []
    for i in range(num_layers):
        return_sequences = True
        if i == num_layers - 1:
            return_sequences = return_sequences
        layer_kwargs = {
            "cell_type": cell_type,
            "cell_kwargs": {
                "units": lstm_units,
                "peephole_connections": peephole_connetions,
                "residual_connection": residual_connections,
                "layer_normalization": layer_normalization,
                "ln_epsilon": ln_epsilon,
                "ln_center": ln_center,
                "ln_scale": ln_scale,
                "ln_beta_initializer": ln_beta_initializer,
                "ln_gamma_initializer": ln_gamma_initializer,
            },
            "layer_kwargs": {"return_sequences": return_sequences},
            "dilation_kwargs": {"dilation_rate": dilation_base**i},
        }
        lstm_layers_params.append(layer_kwargs)

    output_layer_params = {
        "units": output_units,
        "activation": None,  # linear adapter (adaptor)
    }
    return output_layer_params, lstm_layers_params, [], residual_block_size
