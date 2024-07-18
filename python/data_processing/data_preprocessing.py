import pickle
from typing import Tuple

import numpy as np
import pandas as pd


def identity(X: np.ndarray, y: np.ndarray, window_size: int, inputs_lag: int, outputs_lag: int) -> Tuple[np.ndarray, np.ndarray]:
    """Identity function that returns the input and output unchanged."""
    return (X, y)

def lag_features(X: np.ndarray, y: np.ndarray, inputs_lag=0, outputs_lag=3) -> Tuple[np.ndarray, np.ndarray]:
    """Add lag features.
    
    Authors of https://www.mdpi.com/2073-4352/11/2/138 for example use 3 previous timesteps outputs (outputs_lag=3) as lag features. Meaning that the input of each timestep consists of the current input and the three previous outputs. The target is the current output.

    Args:
        X: The inputs to process of shape ([num_samples,] num_timesteps, num_features) where num_samples is optional.
        y: The outputs to process with same number of dimensions as X.
        inputs_lag: The number of past input timesteps to consider.
        outputs_lag: The number of past output timesteps to consider.
        has_batch_dim: Whether the data has a batch dimension or not.
    Returns:
        The processed inputs and outputs.
    """
    # add batch dimension if there is none
    if X.ndim != y.ndim:
        raise ValueError("X and y must have the same number of dimensions.")
    missing_batch_dim = False
    if X.ndim == 2:
        missing_batch_dim = True
        X = X.reshape(1, X.shape[0], X.shape[1])
        y = y.reshape(1, y.shape[0], y.shape[1])

    lag_timesteps = max(inputs_lag, outputs_lag)

    time_dim = 1 # if has_batch_dim else 0
    feature_dim = 2 # if has_batch_dim else 1

    # get shape of processed input
    in_features_count = X.shape[feature_dim]
    out_features_count = y.shape[feature_dim]
    proc_in_time_shape = X.shape[time_dim] - lag_timesteps
    proc_in_feature_shape = in_features_count * (inputs_lag + 1) + out_features_count * outputs_lag # inputs_lag + 1 because we also include the current input

    # proc_in_shape = (X.shape[0],) if has_batch_dim else ()
    proc_in_shape = (X.shape[0], proc_in_time_shape, proc_in_feature_shape)

    # Initialize processed inputs with zeros
    processed_X = np.zeros(proc_in_shape)

    # add the current input (lag=0) to the processed inputs
    processed_X[:, :, :in_features_count] = X[:, lag_timesteps:, :]
    # add outputs from the previous outputs_lag timesteps to the processed inputs
    for lag in range(1, outputs_lag+1):
        first_index = in_features_count + out_features_count * (lag-1)
        last_index = first_index + out_features_count
        processed_X[:, :, first_index:last_index] = y[
            :, lag_timesteps-lag : -lag, :
        ]
    
    # add inputs from the previous inputs_lag timesteps to the processed inputs
    for lag in range(1, inputs_lag+1):
        first_index = in_features_count * lag + out_features_count * outputs_lag
        last_index = first_index + in_features_count
        processed_X[:, :, first_index:last_index] = X[
            :, lag_timesteps -lag : -lag, :
        ]

    processed_y = y[:, lag_timesteps:, :]

    # remove batch dimension if there was none
    if missing_batch_dim:
        processed_X = processed_X.reshape(processed_X.shape[1], processed_X.shape[2])
        processed_y = processed_y.reshape(processed_y.shape[1], processed_y.shape[2])

    return (processed_X, processed_y)

def windowed_preprocessing(X: np.ndarray, y: np.ndarray, window_size: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    """Creates windows of length window_size from the data with same features as the original data.

    Compared to the lag_features(), this function does not add the previous timesteps to the input. Instead this function creates subsequences of length window_size from the data. This way we get more training samples which are shorter (in sense of number of time steps) than the original data.
    """

    def create_windows(data: np.ndarray, window_size: int = 10) -> np.ndarray:
        """Given time series data, this function creates subsequences of length window_size.

        The data is expected to be in the shape ([num_samples,] num_timesteps, num_features). The output will be in the shape ([num_samples *] num_windows, window_size, num_features) where num_windows=num_samples-window_size+1.
        """
        num_samples = data.shape[1]
        num_windows = num_samples - window_size + 1
        data_w = []

        for i in range(num_windows):
            data_w.append(data[:, i : i + window_size, :])
        data_w = np.array(data_w)
        data_w = data_w.reshape(
            data_w.shape[0] * data_w.shape[1], data_w.shape[2], data_w.shape[3]
        )
        return data_w

    if X.ndim == 2 and y.ndim == 2:
        X = X.reshape(1, X.shape[0], X.shape[1])
        y = y.reshape(1, y.shape[0], y.shape[1])
            
    num_timesteps = X.shape[1]
    if window_size < 1 or window_size > num_timesteps:
        raise ValueError(
            f"window_size must be between 1 and {X.shape[1]} (number of timesteps in the original data). Got {window_size}."
        )

    processed_inputs = create_windows(X, window_size)
    processed_outputs = create_windows(y, window_size)
    return (processed_inputs, processed_outputs)

def cyclical_time_feature_encoding(dataframe: pd.DataFrame, column_name: str, min_period: str = 'hour'):
    """Transform date time series using cyclical time encoding.

    (For more details see Time-related feature engineering notebook by scikit-learn.org: https://scikit-learn.org/stable/auto_examples/applications/plot_cyclical_feature_engineering.html)    
    This way we can encode the cyclical nature of time. For example instead of having
    increasing numbers for hours in day (0 to 23) and therefore having a big difference between 23 and 0, we can encode the hours using sin and cos. This way the euclidean distance between each preceding and succeeding hours (in sin and cos features) will be similar.

    Args:
        period: The period  of the cyclical feature. For example for hours this would be 24 (hours in day).
        dataframe: The dataframe containing the time series.
        column_name: The name of the column containing the time series.
    Returns:
        The transformed time series as a tuple of two pandas series (sin_series, cos_series).
    """
    def encode_cyclical(df: pd.DataFrame, col:str, period: int):
        """Add sin and cos features to the dataframe for the given column."""
        df.loc[:, col + '_sin'] = np.sin(2 * np.pi * df[col]/period)
        df.loc[:, col + '_cos'] = np.cos(2 * np.pi * df[col]/period)
        return df
    
    dataframe = dataframe.copy()
    if min_period in ['month', 'day', 'hour']:
        dataframe.loc[:, 'month'] = dataframe[column_name].dt.month
        dataframe = encode_cyclical(dataframe, 'month', 12)
        dataframe.drop(columns=['month'], inplace=True)
    if min_period in ['day', 'hour']:
        # encode day of month
        dataframe.loc[:, 'day'] = dataframe[column_name].dt.day
        dataframe.loc[:, 'days_in_month'] = dataframe[column_name].dt.days_in_month
        dataframe = encode_cyclical(dataframe, 'day', dataframe['days_in_month'])
        # encode day of week
        dataframe.loc[:, 'day_of_week'] = dataframe[column_name].dt.dayofweek
        dataframe = encode_cyclical(dataframe, 'day_of_week', 7)
        dataframe.drop(columns=['day', 'days_in_month', 'day_of_week'], inplace=True)
        # encode day of year if min_period is day
        if min_period == 'day':
            dataframe.loc[:, 'day_of_year'] = dataframe[column_name].dt.dayofyear
            dataframe.loc[:, 'days_in_year'] = dataframe[column_name].dt.is_leap_year * 366 + (1 - dataframe[column_name].dt.is_leap_year) * 365
            dataframe = encode_cyclical(dataframe, 'day_of_year', dataframe['days_in_year'])
            dataframe.drop(columns=['day_of_year','days_in_year'], inplace=True)
    if min_period == 'hour':
        # encode hour of day
        dataframe.loc[:, 'hour'] = dataframe[column_name].dt.hour
        dataframe = encode_cyclical(dataframe, 'hour', 24)
        # encode hour of year
        dataframe.loc[:, 'hour_of_year'] = dataframe[column_name].dt.dayofyear * 24 + dataframe[column_name].dt.hour
        dataframe.loc[:, 'hours_in_year'] = dataframe[column_name].dt.is_leap_year * 366 * 24 + (1 - dataframe[column_name].dt.is_leap_year) * 365 * 24
        dataframe = encode_cyclical(dataframe, 'hour_of_year', dataframe['hours_in_year'])
        dataframe.drop(columns=['hour', 'hour_of_year', 'hours_in_year'], inplace=True)

    return dataframe

def _test_lag_features():
    timesteps = 4
    has_batch_dim = False
    in_features = 1
    out_features = 1 
    #X = np.arange(2*timesteps*2).reshape(2, timesteps, 2)
    #y = np.arange(2*timesteps*6).reshape(2, timesteps, 6) + X.size
    X = np.arange(timesteps*in_features).reshape(timesteps, in_features)
    y = np.arange(timesteps*out_features).reshape(timesteps, out_features) + X.size
    out_lag = 3
    in_lag = 3
    max_lag = max(out_lag, in_lag)

    X_lag, y_lag = lag_features(X, y, inputs_lag=in_lag, outputs_lag=out_lag, has_batch_dim=has_batch_dim)

    if not has_batch_dim:
        X = X.reshape(1, X.shape[0], X.shape[1])
        y = y.reshape(1, y.shape[0], y.shape[1])
        X_lag = X_lag.reshape(1, X_lag.shape[0], X_lag.shape[1])
        y_lag = y_lag.reshape(1, y_lag.shape[0], y_lag.shape[1])

    print("Raw X shape:", X.shape)
    print("Raw y shape:", y.shape)
    print("X features:", X.shape[2], ", lag y features:", y.shape[2]*out_lag, ", lag X features:", X.shape[2]*in_lag, ", total features:", X.shape[2] * (in_lag + 1) + y.shape[2] * out_lag)
    print("Preprocessed X shape:", X_lag.shape)
    print("Correct current inputs:", np.all(X_lag[:, :, :in_features] == X[:, in_lag:, :]))

    def check_lag_features(lags, array, num_features, first_feature_index):
        lag_correct = True
        for lag in range(1, 1 + lags):
            first_index = first_feature_index + num_features * (lag - 1)
            last_index = first_index + num_features

            if np.any(X_lag[:, :, first_index:last_index] != array[:, max_lag - lag : -lag, :]):
                lag_correct = False
                break
        return lag_correct

    out_correct = check_lag_features(out_lag, y, out_features, in_features)
    print("Correct previous outputs:", out_correct)

    in_correct = check_lag_features(in_lag, X, in_features, in_features + out_features * out_lag)
    print("Correct previous inputs:", in_correct)

    print("Correct lag outputs:", np.all(y_lag == y[:, max_lag:, :]))

if __name__ == "__main__":
    #_test_lag_features()
    #_test_dropka_preprocessing()
    pass
