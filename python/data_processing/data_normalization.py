import copy
import pickle

import numpy as np
import scipy.stats
import sklearn
import sklearn.preprocessing
from scipy import sparse


class PartitionsScaler:
    """Scales the data with the given class. If partitions is not None, the data is scaled separately for each features partition.

    The partitions are needed only if the scaler should not scale all the features (Only one partition is needed), or if the scaler is global scaler (i.e. it scales multiple features together).

    Example of partitions: [[0,1,2], [3,5]] means that the first three features are scaled together and the features 3, and 5 are scaled together.

    Args:
        scaler_class: The class of the scaler to use.
        partitions: The partitions of the features to scale together.
    """
    def __init__(self, scaler_class, partitions=None, scaler_kwargs={}):
        self.scaler_class = scaler_class
        self.partitions = partitions
        self._check_partition_validity()
        self.scalers = []
        self.scaler_kwargs = scaler_kwargs

    def fit(self, X, y=None, partitions=None):
        """Fit the scaler on the data."""
        self.scalers = []
        if partitions:
            self.partitions = partitions
            self._check_partition_validity(X)
        if self.partitions:
            for partition in self.partitions:
                partition_scaler = self.scaler_class(**self.scaler_kwargs)
                partition_scaler.fit(X[:,partition])
                self.scalers.append(partition_scaler)
        else:
            scaler = self.scaler_class(**self.scaler_kwargs)
            scaler.fit(X)
            self.scalers.append(scaler)

    def transform(self, X, y=None):
        if self.partitions:
            transformed_data = X.copy()
            for partition, scaler in zip(self.partitions, self.scalers):
                transformed_data[:,partition] = scaler.transform(X[:,partition])
        else:
            transformed_data = self.scalers[0].transform(X)
        return transformed_data

    def inverse_transform(self, X, y=None):
        if self.partitions:
            transformed_data = X.copy()
            for partition, scaler in zip(self.partitions, self.scalers):
                transformed_data[:,partition] = scaler.inverse_transform(X[:,partition])
        else:
            transformed_data = self.scalers[0].inverse_transform(X)
        return transformed_data

    def _check_partition_validity(self, X=None):
        if self.partitions:
            for p in self.partitions:
                if not isinstance(p, list):
                    raise ValueError("Partitions must be a list of lists.")
                for i in p:
                    if not isinstance(i, int):
                        raise ValueError("Partitions must be a list of lists of integers.")
                    if i < 0:
                        raise ValueError("Partitions must be a list of lists of non-negative integers.")
                    if X is not None and i >= X.shape[1]:
                        raise ValueError("Partitions must be a list of lists of integers smaller than the number of features in X.") 

class GlobalMinMaxScaler(sklearn.preprocessing.MinMaxScaler):
    """Scales the data between 0 and 1 using the global minimum and maximum instead of min and max for each feature separately.
    
    This is a custom implementation of sklearn.preprocessing.MinMaxScaler (see https://github.com/scikit-learn/scikit-learn/blob/3f89022fa/sklearn/preprocessing/_data.py#L276).
    
    Instead of using axis=0, we use axis=None to calculate the global minimum and maximum.

    Use only fit, transform and inverse_transform methods as other methods are not properly implemented.
    """
    def fit(self, X, y=None):
        self._reset()

        feature_range = self.feature_range
        if feature_range[0] >= feature_range[1]:
            raise ValueError(
                "Minimum of desired feature range must be smaller than maximum. Got %s."
                % str(feature_range)
            )

        if sparse.issparse(X):
            raise TypeError(
                "GlobalMinMaxScaler does not support sparse input. "
                "Consider using MaxAbsScaler instead."
            )

        data_min = np.nanmin(X, axis=None) # axis is None instead of 0 as in sklearn.preprocessing.MinMaxScaler
        data_max = np.nanmax(X, axis=None) # axis is None

        data_range = data_max - data_min
        self.scale_ = (feature_range[1] - feature_range[0]) / sklearn.preprocessing._data._handle_zeros_in_scale(data_range, copy=True)

        self.min_ = feature_range[0] - data_min * self.scale_
        self.data_min_ = data_min
        self.data_max_ = data_max
        self.data_range_ = data_range

        return self

    # transform(self, X) same as in sklearn.preprocessing.MinMaxScaler
    # inverse_transform(self, X) same as in sklearn.preprocessing.MinMaxScaler

class GlobalMaxAbsScaler(sklearn.preprocessing.MaxAbsScaler):
    """Scales the data between -1 and 1 using the global maximum absolute value instead of maxabs for each feature separately.
    
    This is a custom implementation of sklearn.preprocessing.MaxAbsScaler (see https://github.com/scikit-learn/scikit-learn/blob/3f89022fa/sklearn/preprocessing/_data.py#L1075).

    Instead of using axis=0, we use axis=None to calculate the global maximum absolute value.

    Use only fit, transform and inverse_transform methods as other methods are not properly implemented.
    """
    def fit(self, X, y=None):
        self._reset()
        if sparse.issparse(X):
            raise TypeError(
                "GlobalMaxAbsScaler does not support sparse input. "
                "Consider using MaxAbsScaler instead."
            )
        self.max_abs_ = np.nanmax(np.abs(X), axis=None) # axis is None instead of 0
        self.scale_ = sklearn.preprocessing._data._handle_zeros_in_scale(self.max_abs_, copy=True)
        return self

class GlobalStandardScaler(sklearn.preprocessing.StandardScaler):
    """Scales the data to have mean 0 and standard deviation 1 using the global mean and standard deviation instead of mean and std for each feature separately.
    
    This is a custom implementation of sklearn.preprocessing.StandardScaler (see https://github.com/scikit-learn/scikit-learn/blob/3f89022fa/sklearn/preprocessing/_data.py#L657)

    Instead of using axis=0, we use axis=None to calculate the global mean and standard deviation.

    Use only fit, transform and inverse_transform methods as other methods are not properly implemented.
    """
    def fit(self, X, y=None, sample_weight=None):
        self._reset()
        if sparse.issparse(X):
            raise TypeError(
                "GlobalStandardScaler does not support sparse input. "
                "Consider using StandardScaler instead."
            )
        if sample_weight is not None:
            raise NotImplementedError("sample_weight is not implemented for GlobalStandardScaler")
        
        if not self.with_mean and not self.with_std:
            self.mean_ = None
            self.var_ = None
        else:
            self.mean_ = np.nanmean(X, axis=None) # axis is None instead of 0
            self.var_ = np.nanvar(X, axis=None) # axis is None instead of 0

        if self.with_std:
            self.scale_ = sklearn.preprocessing._data._handle_zeros_in_scale(np.sqrt(self.var_), copy=True)
        else:
            self.scale_ = None
        return self

class GlobalRobustScaler(sklearn.preprocessing.RobustScaler):

    def __init__(
        self,
        *,
        with_centering=True,
        with_scaling=True,
        quantile_range=(25.0, 75.0),
        copy=True,
        unit_variance=False,
    ):
        super().__init__(with_centering=with_centering, with_scaling=with_scaling, quantile_range=quantile_range, copy=copy, unit_variance=unit_variance)

    def fit(self, X, y=None):
        q_min, q_max = self.quantile_range
        if not 0 <= q_min <= q_max <= 100:
            raise ValueError("Invalid quantile range: %s" % str(self.quantile_range))

        if self.with_centering:
            if sparse.issparse(X):
                raise ValueError(
                    "Cannot center sparse matrices: use `with_centering=False`"
                    " instead. See docstring for motivation and alternatives."
                )
            self.center_ = np.nanmedian(X, axis=None) # axis is None instead of 0
        else:
            self.center_ = None

        if self.with_scaling:
            quantiles = []
            num_features = X.shape[1]
            if sparse.issparse(X):
                raise ValueError("GlobalRobustScaler does not support sparse input scaling. Consider using RobustScaler instead.")
            single_quantiles = np.nanpercentile(X, self.quantile_range)
            quantiles = np.tile(single_quantiles, (num_features, 1))
            quantiles = np.transpose(quantiles)

            self.scale_ = quantiles[1] - quantiles[0]
            self.scale_ = sklearn.preprocessing._data._handle_zeros_in_scale(self.scale_, copy=False)
            if self.unit_variance:
                adjust = scipy.stats.norm.ppf(q_max / 100.0) - scipy.stats.norm.ppf(q_min / 100.0)
                self.scale_ = self.scale_ / adjust
        else:
            self.scale_ = None

        return self

        

def minmax_scale_numpy_array(array, min=None, max=None):
    if min is None:
        min = array.min()
    if max is None:
        max = array.max()
    delta = max - min
    return (array - min) / delta, min, max


def minmax_scale_data(data, min=[None] * 3, max=[None] * 3):
    """Normalize the data between minimum and maximum to interval [0,1].

    Normalizes the heat fluxes, the temperatures and the interface position separately.
    Args:
        data: The data to normalize.
        min: The minimum value to use for normalization.
        max: The maximum value to use for normalization.
    Returns:
        normalized_data
        min
        max
    """
    if data["Inputs"] is None or data["Outputs"] is None:
        return data, min, max

    heat_fluxes, min[0], max[0] = minmax_scale_numpy_array(
        data["Inputs"], min[0], max[0]
    )

    temperatures, min[1], max[1] = minmax_scale_numpy_array(
        data["Outputs"][:, :, :5], min[1], max[1]
    )
    interface_position, min[2], max[2] = minmax_scale_numpy_array(
        data["Outputs"][:, :, 5], min[2], max[2]
    )

    normalized_data = {}
    normalized_data["Inputs"] = heat_fluxes
    normalized_data["Outputs"] = np.concatenate(
        (temperatures, interface_position[:, :, None]), axis=2
    )
    return normalized_data, min, max


def maxabs_scale_numpy_array(array, maxabs=None):
    if maxabs is None:
        maxabs = np.absolute(array).max()
    return array / maxabs, maxabs


def maxabs_scale_data(data, maxabs=[None] * 3):
    """Normalize the data by dividing all the values by maximum absolute value.

    Normalizes the heat fluxes, the temperatures and the interface position separately.
    Args:
        data: The data to normalize.
        min: The minimum value to use for normalization.
        max: The maximum value to use for normalization.
    Returns:
        normalized_data
        min
        max
    """
    if data["Inputs"] is None or data["Outputs"] is None:
        return data, maxabs

    heat_fluxes, maxabs[0] = maxabs_scale_numpy_array(data["Inputs"], maxabs[0])

    temperatures, maxabs[1] = maxabs_scale_numpy_array(
        data["Outputs"][:, :, :5], maxabs[1]
    )
    interface_position, maxabs[2] = maxabs_scale_numpy_array(
        data["Outputs"][:, :, 5], maxabs[2]
    )

    normalized_data = {}
    normalized_data["Inputs"] = heat_fluxes
    normalized_data["Outputs"] = np.concatenate(
        (temperatures, interface_position[:, :, None]), axis=2
    )
    return normalized_data, maxabs


def mean_scale_data(data, mean=[None] * 3, std=[None] * 3):
    """Normalize the data between minimum and maximum around mean.

    Normalizes the heat fluxes, the temperatures and the interface position separately. Similar to minmax_scale_data, but we subtract the mean instead of
    the minimum.
    Args:
        data: The data to normalize.
        mean: The mean value to use for normalization.
        std: The standard deviation value to use for normalization.
    Returns:
        normalized_data
        mean
        std
    """

    def mean_scale_numpy_array(array, mean=None, std=None):
        if mean is None:
            mean = array.mean()
        if std is None:
            std = array.std()
        return (array - mean) / std, mean, std

    heat_fluxes, mean[0], std[0] = mean_scale_numpy_array(
        data["Inputs"], mean[0], std[0]
    )

    temperatures, mean[1], std[1] = mean_scale_numpy_array(
        data["Outputs"][:, :, :5], mean[1], std[1]
    )
    interface_position, mean[2], std[2] = mean_scale_numpy_array(
        data["Outputs"][:, :, 5], mean[2], std[2]
    )

    normalized_data = {}
    normalized_data["Inputs"] = heat_fluxes
    normalized_data["Outputs"] = np.concatenate(
        (temperatures, interface_position[:, :, None]), axis=2
    )
    return normalized_data, mean, std


def reverse_minmax_scale_numpy_array(array, min, max):
    delta = max - min
    return array * delta + min

def reverse_minmax_scale_output(output, min=[None]*2, max=[None]*2):
    """Denormalize the output of the model.
    
    min and max are the values used for normalization of outputs (the second and third returned value from minmax scale data).
    """
    temperatures = reverse_minmax_scale_numpy_array(
        output[:, :, :5], min[0], max[0]
    )
    interface_position = reverse_minmax_scale_numpy_array(
        output[:, :, 5], min[1], max[1]
    )

    denormalized_outputs = np.concatenate(
        (temperatures, interface_position[:, :, None]), axis=2
    )
    return denormalized_outputs

def _print_scaler_statistics(scaler):
    np.random.seed(42)
    feature_1 = np.random.randint(-10, 101, size=(1000, 1)).astype(float)
    feature_2 = np.random.randint(-50, 11, size=(1000, 1)).astype(float)
    next_n_features = np.random.randint(-10,11, size=(1000, 100)).astype(float)

    data = np.concatenate((feature_1, feature_2, next_n_features), axis=1)

    scaler_not_fitted = copy.deepcopy(scaler)
    scaler.fit(data)
    transformed_data = scaler.transform(data)
    inverse_transformed_data = scaler.inverse_transform(transformed_data)
    print("Inverse transform all close:", np.allclose(data, inverse_transformed_data))
    if not np.allclose(data, inverse_transformed_data):
        print("Inverse transform all close (max diff):", np.max(np.abs(data - inverse_transformed_data)))
    print("Orig:        min", np.min(data), "max", np.max(data), "mean", np.mean(data), "std", np.std(data))
    print("Transformed: min", np.min(transformed_data), "max", np.max(transformed_data), "mean", np.mean(transformed_data), "std", np.std(transformed_data))
    print("Orig        feature 1:", "min", np.min(data[:,0]), "max", np.max(data[:,0]), "mean", np.mean(data[:,0]), "std", np.std(data[:,0]))
    print("Transformed feature 1:", "min", np.min(transformed_data[:,0]), "max", np.max(transformed_data[:,0]), "mean", np.mean(transformed_data[:,0]), "std", np.std(transformed_data[:,0]))
    print("Orig        feature 2:", "min", np.min(data[:,1]), "max", np.max(data[:,1]), "mean", np.mean(data[:,1]), "std", np.std(data[:,1]))
    print("Transformed feature 2:", "min", np.min(transformed_data[:,1]), "max", np.max(transformed_data[:,1]), "mean", np.mean(transformed_data[:,1]), "std", np.std(transformed_data[:,1]))
    scaler = scaler_not_fitted
    scaler.fit(feature_1)
    transformed_data = scaler.transform(feature_1)
    inverse_transformed_data = scaler.inverse_transform(transformed_data)
    print("Orig        only feature 1:", "min", np.min(feature_1), "max", np.max(feature_1), "mean", np.mean(feature_1), "std", np.std(feature_1))
    print("Transformed only feature 1:", "min", np.min(transformed_data), "max", np.max(transformed_data), "mean", np.mean(transformed_data), "std", np.std(transformed_data))

    
def _test_global_scalers():
    print("Compare Min Max Scalers")
    print("***********************")
    print("GlobalMinMaxScaler")
    _print_scaler_statistics(GlobalMinMaxScaler())
    print("***********************")
    print("sklearn.preprocessing.MinMaxScaler")
    _print_scaler_statistics(sklearn.preprocessing.MinMaxScaler())

    print("\n\nCompare Max Abs Scalers")
    print("***********************")
    print("GlobalMaxAbsScaler")
    _print_scaler_statistics(GlobalMaxAbsScaler())
    print("***********************")
    print("sklearn.preprocessing.MaxAbsScaler")
    _print_scaler_statistics(sklearn.preprocessing.MaxAbsScaler())

    print("\n\nCompare Standard Scalers")
    print("***********************")
    print("GlobalStandardScaler")
    _print_scaler_statistics(GlobalStandardScaler(with_mean=False))
    print("***********************")
    print("sklearn.preprocessing.StandardScaler")
    _print_scaler_statistics(sklearn.preprocessing.StandardScaler(with_mean=False))

    print("\n\nCompare Robust Scalers")
    print("***********************")
    print("GlobalRobustScaler")
    _print_scaler_statistics(GlobalRobustScaler())
    print("***********************")
    print("sklearn.preprocessing.RobustScaler")
    _print_scaler_statistics(sklearn.preprocessing.RobustScaler())

def _test_partition_scaler(
    scaler_class=GlobalStandardScaler,
    partitions=[[0, 2, 3], [5,6,8]],
    scaler_kwargs={"with_mean": False, "with_std": True},
):
    _print_scaler_statistics(PartitionsScaler(scaler_class, partitions, **scaler_kwargs))

if __name__ == "__main__":
    _test_global_scalers()
    #_test_partition_scaler()
    #_print_scaler_statistics(GlobalRobustScaler())
    pass