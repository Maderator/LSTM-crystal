import copy
import inspect
import logging
import os
from abc import ABC, abstractmethod
from typing import Optional, Tuple

import data_processing.data_loader as data_loader
import data_processing.data_normalization as data_normalization
import data_processing.data_preprocessing as data_preprocessing
import numpy as np
import pandas as pd
import sklearn.preprocessing
import utils as utils

_repo_path = utils.find_root_path("LSTM-crystal-growth")
DATA_PATH = os.path.join(_repo_path, "python", "data")

class Dataset(ABC):
    @abstractmethod
    def copy():
        pass

class CrystalDataset:
    def __init__(
        self,
        data_path=DATA_PATH,
        scaler=sklearn.preprocessing.MinMaxScaler(),
        preprocessing_function = data_preprocessing.lag_features,
        shuffle=True,
        final_testing=False,
        test_size=0.1,
        outputs_lag=3,
        inputs_lag=0,
        window_size=3,
        return_sequences=True,
        seed=42,
        scaler_partitions=[list(range(2)), list(range(2, 2+5)), list(range(2+5, 2+5+1))], # 2 inputs, 5 fluxes, 1 solid/liquid interface position
        scaler_kwargs={},
    ):
        self.data_path = data_path
        self.scaler = scaler
        self.shuffle = shuffle
        self.final_testing = final_testing
        self.test_size = test_size
        self.outputs_lag = outputs_lag
        self.inputs_lag = inputs_lag
        self.window_size = window_size
        self.return_sequences = return_sequences
        self.seed = seed
        self.scaler_partitions = scaler_partitions
        self.scaler_kwargs = scaler_kwargs
        self.partition_scaler = self._init_partition_scaler()
        
        self.preprocessing_function = self._init_preprocessing_function(preprocessing_function)

        self.raw_train_data, self.raw_test_data = self._load_data()
        self.scaled_train_data, self.scaled_test_data = self.scale_data()
        self.train_data = self.preprocess_data(self.scaled_train_data["Inputs"], self.scaled_train_data["Outputs"])
        if self.scaled_test_data["Inputs"] is None or self.scaled_test_data["Outputs"] is None:
            self.test_data = self.scaled_test_data
        else:
            self.test_data = self.preprocess_data(self.scaled_test_data["Inputs"], self.scaled_test_data["Outputs"])
        if not self.return_sequences:
            self.train_data["Outputs"] = self.train_data["Outputs"][:, -1, :]
            if self.test_data["Outputs"] is not None:
                self.test_data["Outputs"] = self.test_data["Outputs"][:, -1, :]
        

    def preprocess_data(self, X, y, preprocessing_function=None):
        if preprocessing_function:
            self.preprocessing_function = self._init_preprocessing_function(preprocessing_function)
        (processed_X, processed_y) = self.preprocessing_function(X, y)
        return {"Inputs": processed_X, "Outputs": processed_y}

    def scale_data(self):
        def _scale_3d_in_out(data_in_out, fit=False):
            if data_in_out["Inputs"] is None or data_in_out["Outputs"] is None:
                return data_in_out
            data = np.concatenate([data_in_out["Inputs"], data_in_out["Outputs"]], axis=2)
            data_2d = data.reshape(-1, data.shape[2])
            if fit:
                self.partition_scaler.fit(data_2d)
            scaled_data = self.partition_scaler.transform(data_2d)
            scaled_data = scaled_data.reshape(data.shape)
            scaled_data_in_out = {
                "Inputs": scaled_data[:, :, : data_in_out["Inputs"].shape[2]],
                "Outputs": scaled_data[:, :, data_in_out["Inputs"].shape[2] :],
            }
            return scaled_data_in_out

        if self.scaler is None:
            return self.raw_train_data, self.raw_test_data
        else:
            scaled_train_data = _scale_3d_in_out(self.raw_train_data, fit=True)
            scaled_test_data = _scale_3d_in_out(self.raw_test_data, fit=False)

            return scaled_train_data, scaled_test_data

    def reverse_scale_outputs(self, output_data):
        if self.scaler is None or self.partition_scaler is None:
            return output_data
        else:
            # add raw inputs so that we can use partition_scaler
            raw_inputs_features = self.raw_train_data["Inputs"].shape[2]
            inputs_fill = np.zeros((output_data.shape[0], output_data.shape[1], raw_inputs_features))
            concatenated_data_3d = np.concatenate([inputs_fill, output_data], axis=2)
            # flatten and reverse scale
            concatenated_data = concatenated_data_3d.reshape(-1, concatenated_data_3d.shape[2])
            raw_concat_data = self.partition_scaler.inverse_transform(concatenated_data)
            # unflatten
            raw_concat_data_3d = raw_concat_data.reshape(concatenated_data_3d.shape)
            # get raw outputs
            raw_outputs = raw_concat_data_3d[:, :, raw_inputs_features:]
            return raw_outputs

    def deepcopy(self):
        return copy.deepcopy(self)
    
    def get_model_input_output_shapes(self, mode="train"):
        if mode == "train":
            data = self.train_data
        elif mode == "test":
            data = self.test_data
        else:
            raise ValueError("mode must be either 'train' or 'test'")
        input_shape = data["Inputs"].shape[1:] # (timesteps, features)
        output_shape = data["Outputs"].shape[-1] # (predicted_outputs)
        return input_shape, output_shape

    def reassign_and_scale_dataset(self, train_index, test_index, inplace=False):
        cur_ds = self.deepcopy() if not inplace else self
        cur_ds.raw_test_data["Inputs"] = cur_ds.raw_train_data["Inputs"][test_index]
        cur_ds.raw_test_data["Outputs"] = cur_ds.raw_train_data["Outputs"][test_index]
        cur_ds.raw_train_data["Inputs"] = cur_ds.raw_train_data["Inputs"][train_index]
        cur_ds.raw_train_data["Outputs"] = cur_ds.raw_train_data["Outputs"][train_index]
        cur_ds.scaled_train_data, cur_ds.scaled_test_data = cur_ds.scale_data()
        cur_ds.train_data = cur_ds.preprocess_data(cur_ds.scaled_train_data["Inputs"], cur_ds.scaled_train_data["Outputs"])
        cur_ds.test_data = cur_ds.preprocess_data(cur_ds.scaled_test_data["Inputs"], cur_ds.scaled_test_data["Outputs"])
        return cur_ds

    def _init_partition_scaler(self):

        def get_scaler_class_and_kwargs(scaler, scaler_kwargs):
            if inspect.isclass(scaler):
                return scaler, scaler_kwargs
            else:
                logging.warning("Using kwargs of the already initialized scaler instead of the given scaler_kwargs to the CrystalDataset class")
                return scaler.__class__, scaler.__dict__

        scaler_class, scaler_kwargs = get_scaler_class_and_kwargs(self.scaler, self.scaler_kwargs)

        self.partition_scaler = data_normalization.PartitionsScaler(
            scaler_class=scaler_class,
            partitions=self.scaler_partitions,
            scaler_kwargs=scaler_kwargs,
        )
        return self.partition_scaler

    def _load_data(self):
        dl = data_loader.DataLoader(data_path=self.data_path)
        raw_train_data, raw_test_data = dl.load_data(
            training=not self.final_testing, shuffled=self.shuffle, test_size=self.test_size, seed=self.seed
        )
        return raw_train_data, raw_test_data

    def _init_preprocessing_function(self, preprocessing_function):
        arg_names = inspect.getfullargspec(preprocessing_function).args

        kwargs = {name: getattr(self, name) for name in arg_names if hasattr(self, name)}

        general_preprocessing_function = lambda X, y : preprocessing_function(X, y, **kwargs)
        general_preprocessing_function.__name__ = preprocessing_function.__name__

        #if preprocessing_function is data_preprocessing.lag_features:
        #    general_preprocessing_function = lambda X, y: data_preprocessing.lag_features(X, y, outputs_lag=self.outputs_lag, inputs_lag=self.inputs_lag)
        #elif preprocessing_function is data_preprocessing.windowed_preprocessing:
        #    general_preprocessing_function = lambda X, y: data_preprocessing.windowed_preprocessing(X, y, window_size=self.window_size)
        #else:
        #    general_preprocessing_function = lambda X, y: (X, y)
        return general_preprocessing_function
            
class CrystalDatasetV1(Dataset):
    def __init__(
        self,
        training=True,
        shuffled=True,
        test_size=0.1,
        seed=42,
        data_path="data",
        scaling_type=None,
        preprocessing_type="dropka",
        window_size=3,
        return_sequences=True,
    ):
        self.training = training
        self.shuffled = shuffled
        self.test_size = test_size
        self.seed = seed
        self.data_path = data_path
        self.scaling_type = scaling_type
        self.preprocessing_type = preprocessing_type
        self.window_size = window_size
        self.return_sequences = return_sequences

        # scaling function
        self.is_sklearn_scaling = False
        if self.scaling_type and "sklearn" in self.scaling_type:
            self.is_sklearn_scaling = True  # TODO: implement reverse scaling for both sklearn and non sklearn scalings if needed

        self.all_scaling_functions = {
            None: None,
            "minmax": self._minmax_scale_data,
            "maxabs": self._maxabs_scale_data,
            "sklearn_maxabs": self._sklearn_maxabs_scale_data,
            "sklearn_minmax": self._sklearn_minmax_scale_data,
            "sklearn_robust": self._sklearn_robust_scale_data,
            "sklearn_standard": self._sklearn_standard_scale_data,
        }
        self.scaling_function = self.all_scaling_functions[self.scaling_type]

        # preprocessing function
        self.all_preprocessing_functions = {
            "dropka": data_preprocessing.lag_features,
            "windowed": data_preprocessing.windowed_preprocessing,
            None: lambda X, y, inputs_lag=None, outputs_lag=3, window_size=10: (X, y)
        }
        self.preprocessing_function = self.all_preprocessing_functions[
            self.preprocessing_type
        ]

        # load data
        dl = data_loader.DataLoader(data_path=data_path)
        self.raw_train_data, self.raw_test_data = dl.load_data(
            training, shuffled, test_size, seed
        )
        # normalize data
        if self.scaling_function:
            self.scaling_function()
        else:
            self.scaled_train_data = self.raw_train_data
            self.scaled_test_data = self.raw_test_data

        # preprocess data
        self.preprocess_data()

    ##################################
    ######### PUBLIC METHODS #########
    ##################################
    # PREPROCESSING METHODS
    def preprocess_data(self, preprocessing_type=None):
        """Preprocesses the data according to the preprocessing_type.

        Args:
            preprocessing_type (str, optional):
                Type of preprocessing ("dropka", "windowed"). If specified, the preprocessing_type is overwritten with given value and specified preprocessing function is used. If None, the preprocessing_type specified at the object construction time is used. Defaults to None.
        """

        if preprocessing_type:
            self.preprocessing_type = preprocessing_type
            self.preprocessing_function = self.all_preprocessing_functions[
                self.preprocessing_type
            ]
        self.train_data = {}
        self.train_data["Inputs"], self.train_data["Outputs"] = self.preprocessing_function(
            self.scaled_train_data["Inputs"], self.scaled_train_data["Outputs"]
        )
        if (
            self.scaled_test_data["Inputs"] is None
            or self.scaled_test_data["Outputs"] is None
        ):
            self.test_data = self.scaled_test_data
        else:
            self.test_data = {}
            self.test_data["Inputs"], self.test_data["Outputs"] = self.preprocessing_function(
                self.scaled_test_data["Inputs"], self.scaled_test_data["Outputs"]
            )
        if not self.return_sequences:
            self.train_data["Outputs"] = self.train_data["Outputs"][:, -1, :]
            if self.test_data["Outputs"] is not None:
                self.test_data["Outputs"] = self.test_data["Outputs"][:, -1, :]

    # OTHER METHODS
    def copy(self):
        dataset_copy = CrystalDatasetV1(
            training=self.training,
            shuffled=self.shuffled,
            test_size=self.test_size,
            seed=self.seed,
            data_path=self.data_path,
            scaling_type=self.scaling_type,
            preprocessing_type=self.preprocessing_type,
            window_size=self.window_size,
            return_sequences=self.return_sequences,
        )
        dataset_copy.raw_train_data = self.raw_train_data.copy()
        dataset_copy.raw_test_data = self.raw_test_data.copy()
        dataset_copy.scaled_train_data = self.scaled_train_data.copy()
        dataset_copy.scaled_test_data = self.scaled_test_data.copy()
        dataset_copy.train_data = self.train_data.copy()
        dataset_copy.test_data = self.test_data.copy()
        return dataset_copy

    def reverse_scale_outputs(self, output_data):
        if self.is_sklearn_scaling:
            return self._sklearn_reverse_scale_outputs(output_data)

    def get_model_input_output_shapes(self, mode="train"):
        if mode == "train":
            data = self.train_data
        elif mode == "test":
            data = self.test_data
        else:
            raise ValueError("mode must be either 'train' or 'test'")
        input_shape = data["Inputs"].shape[1:] # (timesteps, features)
        output_shape = data["Outputs"].shape[-1] # (predicted_outputs)
        return input_shape, output_shape

    def reassign_and_scale_dataset(self, train_index, test_index, inplace=False):
        cur_ds = self.copy() if not inplace else self
        cur_ds.raw_test_data["Inputs"] = cur_ds.raw_train_data["Inputs"][test_index]
        cur_ds.raw_test_data["Outputs"] = cur_ds.raw_train_data["Outputs"][test_index]
        cur_ds.raw_train_data["Inputs"] = cur_ds.raw_train_data["Inputs"][train_index]
        cur_ds.raw_train_data["Outputs"] = cur_ds.raw_train_data["Outputs"][train_index]
        cur_ds.scaling_function()
        return cur_ds
   
    ###################################
    ######### Protected METHODS #########
    ###################################

    # SCALING METHODS
    def _minmax_scale_data(self, min=[None] * 3, max=[None] * 3):
        self.scaled_train_data, min, max = data_normalization.minmax_scale_data(
            self.raw_train_data
        )
        self.min = min
        self.max = max

        self.scaled_test_data, min, max = data_normalization.minmax_scale_data(
            self.raw_test_data, min, max
        )
        self.preprocess_data()
        return min, max

    def _reverse_minmax_scale_outputs(self, output_data):
        return data_normalization.reverse_minmax_scale_numpy_array(
            output_data, self.min[2], self.max[2]
        )

    def _maxabs_scale_data(self, maxabs=[None] * 3):
        self.scaled_train_data, maxabs = data_normalization.maxabs_scale_data(
            self.raw_train_data
        )

        self.scaled_test_data, maxabs = data_normalization.maxabs_scale_data(
            self.raw_test_data, maxabs
        )
        self.preprocess_data()
        return maxabs

    def _sklearn_maxabs_scale_data(self):
        scaler = sklearn.preprocessing.MaxAbsScaler()

        (
            self.scaled_train_data,
            self.scaled_test_data,
            scaler,
        ) = self._sklearn_scale_train_test_data(
            scaler, self.raw_train_data, self.raw_test_data
        )
        self.preprocess_data()
        return scaler

    def _sklearn_minmax_scale_data(self):
        scaler = sklearn.preprocessing.MinMaxScaler()

        (
            self.scaled_train_data,
            self.scaled_test_data,
            scaler,
        ) = self._sklearn_scale_train_test_data(
            scaler, self.raw_train_data, self.raw_test_data
        )
        self.preprocess_data()
        return scaler

    def _sklearn_robust_scale_data(self):
        scaler = sklearn.preprocessing.RobustScaler()

        (
            self.scaled_train_data,
            self.scaled_test_data,
            scaler,
        ) = self._sklearn_scale_train_test_data(
            scaler, self.raw_train_data, self.raw_test_data
        )
        self.preprocess_data()
        return scaler

    def _sklearn_standard_scale_data(self):
        scaler = sklearn.preprocessing.StandardScaler()

        (
            self.scaled_train_data,
            self.scaled_test_data,
            scaler,
        ) = self._sklearn_scale_train_test_data(
            scaler, self.raw_train_data, self.raw_test_data
        )
        self.preprocess_data()
        return scaler


    def _reverse_preprocess_outputs(self, output_data):
        if self.preprocessing_type == "dropka":
            return self._reverse_dropka_preprocessing(output_data)
        elif self.preprocessing_type == "windowed":
            return self._reverse_windowed_preprocessing(output_data)
        else:
            return output_data

    def _sklearn_scale_timeseries_data(self, scaler, data, fit=True):
        flat_raw_data, concat_shape = self._flatten_data(data)

        if fit:
            scaler = scaler.fit(flat_raw_data)

        scaled_data = scaler.transform(flat_raw_data)

        unflatten_scaled_data = self._unflatten_data(
            scaled_data,
            concat_shape,
            data["Inputs"].shape[2],
        )

        return unflatten_scaled_data, scaler

    def _sklearn_scale_train_test_data(self, scaler, train_data, test_data):
        scaled_train_data, scaler = self._sklearn_scale_timeseries_data(
            scaler, train_data, fit=True
        )

        if (
            test_data["Inputs"] is None or test_data["Outputs"] is None
        ):  # We do not scale None data
            return scaled_train_data, test_data, scaler

        scaled_test_data, scaler = self._sklearn_scale_timeseries_data(
            scaler, test_data, fit=False
        )

        return scaled_train_data, scaled_test_data, scaler

    def _flatten_data(self, data, return_concat_shape=True):
        concated_data = np.concatenate([data["Inputs"], data["Outputs"]], axis=2)
        if return_concat_shape:
            return (
                concated_data.reshape(
                    (
                        concated_data.shape[0] * concated_data.shape[1],
                        concated_data.shape[2],
                    )
                ),
                concated_data.shape,
            )
        else:
            return concated_data.reshape(
                (
                    concated_data.shape[0] * concated_data.shape[1],
                    concated_data.shape[2],
                )
            )

    def _unflatten_data(self, data, shape, input_features_count):
        unflattedned_data = data.reshape((shape[0], shape[1], shape[2]))
        return {
            "Inputs": unflattedned_data[:, :, :input_features_count],
            "Outputs": unflattedned_data[:, :, input_features_count:],
        }

def generate_instances_with_distinct_param(Class_name, default_params, changed_param_key, param_values_list):
    default_params = copy.deepcopy(default_params)
    init_params = {k: default_params[k] for k in Class_name.__init__.__code__.co_varnames if k in default_params}
    instances = []
    for param_value in param_values_list:
        init_params[changed_param_key] = param_value

        cur_instance = Class_name(**init_params)
        instances.append(cur_instance)
    return instances

def _test_crystal_datasets_similarity():
    old_crystal = CrystalDatasetV1(scaling_type="minmax", data_path="python/data/")

    minmax_scaler = data_normalization.GlobalMinMaxScaler 
    new_crystal = CrystalDataset(scaler=minmax_scaler, data_path=old_crystal.data_path)

    scaled_in_close = np.allclose(new_crystal.scaled_train_data["Inputs"], old_crystal.scaled_train_data["Inputs"])
    scaled_out_close = np.allclose(new_crystal.scaled_train_data["Outputs"], old_crystal.scaled_train_data["Outputs"])
    raw_in_close = np.allclose(new_crystal.raw_train_data["Inputs"], old_crystal.raw_train_data["Inputs"])
    raw_out_close = np.allclose(new_crystal.raw_train_data["Outputs"], old_crystal.raw_train_data["Outputs"])
    train_in_close = np.allclose(new_crystal.train_data["Inputs"], old_crystal.train_data["Inputs"])
    train_out_close = np.allclose(new_crystal.train_data["Outputs"], old_crystal.train_data["Outputs"])
    test_in_close = np.allclose(new_crystal.test_data["Inputs"], old_crystal.test_data["Inputs"])
    test_out_close = np.allclose(new_crystal.test_data["Outputs"], old_crystal.test_data["Outputs"])
    all_results = [scaled_in_close, scaled_out_close, raw_in_close, raw_out_close, train_in_close, train_out_close, test_in_close, test_out_close]
    print("Close:", all_results)
    print("All close:", np.all(all_results))

def _test_minmax_similarity():
    import os
    cur_file = os.path.abspath(__file__)
    data_path = os.path.join(os.path.dirname(cur_file), "..", "data")
    cur_dataset = CrystalDatasetV1(
        training=True, 
        shuffled=True, 
        test_size=0.3, 
        seed=42, 
        data_path=data_path, 
        preprocessing_type="dropka", 
        window_size=3, 
        scaling_type="minmax",)

    in_features_count = cur_dataset.scaled_train_data["Inputs"].shape[2]
    out_features_count = cur_dataset.scaled_train_data["Outputs"].shape[2]
    in_feat_idxs = list(range(in_features_count))
    out_feat_idxs = list(range(in_features_count, in_features_count + out_features_count))

    print("Inputs shape:", cur_dataset.raw_train_data["Inputs"].shape)
    print("Outputs shape:", cur_dataset.raw_train_data["Outputs"].shape)
    input_data = np.concatenate([cur_dataset.raw_train_data["Inputs"], cur_dataset.raw_train_data["Outputs"]], axis=2)
    input_data_2d = input_data.reshape(-1, input_data.shape[2])

    pscaler = data_normalization.PartitionsScaler(data_normalization.GlobalMinMaxScaler, partitions=[in_feat_idxs, out_feat_idxs[:-1], out_feat_idxs[-1:]])
    pscaler.fit(input_data_2d)
    scaled_train_data = pscaler.transform(input_data_2d)
    print("All scaled inputs close:", np.allclose(scaled_train_data.reshape(input_data.shape)[:,:,:2], cur_dataset.scaled_train_data["Inputs"]))

    print("Example of scaled inputs:")
    print(cur_dataset.scaled_train_data["Inputs"][:1, :5, :2], "\n")
    print(scaled_train_data.reshape(input_data.shape)[:1, :5, :2])

if __name__ == "__main__":
    #_test_minmax_similarity()
    _test_crystal_datasets_similarity()