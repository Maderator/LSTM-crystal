import os
import pickle
from typing import Optional, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split as sklearn_tt_split


class DataLoader:
    """Data loader for GaAs crystal growth data.

    Each method for data loading returns a dictionaries of training and testing data prepared for a model training or evaluation.

    The data file is a pickle file with the dictionary of the following format:
    {
        "Inputs": numpy array of shape [num_samples, time_steps, num_features],
        "Outputs": numpy array of shape [num_samples, time_steps, num_features],
    }
    """

    def __init__(
        self,
        data_path: str,
        inputs_name: str = "Inputs",
        outputs_name: str = "Outputs",
    ):
        """Initialize the data loader.

        Args:
            data_path: path to the folder with data files
        """
        self.data_path = data_path
        self.inputs_name = inputs_name
        self.outputs_name = outputs_name
        # TODO use inputs_name and outputs_name to generalize this class

    def load_pickle(self, dataset_name):
        file_path = os.path.join(self.data_path, dataset_name + ".pkl")
        with open(file_path, "rb") as f:
            dataset = pickle.load(f)
        return dataset

    def train_test_split(self, dataset, test_size=0.0, seed=42):
        """Given a test_size split the dataset into train and test sets."""
        train_data, test_data = {}, {}
        (
            train_data["Inputs"],
            test_data["Inputs"],
            train_data["Outputs"],
            test_data["Outputs"],
        ) = sklearn_tt_split(
            dataset["Inputs"],
            dataset["Outputs"],
            test_size=test_size,
            random_state=seed,
        )
        return train_data, test_data

    def load_and_split_dataset(self, dataset_name, test_size=0.0, seed=42):
        """Load the dataset and split it into train and test sets.

        Returns:
            train_data: dict of train data
            test_data: dict of test data, if test_size == 0.0, then test_data Inputs and Outputs are None types
        """
        dataset = self.load_pickle(dataset_name)
        if test_size > 0.0:
            (train_data, test_data) = self.train_test_split(
                dataset, test_size=test_size, seed=seed
            )
            return (train_data, test_data)
        else:
            test_data = {"Inputs": None, "Outputs": None}
            return (dataset, test_data)

    def load_data(self, training=True, shuffled=True, test_size=0.1, seed=42):
        if shuffled:
            train, val = self.load_trainingShuffled(test_size=test_size, seed=seed)
        else:
            train, val = self.load_trainingSorted(test_size=test_size, seed=seed)
        if training:
            return train, val
        else: # testing
            if shuffled:
                test, _ = self.load_finalTestingShuffled(seed=seed)
            else:
                test, _ = self.load_finalTestingSorted(seed=seed)
            
            # train data is returned for scaler fitting
            return train, test

    def load_finalTestingShuffled(self, seed=42):
        return self.load_and_split_dataset(
            "finalTestingShuffled", test_size=0.0, seed=seed
        )

    def load_finalTestingSorted(self, seed=42):
        return self.load_and_split_dataset(
            "finalTestingSorted", test_size=0.0, seed=seed
        )

    def load_trainingShuffled(self, test_size=0.1, seed=42):
        return self.load_and_split_dataset(
            "trainingShuffled", test_size=test_size, seed=seed
        )

    def load_trainingSorted(self, test_size=0.1, seed=42):
        return self.load_and_split_dataset(
            "trainingSorted", test_size=test_size, seed=seed
        )