import copy
from enum import Enum
from typing import Dict, List, Union

import keras
import numpy as np


class MetricAggregation(Enum):
    OVERALL = "overall"
    PER_FEATURE = "per_feature"
    PER_SAMPLE = "per_sample"

class MetricResults:
    def __init__(self, pred_y, test_y, metric_obj:keras.metrics.Metric):
        self.metric = self.compute_metric(pred_y, test_y, metric_obj)
        self.metric_per_feature = self.compute_metric_per_axis(pred_y, test_y, metric_obj, axis=-1)
        self.metric_per_sample = self.compute_metric_per_axis(pred_y, test_y, metric_obj, axis=0)

    def get_metric(self, aggregation: MetricAggregation=MetricAggregation.OVERALL) -> Union[float, List[float]]:
        if aggregation == MetricAggregation.OVERALL:
            return self.metric
        elif aggregation == MetricAggregation.PER_FEATURE:
            return self.metric_per_feature
        elif aggregation == MetricAggregation.PER_SAMPLE:
            return self.metric_per_sample
        else:
            raise ValueError("metric_aggregation must be a MetricType object")

    def compute_metric(self, pred_y, test_y, metric_obj:keras.metrics.Metric) -> float:
        metric_obj.reset_state()
        return metric_obj(pred_y, test_y).numpy()

    def compute_metric_per_axis(self, pred_y, test_y, metric_obj:keras.metrics.Metric, axis=0) -> List[float]:
        if metric_obj is None:
            raise ValueError("metric_obj must be a keras.metrics.Metric object")
        if axis >= len(pred_y.shape):
            raise ValueError("axis must be less than the number of dimensions of pred_y")
        num_samples = pred_y.shape[axis]
        metric_per_axis = []
        for i in range(num_samples):
            pred_sample = np.take(pred_y, i, axis=axis)
            true_sample = np.take(test_y, i, axis=axis)
            metric_obj.reset_state()
            metric_obj.update_state(pred_sample, true_sample)
            metric_per_axis.append(metric_obj.result().numpy())
        return metric_per_axis

class TrainingResults:
    def __init__(self, model=None, pred_y=None, true_y=None, unscaled_pred_y=None, unscaled_true_y=None, metrics=[keras.metrics.MeanSquaredError(), keras.metrics.RootMeanSquaredError(), keras.metrics.MeanAbsoluteError()]):
        self.model = model
        self.pred_y = pred_y
        self.true_y = true_y
        self.unscaled_pred_y = unscaled_pred_y
        self.unscaled_true_y = unscaled_true_y
        self.metrics = metrics
        self.metrics_values = {}

        if self.pred_y is not None and self.unscaled_pred_y is not None:
            self.compute_metrics(self.pred_y, self.true_y, is_scaled=True)
            self.compute_metrics(self.unscaled_pred_y, self.unscaled_true_y, is_scaled=False)
        else:
            self.metrics_values = None
    
    def compute_metrics(self, pred_y, true_y, is_scaled: bool) -> None:
        for metric in self.metrics:
            if metric.name not in self.metrics_values:
                self.metrics_values[metric.name] = {}
            self.metrics_values[metric.name]["scaled" if is_scaled else "unscaled"] = MetricResults(pred_y, true_y, metric)
    
    def get_metrics(self, get_unscaled: bool):
        """Returns a dictionary of MetricResults objects with the metric name as key and the MetricResults object as value

        Args:
            get_unscaled (bool): If True, returns the unscaled metrics, otherwise returns the scaled metrics. Unscaled metrics are computed by reversing the scaling function used in training.
        """
        if self.metrics_values is None:
            return None
        return {metric_name: metric_results["unscaled" if get_unscaled else "scaled"] for metric_name, metric_results in self.metrics_values.items()}


    def __str__(self):
        return f"TrainingResults: {self.metrics_values}"

    def __repr__(self):
        return self.__str__()

class KFoldResults:
    def __init__(self, training_results: List[TrainingResults]):
        self.training_results = training_results
        self.models = [result.model for result in training_results]

    def get_models(self):
        models = [result.model for result in self.training_results]
        return copy.deepcopy(models)
    
    def get_metrics_names(self):
        metrics = self.training_results[0].metrics
        return [copy.deepcopy(metric.name) for metric in metrics]
    
    def get_metrics_objects(self, copy=True):
        metrics = self.training_results[0].metrics
        return copy.deepcopy(metrics) if copy else metrics

    def get_list_of_results(self) -> List[Dict]:
        return [train_output.metrics_values for train_output in self.training_results]