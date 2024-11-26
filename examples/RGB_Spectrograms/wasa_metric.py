import keras.ops as ops
from keras.metrics import Metric
import tensorflow as tf
import jax.lax
import jax.numpy as jnp


class WASAMetric(Metric):
    def __init__(self, sleep_accuracy=0.95, **kwargs):
        name = f"WASA{int(100 * sleep_accuracy)}"
        super().__init__(name=name, **kwargs)
        self.sleep_accuracy = sleep_accuracy
        self.specificity_metric = keras.metrics.SpecificityAtSensitivity(sleep_accuracy)
        # self.thresholds = ops.linspace(0.0, 1.0, 101)  # Test thresholds from 0.0 to 1.0
        # self.optimal_threshold = self.add_weight(name="optimal_threshold", initializer="zeros")
        # self.true_sleep = self.add_weight(name="true_sleep", initializer="zeros")
        # self.false_sleep = self.add_weight(name="false_sleep", initializer="zeros")
        # self.false_wake = self.add_weight(name="false_wake", initializer="zeros")
        # self.true_wake = self.add_weight(name="true_wake", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Convert 4-class probabilities to binary probabilities
        if sample_weight is None:
            sample_weight = 1.0
        binary_probs = ops.sum(y_pred[..., 1:], axis=-1)  # Sum probabilities for classes 1, 2, 3 (sleep)
        binary_labels = ops.where(y_true > 0, 1.0, 0.0)  # 0 for wake, 1 for sleep
        self.specificity_metric.update_state(binary_labels, binary_probs, sample_weight)
    
    def result(self):
        return self.specificity_metric.result()