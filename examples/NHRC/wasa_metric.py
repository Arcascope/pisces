import keras.ops as ops
from keras.metrics import Metric
import tensorflow as tf
import jax.lax
import jax.numpy as jnp


class SleepAccuracyMetric(Metric):
    def __init__(self, sleep_accuracy=0.95, **kwargs):
        name = f"WASA{int(100 * sleep_accuracy)}"
        super().__init__(name=name, **kwargs)
        self.sleep_accuracy = sleep_accuracy
        self.thresholds = ops.linspace(0.0, 1.0, 101)  # Test thresholds from 0.0 to 1.0
        self.optimal_threshold = self.add_weight(name="optimal_threshold", initializer="zeros")
        self.true_sleep = self.add_weight(name="true_sleep", initializer="zeros")
        self.false_sleep = self.add_weight(name="false_sleep", initializer="zeros")
        self.false_wake = self.add_weight(name="false_wake", initializer="zeros")
        self.true_wake = self.add_weight(name="true_wake", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Convert 4-class probabilities to binary probabilities
        if sample_weight is None:
            sample_weight = 1.0
        binary_probs = ops.sum(y_pred[..., 1:], axis=-1)  # Sum probabilities for classes 1, 2, 3 (sleep)
        binary_labels = ops.where(y_true > 0, 1.0, 0.0)  # 0 for wake, 1 for sleep
        # Initialize variables for tracking the best threshold
        best_threshold = None
        best_diff = float("inf")
        best_sensitivity = 0.0

        for threshold in self.thresholds:
            # Threshold the predictions
            binary_preds = ops.cast(binary_probs >= threshold, "float32")

            # Calculate true positives and false negatives for sleep
            tp = ops.sum(binary_preds * binary_labels * sample_weight)  # True Sleep
            fn = ops.sum((1 - binary_preds) * binary_labels * sample_weight)  # False Wake

            # Calculate sensitivity
            sensitivity = tp / (tp + fn + 1e-10)
            
            # Calculate the difference between the sensitivity and the target sleep accuracy
            diff = ops.abs(sensitivity - self.sleep_accuracy)

            # Check if the current threshold is better than the previous best
            if diff < best_diff:
                best_diff = diff
                best_threshold = threshold
                best_sensitivity = sensitivity

        # Check if a valid threshold was found
        if best_threshold is None:
            raise ValueError("No valid threshold found during update_state computation.")
        else:
            print(f"Best threshold: {best_threshold}, declaring victory with sensitivity: {best_sensitivity}")

        # Store the best threshold
        self.optimal_threshold.assign(best_threshold)

        # Recalculate with the best threshold
        binary_preds = ops.cast(binary_probs >= best_threshold, "float32")

        # Calculate true/false sleep and wake
        tp = ops.sum(binary_preds * binary_labels * sample_weight)
        tn = ops.sum((1 - binary_preds) * (1 - binary_labels) * sample_weight)
        fp = ops.sum(binary_preds * (1 - binary_labels) * sample_weight)
        fn = ops.sum((1 - binary_preds) * binary_labels * sample_weight)

        self.true_sleep.assign_add(tp)
        self.true_wake.assign_add(tn)
        self.false_sleep.assign_add(fp)
        self.false_wake.assign_add(fn)






    def result(self):
        # Compute final sensitivity (sleep accuracy) and optimal threshold
        sensitivity = self.true_sleep / (self.true_sleep + self.false_wake + 1e-10)
        return {
            "sleep_accuracy": sensitivity,
            "optimal_threshold": self.optimal_threshold,
        }

    def reset_states(self):
        for var in self.variables:
            var.assign(0)
class BinaryTruePositives(Metric):

    def __init__(self, name='binary_true_positives', **kwargs):
        super().__init__(name=name, **kwargs)
        self.true_positives = self.add_variable(
            shape=(),
            initializer='zeros',
            name='true_positives'
        )

    def update_state(self, y_true, y_pred, sample_weight=None):
        binary_probs = ops.sum(y_pred[..., 1:], axis=-1)  # Sum probabilities for classes 1, 2, 3 (sleep)
        y_pred = ops.cast(binary_probs, "bool")
        y_true = ops.cast(y_true, "bool")

        values = ops.logical_and(
            ops.equal(y_true, True), ops.equal(y_pred, True))
        values = ops.cast(values, self.dtype)
        if sample_weight is not None:
            sample_weight = ops.cast(sample_weight, self.dtype)
            sample_weight = ops.broadcast_to(
                sample_weight, ops.shape(values)
            )
            values = ops.multiply(values, sample_weight)
        self.true_positives.assign(self.true_positives + ops.sum(values))

    def result(self):
        return self.true_positives