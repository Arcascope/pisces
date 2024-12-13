import keras
import keras.ops as knp
import tensorflow as tf

class MaskedLoss(keras.losses.Loss):
    def __init__(self, loss, mask_value=-1):
        super().__init__()
        self.loss = loss
        self.mask_value = mask_value

    def call(self, y_true, y_pred, sample_weight=None):
        mask = tf.not_equal(y_true, self.mask_value)
        y_true = tf.boolean_mask(y_true, mask)
        y_pred = tf.boolean_mask(y_pred, mask)
        if sample_weight is not None:
            sample_weight = tf.boolean_mask(sample_weight, mask)
        return self.loss(y_true, y_pred, sample_weight)

    def get_config(self):
        return {"loss": self.loss, "mask_value": self.mask_value}

    def from_config(cls, config):
        return cls(**config)

class MaskedMetric(keras.metrics.Metric):
    def __init__(self, dtype=None, name=None):
        super().__init__(dtype, name)
