import keras
import keras.ops as knp
import tensorflow as tf

class MaskedTemporalCategoricalCrossEntropy(keras.losses.Loss):
    """When doing classification tasks on time series with certain timestamps masked, this computes cross entropy loss along the time axis (axis 1) with the masked values ignored.
    """
    def __init__(
        self, 
        n_classes: int,
        mask_value=-1, 
        sparse: bool = True, 
        from_logits: bool = False, 
        class_weights=None,
        **kwargs
    ):
        super(MaskedTemporalCategoricalCrossEntropy, self).__init__(**kwargs)
        self.mask_value = mask_value
        self.sparse = sparse
        self.from_logits = from_logits
        self.n_classes = n_classes
        self.class_weights = tf.constant([1.0] * n_classes) if class_weights is None else class_weights
    
    def call(self, y_true, y_pred):

        pred_dtype = y_pred.dtype
        batch_dim = y_true.shape[0]
        y_true = knp.cast(knp.squeeze(y_true), pred_dtype)
        if batch_dim == 1:
            y_true = y_true[tf.newaxis, ...]
        mask = y_true != self.mask_value

        if self.sparse:
            y_true = knp.one_hot(y_true - self.mask_value, 1 + y_pred.shape[-1]) # one-hot encode mask value as 0
            # now cut out the 0th class
            y_true = y_true[..., 1:]

        if self.from_logits:
            y_pred = knp.softmax(y_pred)

        # Compute the element-wise cross-entropy for each (batch, time, class)
        # Add a small value to prevent log(0).
        epsilon = 1e-7
        y_pred = knp.clip(y_pred, epsilon, 1.0 - epsilon)
        
        print("Y_PRED SHAPE:", y_pred.shape)
        print("Y_TRUE SHAPE:", y_true.shape)
        log_comparison = y_true * knp.log(y_pred)
        # multiply by class weights
        # log_comparison = log_comparison * self.class_weights
        print("LOG_COMP SHAPE:", log_comparison.shape)

        ce = -tf.reduce_sum(log_comparison, axis=-1)
        print("CE SHAPE:", ce.shape)
        print(mask.shape)
        ce = tf.boolean_mask(ce, mask)
        # now shape is (batch, time)

        
        # Average over the time dimension
        ce_time_mean = tf.reduce_mean(ce)  # shape is (batch,)

        # Average over the batch dimension
        return tf.reduce_mean(ce_time_mean)

    
    def _custom_config(self) -> dict:
        return {
            'mask_value': self.mask_value,
            'sparse': self.sparse,
            'from_logits': self.from_logits,
            'n_classes': self.n_classes
        }

    def get_config(self):
        config = super(MaskedTemporalCategoricalCrossEntropy, self).get_config()
        # Add custom config for this class
        config |= self._custom_config()
        return config