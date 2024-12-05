import keras
import keras.ops as knp
import tensorflow as tf

class MaskedTemporalCategoricalCrossEntropy(keras.losses.Loss):
    """When doing classification tasks on time series with certain timestamps masked, this computes cross entropy loss along the time axis (axis 1) with the masked values ignored.
    """
    def __init__(
        self, 
        mask_value=-1, 
        sparse: bool = True, 
        from_logits: bool = False, 
        **kwargs
    ):
        super(MaskedTemporalCategoricalCrossEntropy, self).__init__(**kwargs)
        self.mask_value = mask_value
        self.sparse = sparse
        self.from_logits = from_logits
    
    def compute_class_weight(self, y_true, num_classes, dtype=tf.float32):
        # Flatten y_true to get global class counts
        # Start with turning into sparse encoding if needed
        if not self.sparse:
            y_true = knp.argmax(y_true, axis=-1)
        
        class_counts = knp.bincount(y_true, minlength=num_classes)
        print("class counts:", class_counts)
        class_counts_float = tf.cast(class_counts, dtype)
        total_samples = tf.reduce_sum(class_counts_float)

        # Compute class weights
        # A common formula for balancing is:
        # weight_c = total_samples / (num_classes * class_counts[c])
        # This assigns higher weights to underrepresented classes.
        class_weights = total_samples / (tf.cast(num_classes, dtype) * class_counts_float)
        return class_weights

    def call(self, y_true, y_pred):
        # mask = knp.all(knp.not_equal(y_true, self.mask_value), axis=-1)
        mask = knp.not_equal(y_true, self.mask_value)

        pred_dtype = y_pred.dtype
        num_classes = y_pred.shape[-1]

        class_weights = self.compute_class_weight(y_true[mask], num_classes, dtype=pred_dtype)

        y_true = knp.cast(y_true, pred_dtype)
        if self.sparse:
            y_true = knp.one_hot(y_true, y_pred.shape[-1])

        if self.from_logits:
            y_pred = knp.softmax(y_pred)

        # Compute the element-wise cross-entropy for each (batch, time, class)
        # Add a small value to prevent log(0).
        epsilon = 1e-7
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
        print("true shape:", y_true.shape, "\npred shape:", y_pred.shape, "\nclass_weights shape:", class_weights.shape)
        print("class weights:", class_weights)
        ce = -tf.reduce_sum(y_true * tf.math.log(y_pred), axis=-1)
        # now shape is (batch, time)

        # Mask the values
        mask = tf.cast(mask, ce.dtype)
        print("mask shape:", mask.shape, "\nCE shape:", ce.shape) 
        ce = ce * mask
        
        # Average over the time dimension
        ce_time_mean = tf.reduce_mean(ce, axis=1)  # shape is (batch,)

        # Average over the batch dimension
        return tf.reduce_mean(ce_time_mean)

    
    def _custom_config(self) -> dict:
        return {
            'mask_value': self.mask_value,
            'sparse': self.sparse,
            'from_logits': self.from_logits
        }

    def get_config(self):
        config = super(MaskedTemporalCategoricalCrossEntropy, self).get_config()
        # Add custom config for this class
        config |= self._custom_config()
        return config