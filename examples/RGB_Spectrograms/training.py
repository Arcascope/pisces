
import tensorflow as tf
import keras


def sparse_kl_divergence(y_true, y_pred):
    y_pred_clipped = tf.clip_by_value(y_pred, 1e-7, 1.0)  # Prevent log(0)
    y_true_one_hot = tf.one_hot(tf.cast(y_true, tf.int32), depth=y_pred.shape[-1])  # Convert to one-hot
    return tf.reduce_sum(y_true_one_hot * tf.math.log(y_true_one_hot / y_pred_clipped), axis=-1)

def sparse_categorical_focal(y_true, y_pred):
    gamma = 2.0
    alpha = 0.25
    y_pred_clipped = tf.clip_by_value(y_pred, 1e-7, 1.0)  # Prevent log(0)
    y_true_one_hot = tf.one_hot(tf.cast(y_true, tf.int32), depth=y_pred.shape[-1])  # Convert to one-hot
    return keras.losses.CategoricalFocalCrossentropy(gamma=gamma, alpha=alpha)(y_true_one_hot, y_pred_clipped)