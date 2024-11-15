from math import floor
from sklearn.calibration import expit
import tensorflow as tf
from scipy.special import softmax
import keras


MAX_PSG_EPOCHS = 1024
PSG_EPOCH_SECONDS = 30
PSG_PER_MINUTE = 60 // PSG_EPOCH_SECONDS
ACTIVITY_EPOCH_SECONDS = 15

ACTIVITY_PER_PSG = PSG_EPOCH_SECONDS // ACTIVITY_EPOCH_SECONDS

LR_KERNEL_MINUTES = 15
LR_KERNEL_SIZE = 1 + ACTIVITY_PER_PSG * PSG_PER_MINUTE * LR_KERNEL_MINUTES
LR_ACTIVITY_INPUTS = ACTIVITY_PER_PSG * MAX_PSG_EPOCHS
LR_TOTAL_PAD = (2 * LR_KERNEL_SIZE // 2) - 1
LR_PRE_PAD = floor(LR_TOTAL_PAD / 2)
LR_POST_PAD = LR_TOTAL_PAD - LR_PRE_PAD

LR_INPUT_LENGTH = LR_PRE_PAD + LR_ACTIVITY_INPUTS + LR_POST_PAD
print(f"LR_KERNEL_SIZE: {LR_KERNEL_SIZE}")

# input shapes
FINETUNING_INPUT_SHAPE = (-1, MAX_PSG_EPOCHS, 4)
LR_INPUT_SHAPE = (-1, LR_INPUT_LENGTH, 1)
LABEL_SHAPE = (-1, MAX_PSG_EPOCHS, 1)

# model names
EXTRA_LAYERS_NAME = "Fine Tuning"
LR_CNN_NAME = "LR CNN"
LR_LOWER = "lr"
EXTRA_LOWER = "finetuning"
NAIVE_NAME = "Naive"
NAIVE_LOWER = "naive"
MODEL_TYPES = [LR_LOWER, EXTRA_LOWER, NAIVE_LOWER]

# Custom loss function that takes weights into account


def weighted_binary_crossentropy(y_true, y_pred, sample_weight):
    bce = keras.losses.binary_crossentropy(
        y_true, y_pred, from_logits=True)[..., None]
    weighted_bce = bce * sample_weight
    return tf.reduce_mean(weighted_bce)

# Function to build the CNN-based mixture model


def build_finetuning_model(input_shape):
    inputs = keras.Input(shape=input_shape)  # Input shape (1024, 4)

    # First 1D Convolutional Layer
    x = keras.layers.Conv1D(filters=16, kernel_size=5,
                            padding='same', activation='linear')(inputs)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.LeakyReLU(negative_slope=0.1)(x)

    # # Second 1D Convolutional Layer
    x = keras.layers.Conv1D(filters=32, kernel_size=7,
                            padding='same', activation='linear')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.LeakyReLU(negative_slope=0.1)(x)

    # x = keras.layers.Dropout(0.1)(x)

    # Output Layer: Binary classification (Wake or Sleep)
    x = keras.layers.Conv1D(filters=2, kernel_size=3, padding='same')(x)
    x = keras.layers.LeakyReLU(negative_slope=0.1)(x)

    # Logits output! use 'linear' activation for numerical stability
    x = keras.layers.Conv1D(filters=1, kernel_size=1, activation='linear')(x)

    # The output shape will be (1024, 1) per example, representing the probability of Sleep at each timestep
    return keras.Model(inputs=inputs, outputs=x)


def cnn_pred_proba(cnn, data):
    return expit(
        cnn.predict(
            data.reshape(1, 1024, 4),
            verbose=0
        )).reshape(-1,)


def naive_pred_proba(data):
    return 1 - softmax(data, axis=-1)[:, 0]

# Custom model class


class WeightedModel(keras.Model):
    def __init__(self, original_model: keras.Model):
        super(WeightedModel, self).__init__()
        self.original_model = original_model

    def call(self, inputs):
        x = inputs
        return self.original_model(x)

    def train_step(self, data):
        x, y_true, sample_weight = data

        with tf.GradientTape() as tape:
            y_pred = self.original_model(x, training=True)
            loss = weighted_binary_crossentropy(y_true, y_pred, sample_weight)

        # Compute gradients and update weights
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Update metrics (includes the metric that tracks the loss)
        for metric in self.metrics:
            metric.update_state(y_true, y_pred)
        return {"loss": loss}

# Original model


def build_lr_cnn(kernel_size: int = LR_KERNEL_SIZE):
    input_layer = keras.layers.Input(
        shape=(LR_INPUT_LENGTH, 1), name="activity_input")
    x = keras.layers.Conv1D(
        filters=1, kernel_size=kernel_size, strides=2)(input_layer)
    x = keras.layers.BatchNormalization()(x)
    model = keras.models.Model(inputs=input_layer, outputs=x)
    return model


def lr_cnn_pred_proba(lr_cnn, data: tf.Tensor):
    return expit(
        lr_cnn.predict(
            tf.reshape(data, (1, LR_INPUT_LENGTH, 1)),
            verbose=0
        )).reshape(-1,)


if __name__ == "__main__":
    # Create original and weighted models
    original_model = build_lr_cnn()
    weighted_model = WeightedModel(original_model)

    # Compile the model with an optimizer
    weighted_model.compile(optimizer='adam', metrics=['auc'])

    # Generate dummy data
    N_SAMPLES = 5
    x_data = tf.random.normal(
        (N_SAMPLES, LR_INPUT_LENGTH, 1), dtype=tf.float32)
    y_data = tf.random.normal((N_SAMPLES, 1024, 1), dtype=tf.float32)
    sample_weights = tf.random.normal((N_SAMPLES, 1024, 1), dtype=tf.float32)

    # Train the model using a dataset
    dataset = tf.data.Dataset.from_tensor_slices(
        (x_data, y_data, sample_weights))
    dataset = dataset.batch(1)

    # Fit the model
    weighted_model.fit(dataset, epochs=2, validation_data=None)
