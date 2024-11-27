import numpy as np
from keras.utils import Sequence
import tensorflow as tf


class PermutationDataGenerator(Sequence):
    def __init__(self, images, labels, sample_weights=None, batch_size=8):
        self.images = images
        self.labels = labels
        self.sample_weights = sample_weights
        self.batch_size = batch_size
        self.indices = np.arange(len(images))
        
    def __len__(self):
        return len(self.images) // self.batch_size

    def __getitem__(self, index):
        # Select batch indices
        batch_indices = self.indices[index * self.batch_size: (index + 1) * self.batch_size]
        batch_images = self.images[batch_indices]
        batch_labels = self.labels[batch_indices]
        batch_weights = self.sample_weights[batch_indices] if self.sample_weights is not None else None
        
        # Permute channels for a batch of images
        permuted_images = np.array([img[..., np.random.permutation(3)] for img in batch_images])

        
        return permuted_images, batch_labels, batch_weights

    def on_epoch_end(self):
        np.random.shuffle(self.indices)


class Random3DRotationGenerator(Sequence):
    def __init__(self, images, labels, sample_weights, batch_size=8):
        """
        Initialize the generator.
        Args:
            images (np.ndarray): Input images of shape (num_samples, height, width, channels).
            labels (np.ndarray): Corresponding labels.
            sample_weights (np.ndarray): Sample weights for each image.
            batch_size (int): Batch size for the generator.
        """
        self.images = images
        self.labels = labels
        self.sample_weights = sample_weights
        self.batch_size = batch_size
        self.indices = np.arange(len(images))
        self.rotation_matrix = self._generate_random_rotation_matrix()  # Initialize rotation matrix

    def __len__(self):
        """Number of batches per epoch."""
        return len(self.images) // self.batch_size

    def __getitem__(self, index):
        """
        Generate a batch of data.
        Args:
            index (int): Batch index.
        Returns:
            tuple: (rotated_images, labels, sample_weights).
        """
        # Get batch indices
        batch_indices = self.indices[index * self.batch_size: (index + 1) * self.batch_size]
        
        # Get batch data
        batch_images = self.images[batch_indices]
        batch_labels = self.labels[batch_indices]
        batch_weights = self.sample_weights[batch_indices]
        
        # Apply the current rotation matrix to the entire batch
        rotated_images = self._apply_rotation(batch_images)
        
        return rotated_images, batch_labels, batch_weights

    def on_epoch_end(self):
        """Shuffle indices and generate a new rotation matrix at the end of each epoch."""
        print("Shuffling indices and generating a new rotation matrix.")
        np.random.shuffle(self.indices)
        self.rotation_matrix = self._generate_random_rotation_matrix()

    def _generate_random_rotation_matrix(self):
        """
        Generate a random 3D rotation matrix using TensorFlow.
        Returns:
            tf.Tensor: A 3x3 rotation matrix.
        """
        # Random angles for rotation around each axis
        angle_x = tf.random.uniform([], minval=-np.pi / 6, maxval=np.pi / 6)  # Rotation around x-axis
        angle_y = tf.random.uniform([], minval=-np.pi / 6, maxval=np.pi / 6)  # Rotation around y-axis
        angle_z = tf.random.uniform([], minval=-np.pi / 6, maxval=np.pi / 6)  # Rotation around z-axis

        # Rotation matrices for each axis
        rotation_x = tf.convert_to_tensor([
            [1, 0, 0],
            [0, tf.cos(angle_x), -tf.sin(angle_x)],
            [0, tf.sin(angle_x), tf.cos(angle_x)],
        ], dtype=tf.float32)

        rotation_y = tf.convert_to_tensor([
            [tf.cos(angle_y), 0, tf.sin(angle_y)],
            [0, 1, 0],
            [-tf.sin(angle_y), 0, tf.cos(angle_y)],
        ], dtype=tf.float32)

        rotation_z = tf.convert_to_tensor([
            [tf.cos(angle_z), -tf.sin(angle_z), 0],
            [tf.sin(angle_z), tf.cos(angle_z), 0],
            [0, 0, 1],
        ], dtype=tf.float32)

        # Combine rotations: R = Rz * Ry * Rx
        rotation_matrix = tf.linalg.matmul(rotation_z, tf.linalg.matmul(rotation_y, rotation_x))
        return rotation_matrix

    def _apply_rotation(self, images):
        """
        Apply the current rotation matrix to a batch of images.
        Args:
            images (np.ndarray): Batch of images with shape (batch_size, height, width, channels).
        Returns:
            tf.Tensor: Rotated batch of images with the same shape.
        """
        # Flatten the spatial dimensions to a single tensor of shape (batch_size, num_pixels, 3)
        batch_size, height, width, channels = images.shape
        assert channels == 3, "Input images must have 3 channels for 3D rotation."

        flat_images = tf.reshape(images, [batch_size, height * width, 3])  # (batch_size, num_pixels, 3)

        # Apply rotation: rotated_pixels = dot(flat_images, rotation_matrix)
        rotated_pixels = tf.linalg.matmul(flat_images, self.rotation_matrix)

        # Reshape back to original dimensions
        rotated_images = tf.reshape(rotated_pixels, [batch_size, height, width, 3])
        return rotated_images


