import numpy as np
from keras.utils import Sequence
from scipy.ndimage import rotate


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
            images (np.ndarray): Array of input images with shape (num_samples, height, width, channels).
            labels (np.ndarray): Array of corresponding labels.
            sample_weights (np.ndarray): Array of sample weights for each image.
            batch_size (int): Size of the batches to generate.
        """
        self.images = images
        self.labels = labels
        self.sample_weights = sample_weights
        self.batch_size = batch_size
        self.indices = np.arange(len(images))

    def __len__(self):
        """Number of batches per epoch."""
        return len(self.images) // self.batch_size

    def __getitem__(self, index):
        """
        Generate one batch of data.
        Args:
            index (int): Batch index.
        Returns:
            tuple: (rotated_images, labels, sample_weights) for the batch.
        """
        # Get the indices for this batch
        batch_indices = self.indices[index * self.batch_size: (index + 1) * self.batch_size]
        
        # Fetch batch data
        batch_images = self.images[batch_indices]
        batch_labels = self.labels[batch_indices]
        batch_weights = self.sample_weights[batch_indices]
        
        # Apply the same random 3D rotation to the whole batch
        rotated_images = self.batch_random_3d_rotate(batch_images)
        
        return rotated_images, batch_labels, batch_weights

    def on_epoch_end(self):
        """Shuffle the indices after each epoch."""
        np.random.shuffle(self.indices)

    def batch_random_3d_rotate(self, images):
        """
        Applies the same random 3D rotation to the entire batch of images.
        Args:
            images (np.ndarray): Batch of images with shape (batch_size, height, width, channels).
        Returns:
            np.ndarray: Rotated batch of images with the same shape.
        """
        # Generate random rotation angles for each axis
        angle_x = np.random.uniform(-30, 30)  # Rotation around x-axis
        angle_y = np.random.uniform(-30, 30)  # Rotation around y-axis
        angle_z = np.random.uniform(-30, 30)  # Rotation around z-axis

        # Apply the same rotation to all images in the batch
        rotated_batch = np.array([
            self.apply_rotation(image, angle_x, angle_y, angle_z) for image in images
        ])
        return rotated_batch

    def apply_rotation(self, image, angle_x, angle_y, angle_z):
        """
        Applies 3D rotation to a single image.
        Args:
            image (np.ndarray): Input image of shape (height, width, channels).
            angle_x (float): Rotation angle around x-axis.
            angle_y (float): Rotation angle around y-axis.
            angle_z (float): Rotation angle around z-axis.
        Returns:
            np.ndarray: Rotated image with the same shape.
        """
        # Rotate around x-axis
        rotated = rotate(image, angle_x, axes=(1, 2), reshape=False, mode='reflect')
        # Rotate around y-axis
        rotated = rotate(rotated, angle_y, axes=(0, 2), reshape=False, mode='reflect')
        # Rotate around z-axis
        rotated = rotate(rotated, angle_z, axes=(0, 1), reshape=False, mode='reflect')
        return rotated

