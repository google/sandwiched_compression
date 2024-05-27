"""Various distortion functions for train/eval."""
from typing import Dict

import tensorflow as tf


def distortion_l2norm(
    ground_truth: Dict[str, tf.Tensor],
    outputs: Dict[str, tf.Tensor],
    image_key: str = 'image',
    scaler: float = 1.0,
) -> tf.Tensor:
  error = ground_truth[image_key] - outputs['prediction']
  error = tf.keras.backend.reshape(error, (error.shape[0], -1))
  return tf.keras.backend.sum(tf.keras.backend.square(error), axis=1) * scaler
