"""Datasets for sandwich compression."""
import tensorflow as tf
# https://www.tensorflow.org/datasets
import tensorflow_datasets as tfds


################################################################################
# Routines for loading basic datasets. See  https://www.tensorflow.org/datasets 
# for options.
################################################################################

def load_tfds_image_dataset(batch_size: int,
                            training_mode: bool,
                            dataset_name: str = 'clic',
                            initial_crop_size: int = 512,
                            target_size: int = 128) -> tf.data.Dataset:
  """Load the Clic/Pets train/test dataset."""
  test_set = tfds.load(
      dataset_name,
      split='train' if training_mode else 'test',
      shuffle_files=True,
      download=True,
      try_gcs=True)

  def filter_fn(example):
    image = example['image']
    tf_shape = tf.shape(image)
    return tf_shape[2] == 3 and tf_shape[0] >= initial_crop_size and tf_shape[
        1] >= initial_crop_size

  def crop_resize(example):
    image = example['image']
    image = tf.image.random_crop(
        image, [initial_crop_size, initial_crop_size, image.shape[-1]])
    image = tf.image.resize(
        image, (target_size, target_size),
        method=tf.image.ResizeMethod.LANCZOS3,
        antialias=True)
    return {'image': image}

  test_set = test_set.filter(filter_fn)
  test_set = test_set.map(
      crop_resize, num_parallel_calls=tf.data.experimental.AUTOTUNE)

  test_set = test_set.shuffle(batch_size * 100)

  return test_set.batch(
      batch_size, drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE)
