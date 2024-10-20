# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Datasets for sandwich compression."""
import dataclasses
import functools
from typing import Dict

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

################################################################################
# Routines for loading video datasets. 
################################################################################

@dataclasses.dataclass(frozen=True)
class RankType:
  rank: int = 5
  channels: int = 1
  type: tf.dtypes.DType = tf.float32


_video_data = {
    # Image data: [b, n, h, w, c] where b is batch size, n is the number of
    # frames, [h, w] is the image shape, and c is the number of channels.
    'clip': RankType(channels=3),  # A sequence of video frames.
    'forward_flow': RankType(  # Flow from the current frame to the next frame.
        channels=2
    ),
    'backward_flow': RankType(  # Flow from the current frame to the past frame.
        channels=2
    ),
    'occlusion': (  # Pixels in the current frame to be occluded in the next.
        RankType(channels=1)
    ),
    # Non-image data: [b, m] where b is the batch size and m is modality
    # specific.
    'is_training': RankType(rank=2, type=tf.bool),
    'is_444': RankType(rank=2, type=tf.bool),
    'fullpath': RankType(rank=2, type=tf.string),
    'bit_depth': RankType(rank=2, type=tf.int64),
}


def _load_records_into_dataset_and_parse(
    path: str,
    batch_size: int,
    feature_keys_and_types: Dict[str, tf.dtypes.DType],
    filename_shuffle_size: int = 100,
    batch_shuffle_multiplier: int = 100,
) -> tf.data.Dataset:
  """Constructs a dataset from the records using feature_keys_and_types."""

  def load_dataset_from_file(filename: str) -> tf.data.Dataset:
    return tf.data.TFRecordDataset(filename)

  # Start with dataset of filenames.
  filenames = tf.data.Dataset.list_files(path)
  filenames.shuffle(filename_shuffle_size)
  serialized_dataset = filenames.interleave(
      functools.partial(load_dataset_from_file),
      cycle_length=tf.data.experimental.AUTOTUNE,
      num_parallel_calls=tf.data.experimental.AUTOTUNE,
  )

  serialized_dataset = serialized_dataset.shuffle(
      batch_shuffle_multiplier * batch_size
  )

  # Construct example features.
  features = {}
  for key in feature_keys_and_types:
    features[key] = tf.io.FixedLenFeature([], tf.string)

  # Parse serialized data
  serialized_dataset = serialized_dataset.map(
      functools.partial(tf.io.parse_single_example, features=features),
      num_parallel_calls=tf.data.experimental.AUTOTUNE,
  )

  def parse_tensor(data_features) -> Dict[str, tf.Tensor]:
    """Parses individual tensors from features."""
    # Parse the tensor within data_features[key]. Shape is dynamic.
    return {
        key: tf.io.parse_tensor(data_features[key], feature_keys_and_types[key])
        for key in data_features
    }

  # Parse into tensors.
  return serialized_dataset.map(
      parse_tensor, num_parallel_calls=tf.data.experimental.AUTOTUNE
  )


def load_dataset(
    path: str,
    batch_size: int,
    dataset_keys_and_types: Dict[str, tf.dtypes.DType],
    is_training: bool = False,
) -> tf.data.Dataset:
  """Loads the tensors corresponding to dataset_keys into a tf.data.Dataset.

  Args:
    sstable_path: Path to the sstable.
    batch_size: Batch size.
    dataset_keys_and_types: {key: type} dictionary for the requested sstable
      features. (Type is needed for tf.io.parse_tensor.)
    is_training: True to construct the train set, False to construct eval.

  Returns:
    Constructed dataset.
  """

  dataset = _load_records_into_dataset_and_parse(
      path, batch_size, feature_keys_and_types=dataset_keys_and_types
  )
  dataset = dataset.filter(lambda x: x.get('is_training') == is_training)

  return dataset.batch(batch_size, drop_remainder=True).prefetch(
      buffer_size=tf.data.experimental.AUTOTUNE
  )


def load_video_dataset(
    path: str,
    batch_size: int,
    is_training: bool,
) -> tf.data.Dataset:
  """Loads a dataset of image and non-image keys.

  Args:
    sstable_path: Path to the sstable.
    batch_size: Batch size.
    is_training: True to construct the train set, False to construct eval.

  Returns:
    Constructed dataset.
  """
  source_data = _video_data

  dataset_keys_and_types = {
      key: value.type for key, value in source_data.items()
  }

  dataset = load_dataset(
      path=path,
      batch_size=batch_size,
      dataset_keys_and_types=dataset_keys_and_types,
      is_training=is_training,
  )

  # Tensor shapes are dynamic and known at run-time. Some downstream dataset
  # maps (e.g., tf.image.resize) may need to know the static rank of the tensors
  # to operate. Set the tensor ranks.
  tensor_ranks = {
      key: source_data[key].rank for key in dataset_keys_and_types.keys()
  }

  def set_tensor_ranks(example: Dict[str, tf.Tensor]) -> Dict[str, tf.Tensor]:
    """Sets the rank of all tensors."""
    for key, rank in tensor_ranks.items():
      tf.debugging.assert_equal(
          tf.rank(example[key]),
          rank,
          message=f'Tensor with key: {key} should have rank {rank}',
      )
      example[key].set_shape([None] * rank)
    return example

  return dataset.map(
      set_tensor_ranks, num_parallel_calls=tf.data.experimental.AUTOTUNE
  )
