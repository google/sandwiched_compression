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
"""Serialize/deserialize models."""

import logging
import tensorflow as tf


def load_checkpoint(
    checkpoint_folder: str, model: tf.keras.Model
) -> tf.keras.Model:
  """Loads a pretrained model.

  Args:
      checkpoint_folder: Directory to the checkpoint.
      model: Initialized model to be loaded with the pretrained model.

  Returns:
      outputs: The loaded model and the qstep.
  """
  checkpoint = tf.train.Checkpoint(
      model=model,
      step=tf.Variable(-1, dtype=tf.int64),
      training_finished=tf.Variable(False, dtype=tf.bool),
  )

  checkpoint_path = tf.train.latest_checkpoint(checkpoint_folder)
  logging.info('-----------------------------------')
  logging.info('Loading model from: %s', checkpoint_path)

  try:
    status = checkpoint.restore(checkpoint_path)
    status.assert_existing_objects_matched()
    status.expect_partial()
  except (tf.errors.NotFoundError, AssertionError) as err:
    logging.info(
        'Failed to restore checkpoint from %s. Error:\n%s', checkpoint_path, err
    )
  return model

