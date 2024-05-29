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
