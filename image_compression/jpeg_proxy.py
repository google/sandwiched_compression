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
r"""Jpeg proxy that emulates intra compression with jpeg."""
import itertools
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import tensorflow as tf


def pad_spatially_to_multiple_of_bsize(
    inputs: tf.Tensor, bsize: int, mode: str = 'SYMMETRIC'
) -> tf.Tensor:
  """Pads a tensor spatially to a multiple of bsize.

  Args:
    inputs: Tensor of size [b, n, m, c] where b is batch size, [n, m] is the
      image shape, and c is the number of channels.
    bsize: Desired pad multiple.
    mode: Padding mode.

  Returns:
    outputs: Tensor of size [b, n', m', c] where n' (n' >= n) and m' (m' >= m)
    are multiples of bsize.
  """
  height, width = inputs.shape[1:3]
  pad_height = ((height - 1) // bsize + 1) * bsize - height
  pad_width = ((width - 1) // bsize + 1) * bsize - width
  if pad_height or pad_width:
    return tf.pad(
        inputs, [[0, 0], [0, pad_height], [0, pad_width], [0, 0]], mode
    )
  return inputs


class JpegProxy:
  """Differentiable JPEG-like layer. Simplified for sandwich use."""

  def __init__(
      self,
      downsample_chroma: bool,
      luma_quantization_table: tf.Tensor,
      chroma_quantization_table: tf.Tensor,
      convert_to_yuv: bool,
      clip_to_image_max: bool,
      dct_size: int = 8,
      upsample_method: tf.image.ResizeMethod = tf.image.ResizeMethod.LANCZOS3,
  ):  # pytype: disable=annotation-type-mismatch
    """Constructor.

    Module that accomplishes the JPEG-like pipeline:
      (rgb-to-yuv) -> (chroma-downsampling) -> forward-DCT -> quantization ->
      dequantization -> inverse-DCT -> (chroma-upsampling) -> (yuv-to-rgb)

    Args:
      downsample_chroma: Whether to downsample chroma channels. Downsampling is
        bilinear, upsampling is as set by upsample_method.
      luma_quantization_table: [dct_size, dct_size] (or [dct_size**2]) tensor
        for Y channel (luma) quantization.
      chroma_quantization_table: [dct_size, dct_size] (or [dct_size**2]) tensor
        for U and V channels (chroma) quantization.
      convert_to_yuv: When True rgb-to-yuv and yuv-to-rgb color conversions are
        enabled. When false the conversions are skipped.
      clip_to_image_max: True if final output should be clipped to [0,
        image_max].
      dct_size: Size of the core 1D DCT transform.
      upsample_method: Method to use in upsampling chroma after compression when
        downsample_chroma is True. Downsampling is always bilinear.
    """
    self.downsample_chroma = downsample_chroma
    self.upsample_method = upsample_method
    self.luma_quantization_table = tf.reshape(
        tf.convert_to_tensor(luma_quantization_table), [-1]
    )
    self.chroma_quantization_table = tf.reshape(
        tf.convert_to_tensor(chroma_quantization_table), [-1]
    )
    self.convert_to_yuv = convert_to_yuv
    self.clip_to_image_max = clip_to_image_max

    # Color conversion matrices (https://en.wikipedia.org/wiki/YCbCr).
    self.rgb_from_yuv_matrix = [
        [1.0, 1.0, 1.0],
        [0, -0.344136, 1.772],
        [1.402, -0.714136, 0],
    ]
    self.yuv_from_rgb_matrix = [
        [0.299, -0.168736, 0.5],
        [0.587, -0.331264, -0.418688],
        [0.114, 0.5, -0.081312],
    ]
    assert (
        np.sum(
            np.abs(
                np.matmul(self.rgb_from_yuv_matrix, self.yuv_from_rgb_matrix)
                - np.eye(3)
            )
        )
        < 1e-3
    )
    self.rgb_from_yuv_matrix = tf.constant(
        self.rgb_from_yuv_matrix, dtype=tf.float32
    )
    self.yuv_from_rgb_matrix = tf.constant(
        self.yuv_from_rgb_matrix, dtype=tf.float32
    )
    self.dct_size = dct_size
    self.dct_2d_mat = self._construct_dct_2d(self.dct_size)

  def _yuv_to_rgb(self, yuv: tf.Tensor) -> tf.Tensor:
    return tf.linalg.matmul(
        yuv - tf.constant([0, 128, 128], dtype=tf.float32),
        self.rgb_from_yuv_matrix,
    )

  def _rgb_to_yuv(self, rgb: tf.Tensor) -> tf.Tensor:
    return tf.linalg.matmul(
        rgb,
        self.yuv_from_rgb_matrix,
    ) + tf.constant([0, 128, 128], dtype=tf.float32)

  def _construct_dct_2d(self, dct_size: int) -> tf.Tensor:
    """Returns a matrix containing the 2D DCT basis in its columns."""
    # 1D basis, unit norm, in columns.
    # https://en.wikipedia.org/wiki/Discrete_cosine_transform (DCTII)
    dct_1d_mat = np.zeros((dct_size, dct_size), dtype=np.float32)

    for i, j in itertools.product(range(dct_size), repeat=2):
      dct_1d_mat[i, j] = np.cos((2 * i + 1) * j * np.pi / (2 * dct_size))

    # Scale for unit norm.
    dct_1d_mat *= np.sqrt(2 / dct_size)
    dct_1d_mat[:, 0] *= 1 / np.sqrt(2)

    block_size = dct_size**2

    # 2D basis, unit norm, in columns.
    dct_2d_mat = np.zeros((block_size, block_size), dtype=np.float32)

    for i in range(block_size):
      dct_2d_mat[:, i] = np.reshape(
          np.outer(dct_1d_mat[:, i // dct_size], dct_1d_mat[:, i % dct_size]),
          [-1],
      )
    assert (
        np.sum(np.abs(np.matmul(dct_2d_mat, dct_2d_mat.T) - np.eye(block_size)))
        <= 1e-3
    )
    return tf.convert_to_tensor(dct_2d_mat, dtype=tf.float32)

  def _forward_dct_2d(self, image_channel: tf.Tensor) -> tf.Tensor:
    """Returns the 2D DCT coefficients of the input image channel."""
    image_patches = tf.image.extract_patches(
        images=image_channel,
        sizes=[1, self.dct_size, self.dct_size, 1],
        strides=[1, self.dct_size, self.dct_size, 1],
        rates=[1, 1, 1, 1],
        padding='VALID',
    )
    offset = 128
    # return coefficients, with shape:
    #   [batch_size, height / dct_size, width / dct_size, dct_size * dct_size]
    return tf.linalg.matmul(image_patches - offset, self.dct_2d_mat)

  def _inverse_dct_2d(self, dct_coeffs: tf.Tensor) -> tf.Tensor:
    """Returns the image that is the inverse transform of the coeffs."""
    offset = 128
    # channel is pixel domain, with shape:
    #   [batch_size, height / dct_size, width / dct_size, dct_size * dct_size]
    channel = (
        tf.linalg.matmul(dct_coeffs, self.dct_2d_mat, transpose_b=True) + offset
    )
    # after reshape:
    #   [batch_size, height / dct_size, width / dct_size, dct_size, dct_size]
    channel = tf.reshape(
        channel, [*channel.shape[0:-1], self.dct_size, self.dct_size]
    )
    # return channel with shape: [batch_size, height, width, 1]
    return tf.reshape(
        tf.transpose(channel, [0, 1, 3, 2, 4]),
        [
            channel.shape[0],
            channel.shape[1] * self.dct_size,
            channel.shape[2] * self.dct_size,
            -1,
        ],
    )

  def __call__(
      self,
      image: tf.Tensor,
      rounding_fn: Callable[[tf.Tensor], tf.Tensor] = tf.round,
      image_max: Optional[tf.Tensor] = None,
  ) -> Tuple[tf.Tensor, Dict[str, tf.Tensor]]:
    """Compresses and decompresses input an image similar to JPEG.

    Args:
      image: Input image of size [batch_size, height, width, 3].
      rounding_fn: Rounding function used in the quantization. One can establish
        quantization via two ways: (i) Establish quantization tables then use a
        rounding function that rounds to the closest integer. (ii) Set
        quantization tables to 1 and implement quantization in the rounding_fn.
        This is the preferred mode when the qstep(s) used during quantization
        are optimized and maintained elsewhere. See EncodeDecodeIntra in
        encode_decode_intra_lib.py.
      image_max: Maximum possible value of the image.

    Returns:
      Compressed image of the same size and type as input.
      Dictionary containing the quantized DCT coefficients of Y, U and V
        channels. Each channel has shape:
          [batch_size, h / dct_size, w / dct_size, dct_size * dct_size],
        where (h, w) are the suitably padded height and width of that channel.
        For images with dimensions that are not a multiple of dct_size,
        symmetric extension is used.
    """
    image = tf.convert_to_tensor(image)
    image.shape.assert_has_rank(4)
    assert image.shape[-1] == 3
    height, width = image.shape[1:3]
    if image_max is None:
      image_max = tf.constant(255.0, dtype=tf.float32)

    # Pad to a multiple of dct_size (or 2 * dct_size if downsampling chroma.)
    pad_multiple = (
        2 * self.dct_size if self.downsample_chroma else self.dct_size
    )
    image = pad_spatially_to_multiple_of_bsize(
        image, pad_multiple, mode='SYMMETRIC'
    )
    padded_height, padded_width = image.shape[1:3]

    # Encode-Decode
    if self.convert_to_yuv:
      image = self._rgb_to_yuv(image)

    downsample = [False, self.downsample_chroma, self.downsample_chroma]
    dct_keys = ['y', 'u', 'v']
    decoded_image = []

    quantized_dct_coeffs = {}
    for ch in range(3):
      if downsample[ch]:
        channel = tf.image.resize(
            image[..., ch : ch + 1],
            [padded_height // 2, padded_width // 2],
            tf.image.ResizeMethod.BILINEAR,  # Should result in the mid pixel.
            antialias=True,
        )
      else:
        channel = image[..., ch : ch + 1]

      # Forward DCT.
      coeffs = self._forward_dct_2d(channel)

      # Quantized DCT coefficients.
      quantization_table = (
          self.luma_quantization_table
          if ch == 0
          else self.chroma_quantization_table
      )
      quantized_dct_coeffs[dct_keys[ch]] = rounding_fn(
          coeffs / quantization_table
      )

      # Dequantized DCT coefficients.
      dequantized = quantized_dct_coeffs[dct_keys[ch]] * quantization_table

      # Inverse DCT.
      channel = self._inverse_dct_2d(dequantized)

      if downsample[ch]:
        channel = tf.image.resize(
            channel,
            [padded_height, padded_width],
            method=self.upsample_method,
            antialias=True,
        )
      decoded_image = (
          channel if ch == 0 else tf.concat([decoded_image, channel], axis=-1)
      )

    # Convert YUV to RGB color space.
    if self.convert_to_yuv:
      decoded_image = self._yuv_to_rgb(decoded_image)

    # Back to original size.
    decoded_image = decoded_image[:, 0:height, 0:width, :]
    if self.clip_to_image_max:
      decoded_image = tf.clip_by_value(decoded_image, 0.0, image_max)
    return decoded_image, quantized_dct_coeffs
