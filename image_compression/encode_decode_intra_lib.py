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
"""Encode-Decode library of functions that emulate intra compression scenarios."""
import io
from typing import Callable, Dict, List, Optional, Tuple

from image_compression import jpeg_proxy
import logging
import numpy as np
from PIL import Image
import tensorflow as tf

def _encode_decode_with_jpeg(
    input_images: np.ndarray,
    qstep: np.float32,
    one_channel_at_a_time: bool = False,
    use_420: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
  """Compress-decompress with actual jpeg with fixed qstep.

  Args:
    input_images: Array of shape [b, n, m, c] where b is batch size, n x m is
      the image size, and c is the number of channels.
    qstep: float that determines the step-size of the scalar quantizer.
    one_channel_at_a_time: True if each channel should be encoded independently
      as a grayscale image.
    use_420: True when desired subsmapling is 4:2:0. False when 4:4:4.

  Returns:
    decoded: Array of same size as input_images containing the
      quantized-dequantized version of the input_images.
    rate: Array of size b that contains the total number of bits needed to
      encode the input_images into decoded.
  """

  assert input_images.ndim == 4
  decoded = np.zeros_like(input_images)
  rate = np.zeros(input_images.shape[0])
  # Jpeg needs byte qsteps
  jpeg_qstep = np.clip(np.rint(qstep).astype(int), 0, 255)
  qtable = [jpeg_qstep] * 64

  def run_jpeg(input_image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    img = Image.fromarray(
        np.rint(np.clip(input_image, 0, 255)).astype(np.uint8))
    buf = io.BytesIO()
    img.save(
        buf,
        format='jpeg',
        quality=100,
        optimize=True,
        qtables=[qtable, qtable, qtable],
        subsampling='4:2:0' if use_420 else '4:4:4',
    )
    decoded = np.array(Image.open(buf))
    rate = np.array(8 * len(buf.getbuffer()))
    return decoded, rate

  for index in range(input_images.shape[0]):
    if not one_channel_at_a_time:
      decoded[index], rate[index] = run_jpeg(input_images[index])
    else:
      # Run each channel separately through jpeg as a grayscale image
      # (Image.mode = 'L'.) Useful when RGB <-> YUV conversions need to be
      # skipped.
      for channel in range(input_images.shape[-1]):
        decoded[index, ...,
                channel], channel_rate = run_jpeg(input_images[index, ...,
                                                               channel])
        rate[index] += channel_rate

  return decoded.astype(np.float32), rate.astype(np.float32)


def convert_420_to_444(
    inputs: tf.Tensor,
    method: tf.image.ResizeMethod = tf.image.ResizeMethod.LANCZOS3,
) -> tf.Tensor:
  """Converts a YUV420 tensor to YUV444.

  Args:
    inputs: Tensor of size [b, n, m, c] or [b, f, n, m, c] where b is batch
      size, f is the number of frames in a video clip, [n, m] is the image/frame
      shape, and c is the number of channels. When input rank is 4 inputs is a
      batch of images. When input rank is 5 inputs is a batch of video clips.
    method: Desired chroma resizing method. Using bilinear will result in the
      center pixel.

  Returns:
    outputs: Tensor of the same size as inputs where UV channels have been
    upsampled.
  """
  outputs_chroma = tf.reshape(inputs, [-1, *inputs.shape[-3:]])[..., 1:]
  outputs_chroma = outputs_chroma[
      :, 0 : outputs_chroma.shape[1] // 2, 0 : outputs_chroma.shape[2] // 2, :
  ]
  new_size = [outputs_chroma.shape[1] * 2, outputs_chroma.shape[2] * 2]
  # Upsample chroma.
  outputs_chroma = tf.image.resize(
      outputs_chroma, new_size, method=method, antialias=True
  )
  num_chroma_channels = inputs.shape[-1] - 1
  outputs_chroma = tf.reshape(
      outputs_chroma, [*inputs.shape[:-1], num_chroma_channels]
  )
  return tf.concat([inputs[..., 0:1], outputs_chroma], axis=-1)


def convert_444_to_420(
    inputs: tf.Tensor,
    method: tf.image.ResizeMethod = tf.image.ResizeMethod.BILINEAR,
) -> tf.Tensor:
  """Converts a 444 tensor to 420 by downsampling the chroma channels.

  Args:
    inputs: Tensor of size [b, n, m, c] or [b, f, n, m, c] where b is batch
      size, f is the number of frames in a video clip, [n, m] is the image/frame
      shape, and c is the number of channels. When input rank is 4 inputs is a
      batch of images. When input rank is 5 inputs is a batch of video clips.
    method: Desired chroma resizing method. Using bilinear will result in the
      center pixel.

  Returns:
    outputs: Tensor of the same size as inputs where chroma channels have been
    downsampled.
  """
  outputs_chroma = tf.reshape(inputs, [-1, *inputs.shape[-3:]])[..., 1:]
  new_size = [outputs_chroma.shape[1] // 2, outputs_chroma.shape[2] // 2]
  # Downsample chroma.
  outputs_chroma = tf.image.resize(
      outputs_chroma, new_size, method=method, antialias=True
  )
  num_chroma_channels = inputs.shape[-1] - 1
  outputs_chroma = tf.reshape(
      tf.pad(  # Zero-pad so that a single tensor is returned.
          outputs_chroma,
          [[0, 0], [0, new_size[0]], [0, new_size[1]], [0, 0]],
      ),
      [*inputs.shape[:-1], num_chroma_channels],
  )
  return tf.concat([inputs[..., 0:1], outputs_chroma], axis=-1)


class EncodeDecodeIntra(tf.keras.Model):
  """A class with methods for basic intra compression emulation."""

  def __init__(
      self,
      rounding_fn: Callable[[tf.Tensor], tf.Tensor] = tf.round,
      use_jpeg_rate_model: bool = True,
      qstep_init: float = 1.0,
      train_qstep: bool = True,
      min_qstep: float = 0.0,
      jpeg_clip_to_image_max: bool = True,
      convert_to_yuv: bool = False,
      downsample_chroma: bool = False,
  ):
    """Constructor.

    Args:
      rounding_fn: Callable that is used to round transform coefficients for
        JPEG during quantization.
      use_jpeg_rate_model: True for JPEG-specific rate model, False for
        Gaussian-distribution-based rate model.
      qstep_init: float that determines initial value for the step-size of the
        scalar quantizer.
      train_qstep: Whether qstep should be trained. When False the class will
        use qstep_init or any qsteps provided in the call(). The latter is
        useful when the same module is used in video with different qsteps for
        INTRA and INTER.
      min_qstep: Minimum value which qstep should be greater than. Set to 1 to
        reflect for some practical codecs that cannot go below integer values.
      jpeg_clip_to_image_max: True if jpeg proxy should clip the final output to
        [0, image_max]. Set to False when handling INTER frames.
      convert_to_yuv: True if color conversion should be applied during
        compression.
      downsample_chroma: Whether chroma planes should be downsampled during
        compression.
    """
    super().__init__(name='EncodeDecodeIntra')

    self.train_qstep = train_qstep
    if self.train_qstep:
      self.qstep = tf.Variable(
          initial_value=qstep_init,
          trainable=True,
          name='qstep',
          dtype=tf.float32)
    else:
      self.qstep = tf.cast(qstep_init, tf.float32)

    self.min_qstep = tf.Variable(
        initial_value=min_qstep,
        trainable=False,
        name='min_qstep',
        dtype=tf.float32)

    self.clip_to_image_max = jpeg_clip_to_image_max

    def _quantizer_fn(x: tf.Tensor) -> tf.Tensor:
      """Implements quantize-dequantize with the trainable qstep."""
      positive_qstep = self._positive_qstep()
      return rounding_fn(x / positive_qstep) * positive_qstep

    self._jpeg_quantizer_fn = _quantizer_fn
    self._rounding_fn = rounding_fn

    def add_variable_conditionally(
        variable_name: str, condition: Optional[bool] = None
    ) -> tf.Tensor:
      if condition is not None:
        return self.add_weight(
            initializer=tf.constant_initializer(condition),
            trainable=False,
            name=variable_name,
            dtype=tf.bool,
        )
      else:
        return self.add_weight(
            trainable=False, name=variable_name, dtype=tf.bool
        )

    self.use_jpeg_rate_model = add_variable_conditionally(
        'use_jpeg_rate_model', use_jpeg_rate_model
    )

    self._init_jpeg_layer(convert_to_yuv, downsample_chroma)

    # Actual jpeg is run on processed data for rate estimates. All color
    # conversion and chroma downsampling is done during differentiable
    # processing. We hence need actual jpeg to encode single channel data
    # without any downsampling unless convert_to_yuv is True. convert_to_yuv is
    # only useful in cases where the sandwiched codec is hard-coded to convert
    # to YUV.
    self.run_jpeg_one_channel_at_a_time = add_variable_conditionally(
        'run_jpeg_one_channel_at_a_time', False if convert_to_yuv else True
    )
    self.run_jpeg_with_downsampled_chroma = add_variable_conditionally(
        'run_jpeg_with_downsampled_chroma', downsample_chroma
    )

    logging.info(
        'EncodeDecodeIntra configured with %s',
        'jpeg-rate' if use_jpeg_rate_model else 'gaussian-rate',
    )

    logging.info(
        'EncodeDecodeIntra running %s',
        '420' if downsample_chroma else '444',
    )

    logging.info(
        'EncodeDecodeIntra yuv conversion is %s',
        'on' if convert_to_yuv else 'off',
    )

    # Workaround thread-unsafe PIL library by calling init in main thread.
    Image.init()

  def _positive_qstep(self):
    return tf.keras.activations.elu(self.qstep, alpha=0.01) + self.min_qstep

  def get_qstep(self) -> tf.Tensor:
    return self._positive_qstep()

  def _init_jpeg_layer(self, convert_to_yuv: bool, downsample_chroma: bool):
    # Configure the JPEG layer to use the defined quantize-dequantize function,
    # _quantizer_fn, so that trained value of qstep gets used: Use a fixed
    # quantizer step-size of 1 for all DCT coefficients and update quantizer
    # stepsize through _quantizer_fn.
    quantization_table = np.full((8, 8), 1.0, dtype=np.float32)
    self._jpeg_layer = jpeg_proxy.JpegProxy(
        downsample_chroma=downsample_chroma,
        luma_quantization_table=quantization_table,
        chroma_quantization_table=quantization_table,
        convert_to_yuv=convert_to_yuv,
        clip_to_image_max=self.clip_to_image_max,
    )

  def _rate_proxy_gaussian(self, inputs: tf.Tensor,
                           axis: List[int]) -> tf.Tensor:
    """Calculates entropy assuming a Gaussian distribution and high-res quantization.

    Args:
      inputs: Tensor of shape [b, n1, ...].
      axis: Axis of random variable realizations, e.g., with inputs b x n1 x n2
        and axis=[1] then there are n2 Gaussian variables with potentially
        different distributions, each with samples along axis=[1].

    Returns:
      rate: Tensor of shape [b] that estimates the total number of bits needed
        to represent the values quantized with self.qstep.
    """
    assert inputs.shape.rank >= np.max(np.abs(axis))
    deviations = tf.math.reduce_std(inputs, axis=axis)
    assert deviations.shape[0] == inputs.shape[0]

    hires_entropy = tf.nn.relu(
        tf.math.log(deviations / self._positive_qstep() + np.finfo(float).eps) +
        .5 * np.log(2 * np.pi * np.exp(1)))

    # Sum the entropies for total rate
    return tf.reduce_sum(
        tf.reshape(hires_entropy, [tf.shape(inputs)[0], -1]),
        axis=1) * tf.reduce_prod(
            tf.gather(tf.cast(tf.shape(inputs), dtype=tf.float32),
                      axis)) / np.log(2)

  def _rate_proxy_jpeg(
      self, three_channel_inputs: tf.Tensor,
      dequantized_dct_coeffs: Dict[str, tf.Tensor]) -> tf.Tensor:
    """Calculates a rate proxy based on a Jpeg-specific rate model."""

    def calculate_non_zeros(dct_coeffs: Dict[str, tf.Tensor],
                            qstep: tf.float32) -> tf.Tensor:
      num_nonzeros = tf.zeros(tf.shape(three_channel_inputs)[0])
      for k in dct_coeffs:
        num_nonzeros += tf.math.reduce_sum(
            tf.reshape(
                tf.math.log(1 + tf.math.abs(dct_coeffs[k] / qstep)
                           ),  # Divide to get the quantized values
                [tf.shape(three_channel_inputs)[0], -1]),
            axis=1)
      return tf.cast(num_nonzeros, dtype=tf.float32)

    def encode_decode_inputs_with_jpeg() -> Tuple[tf.Tensor, tf.Tensor]:
      """Encodes then decodes the three_channel_inputs using actual jpeg."""
      if self.run_jpeg_one_channel_at_a_time:
        # 420 is a meaningless option for the jpeg binary here.
        use_420 = tf.convert_to_tensor(False, dtype=tf.bool)
      else:
        use_420 = self.run_jpeg_with_downsampled_chroma
      jpeg_decoded, jpeg_rate = tf.numpy_function(
          _encode_decode_with_jpeg,
          inp=[
              three_channel_inputs,
              self._positive_qstep(),
              self.run_jpeg_one_channel_at_a_time,
              use_420,
          ],
          Tout=[tf.float32, tf.float32],
      )
      jpeg_decoded.set_shape(three_channel_inputs.shape)
      jpeg_rate.set_shape(three_channel_inputs.shape[0])
      return jpeg_decoded, jpeg_rate

    ###########################################################################
    # Jpeg-specific model fits rate using number of nonzero dct coefficients.
    # For details see:
    #  Z. He and S. K. Mitra, "A unified rate-distortion analysis framework for
    #  transform coding," in IEEE Transactions on Circuits and Systems for Video
    #  Technology, vol. 11, no. 12, pp. 1221-1236, Dec. 2001.
    #
    # Generate (rate, num_nonzero) pairs, fit a weight as
    # rate ~= weight * num_nonzero, return rate approximation as weight *
    # num_nonzero.
    ###########################################################################

    # First pair using current qstep. (May consider fitting to a batch instead.)
    num_nonzero_dct_coeffs = calculate_non_zeros(dequantized_dct_coeffs,
                                                 self._positive_qstep())
    _, jpeg_rate = encode_decode_inputs_with_jpeg()

    nonzero_times_rate = tf.math.multiply(num_nonzero_dct_coeffs, jpeg_rate)
    nonzero_times_nonzero = tf.math.multiply(num_nonzero_dct_coeffs,
                                             num_nonzero_dct_coeffs)

    # This is in effect a fit within the main training using a different loss
    # function whose solution is known. Stop the gradients that may confuse
    # the optimizer.
    line_weights = tf.stop_gradient(
        tf.math.divide(nonzero_times_rate, nonzero_times_nonzero + 1))

    return tf.math.multiply(num_nonzero_dct_coeffs, line_weights)

  def is_codec_proxy_420(self) -> tf.Tensor:
    return self.run_jpeg_with_downsampled_chroma

  def _encode_decode_jpeg(
      self, inputs: tf.Tensor, image_max: tf.Tensor
  ) -> Tuple[tf.Tensor, tf.Tensor]:
    """Encodes then decodes the input using JPEG.

    Args:
      inputs: Tensor of shape [b, n, m, c] where b is batch size, n x m is the
        image size, and c is the number of channels (c <= 3).
      image_max: Maximum possible value of the image.

    Returns:
      outputs: Tensor of same shape as inputs containing the
        quantized-dequantized version of the inputs.
      rate: Tensor of shape [b] that estimates the total number of bits needed
        to encode the input into output.
    """
    if inputs.shape.rank != 4:
      raise ValueError('inputs must have rank 4.')
    if inputs.shape[-1] > 3:
      raise ValueError('jpeg layer can handle up to 3 channels.')

    # Zero-pad to three channels as needed for the jpeg layer.
    pad_dim = 3 - inputs.shape[-1]
    if pad_dim:
      paddings = tf.constant([[0, 0], [0, 0], [0, 0], [0, pad_dim]],
                             dtype=tf.int32)
      three_channel_inputs = tf.pad(
          inputs, paddings, mode='CONSTANT', constant_values=0)
    else:
      three_channel_inputs = inputs

    # Emulate integer inputs needed when using an actual intra codec.
    three_channel_inputs = self._rounding_fn(three_channel_inputs)

    # JPEG quantize-dequantize.
    dequantized_three_channels, dequantized_dct_coeffs = self._jpeg_layer(
        three_channel_inputs, self._jpeg_quantizer_fn, image_max=image_max
    )

    # May consider adding self._rounding_fn(dequantized_three_channels) if
    # desired.

    # Remove padding.
    if pad_dim:
      dequantized = tf.slice(dequantized_three_channels, [0, 0, 0, 0],
                             tf.shape(inputs))
    else:
      dequantized = dequantized_three_channels

    def gaussian_rate():
      gauss_rate = tf.zeros(tf.shape(inputs)[0])
      for k in dequantized_dct_coeffs:
        gauss_rate += self._rate_proxy_gaussian(
            dequantized_dct_coeffs[k], axis=[1])
      return gauss_rate

    def jpeg_rate():
      # When running the rate proxy one channel at a time with downsampled
      # chroma (420 without color conversion case), have to explicitly
      # downsample the chroma since the jpeg binary cannot handle this case.
      conversion_to_420_needed = (
          self.run_jpeg_one_channel_at_a_time
          and self.run_jpeg_with_downsampled_chroma
      )
      if conversion_to_420_needed:
        # Ensure that the input size is a multiple of 2.
        rate_inputs = jpeg_proxy.pad_spatially_to_multiple_of_bsize(
            three_channel_inputs, bsize=2, mode='SYMMETRIC'
        )
        rate_inputs = convert_444_to_420(rate_inputs)
      else:
        rate_inputs = three_channel_inputs

      # Scale to 8-bit range so that the rate proxy can be used with any image
      # max.
      scale = tf.math.divide(255.0, image_max)
      rate_inputs = tf.math.multiply(scale, rate_inputs)
      return self._rate_proxy_jpeg(rate_inputs, dequantized_dct_coeffs)

    rate = tf.cond(self.use_jpeg_rate_model, jpeg_rate, gaussian_rate)

    return dequantized, rate

  def __call__(
      self,
      inputs: tf.Tensor,
      input_qstep: Optional[tf.Tensor] = None,
      image_max: Optional[tf.Tensor] = None,
  ) -> Tuple[tf.Tensor, tf.Tensor]:
    """Encodes then decodes the input.

    Args:
      inputs: Tensor of shape [b, n, m, c] where b is batch size, n x m is the
        image size, and c is the number of channels.
      input_qstep: qstep to use when self.qstep is not trained.
      image_max: Maximum possible value of the image.

    Returns:
      outputs: Tensor of same size as inputs containing the
        quantized-dequantized version of the inputs.
      rate: Tensor of size b that estimates the total number of bits needed to
        encode the input into output.
    """
    if inputs.shape.rank != 4:
      raise ValueError('inputs must have rank 4.')

    if not self.train_qstep and input_qstep is not None:
      self.qstep = input_qstep

    if image_max is None:
      image_max = tf.constant(255.0, dtype=tf.float32)

    def run_jpeg():
      if inputs.shape[-1] <= 3:
        return self._encode_decode_jpeg(inputs, image_max)

      # JPEG layer handles at most three channels. Run three channels at a time.
      # (i) Run first three channels to initialize the return tensors.
      size = np.array(inputs.shape, dtype=np.int32)
      limit = size[-1]
      size[-1] = 3
      begin = np.zeros_like(size, dtype=np.int32)
      dequantized, rate = self._encode_decode_jpeg(
          tf.slice(inputs, begin, size), image_max
      )

      # (ii) Run three channels at a time and update.
      for _ in range(3, limit, 3):
        begin[-1] += 3
        size[-1] = np.minimum(limit - begin[-1], 3)
        dequantized_loop, rate_loop = self._encode_decode_jpeg(
            tf.slice(inputs, begin, size), image_max
        )

        # Update the return variables
        dequantized = tf.concat([dequantized, dequantized_loop], axis=3)
        rate += rate_loop
      return dequantized, rate

    return run_jpeg()
