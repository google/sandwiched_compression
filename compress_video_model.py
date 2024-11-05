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
"""A model that uses pre and postprocessing around existing video codecs to better compress data."""

# Sandwiched Compression: Repurposing Standard Codecs with Neural Network Wrappers
# https://arxiv.org/abs/2402.05887
import enum
import math
from typing import Any, Callable, Dict, Optional, Tuple

import logging
import tensorflow as tf
from tensorflow_addons import image as tfa_image

import compress_intra_model
from image_compression import encode_decode_intra_lib
from utilities import serialization


@enum.unique
class ModelConfiguration(enum.IntEnum):
  """Configurations that help construct better optimized models."""
  MLPS_ONLY = enum.auto()
  MLPS_AND_UNET_POST = enum.auto()
  FULL_MODEL = enum.auto()


_ModelConfigurationThresholds = (
    compress_intra_model.ModelConfigurationThresholds
)


# Follows uflow.uflow_utils.resample().
def _resample_at_coords(source: tf.Tensor, coords: tf.Tensor) -> tf.Tensor:
  """Resample the source image at the passed coordinates.

  Args:
    source: Batch of images to be resampled.
    coords: Batch of coordinates in the image.

  Returns:
    The resampled image.

  Coordinates should be between 0 and size-1. Coordinates outside of this range
  are handled by interpolating with zeros.
  """
  assert coords.shape.rank == 4

  float_coords = tf.cast(coords, tf.float32)
  output = tfa_image.resampler(
      tf.cast(source, tf.float32), float_coords[:, :, :, ::-1]
  )
  return tf.cast(output, source.dtype)


# Follows uflow.uflow_utils.flow_to_warp().
def _flow_to_warp(flow: tf.Tensor) -> tf.Tensor:
  """Compute the warp from the flow field.

  Args:
    flow: Optical flow tensor.

  Returns:
    The warp, i.e. the endpoints of the estimated flow.
  """
  assert flow.shape.rank == 4
  height, width = flow.shape.as_list()[-3:-1]
  i_grid, j_grid = tf.meshgrid(
      tf.linspace(0.0, height - 1.0, height),
      tf.linspace(0.0, width - 1.0, width),
      indexing='ij',
  )
  grid = tf.expand_dims(tf.stack([i_grid, j_grid], axis=2), axis=0)
  return tf.cast(grid + flow, flow.dtype)


def derive_inter_vs_intra_mask(
    orig_frame: tf.Tensor,
    mc_prediction: tf.Tensor,
    block_size: int = 8,
    compression_fn: Optional[Callable[[tf.Tensor], tf.Tensor]] = None
) -> Tuple[tf.Tensor, tf.Tensor]:
  """Derives a mask where motion-compensated prediction outperforms intra pred.

  Args:
    orig_frame: The original video frame.
    mc_prediction: Motion-compensated prediction of the original.
    block_size: block-size over which intra-prediction should be compared to
      motion-compensated prediction.
    compression_fn: Callable that accomplishes compression on the original frame
      in a way to emulate the impact of compression on intra prediction.

  Returns:
    mc_mask: The binary mask marking blocks where motion-compensated prediction
      out-performs intra-prediction.
    masked_mc_prediction: The masked-motion-compensated_prediction.
  """

  def get_patch_mse(picture):
    patched = tf.image.extract_patches(
        images=picture,
        sizes=[1, block_size, block_size, 1],
        strides=[1, block_size, block_size, 1],
        rates=[1, 1, 1, 1],
        padding='VALID')
    return tf.reduce_sum(tf.square(patched), axis=-1)

  # Compressed version of the original frame to derive an intra prediction from.
  # Without the impact of compression at low rates intra_prediction_proxy will
  # win over inter much more frequently.
  compressed_orig_frame = compression_fn(
      orig_frame) if compression_fn else orig_frame

  # Simple intra-prediction proxy that downsamples then upsamples.
  new_size = tf.math.floordiv(orig_frame.shape[1:3], block_size)
  intra_prediction_proxy = tf.image.resize(
      compressed_orig_frame,
      size=new_size,
      method=tf.image.ResizeMethod.BICUBIC,
      antialias=True)
  intra_prediction_proxy = tf.image.resize(
      intra_prediction_proxy,
      size=orig_frame.shape[1:3],
      method=tf.image.ResizeMethod.NEAREST_NEIGHBOR,
      antialias=False)

  intra_pred_mse = get_patch_mse(orig_frame - intra_prediction_proxy)
  mc_pred_mse = get_patch_mse(orig_frame - mc_prediction)

  mc_mask = tf.where(intra_pred_mse >= mc_pred_mse,
                     tf.ones_like(intra_pred_mse),
                     tf.zeros_like(intra_pred_mse))
  mc_mask = tf.expand_dims(mc_mask, axis=-1)
  new_size = tf.math.multiply(mc_mask.shape[1:3], block_size)
  mc_mask = tf.image.resize(
      mc_mask,
      size=new_size,
      method=tf.image.ResizeMethod.NEAREST_NEIGHBOR,
      preserve_aspect_ratio=True,
      antialias=False,
  )
  masked_mc_prediction = tf.math.multiply(
      mc_mask, mc_prediction) + tf.math.multiply(1 - mc_mask,
                                                 intra_prediction_proxy)
  return mc_mask, masked_mc_prediction


class PreprocessCompressPostprocessInter(tf.keras.Model):
  """Builds a preprocess -> compress -> postprocess network.

  This class constructs a network by chaining together two autoencoder models
  (e.g., two UNets), as a preprocessor and a postprocessor around basic video
  compression.
  """

  def __init__(
      self,
      model_keys: Tuple[str, ...] = ('clip',),
      preprocessor_layer: Optional[tf.keras.Model] = None,
      postprocessor_layer: Optional[tf.keras.Model] = None,
      intra_compression_layer: Optional[tf.keras.Model] = None,
      loop_filter_layer: Optional[tf.keras.Model] = None,
      downsample_factor: int = 1,
      gamma: Optional[float] = None,
      bottleneck_channels: int = 1,
      output_channels: int = 3,
      num_mlp_layers: int = 2,
      num_mlp_nodes: int = 16,
      qstep_init: float = 1.0,
      min_qstep: float = 0.0,
      model_config_thresholds: _ModelConfigurationThresholds = _ModelConfigurationThresholds(),
      name: str = 'PreprocessCompressPostprocessInter',
      intra_dir: Optional[str] = None,
      video_is_420: bool = False,
      codec_proxy_is_420: bool = False,
      custom_postprocessor_layer: Optional[tf.keras.Model] = None,
  ):
    """Initializes the PreprocessCompressPostprocessInter model.

    Args:
      model_keys: Keys used to extract model relevant tensors from the model
        dictionary input.
      preprocessor_layer: tf.keras.Model that implements pre-processing.
      postprocessor_layer: tf.keras.Model that implements post-processing.
      intra_compression_layer: Layer that implements intra compression
        emulation.
      loop_filter_layer: Pre-trained layer that implements loop filtering.
      downsample_factor: Amount by which bottleneck channels should be spatially
        downsampled for wrapping around LR standard codecs to transport HR
        content (downsample_factor = 1 for no downsampling).
      gamma: float that establishes the Lagrange multiplier in the optimized
        function "distortion + gamma * rate".
      bottleneck_channels: Number of channels in the bottleneck that undergo
        compression.
      output_channels: Number of channels at the output of the network.
      num_mlp_layers: Number of layers in mlp pre and postprocessors
        (num_mlp_layers=0 for identity, num_mlp_layers=1 for linear layers.)
      num_mlp_nodes: Number of nodes in the mlp hidden layer. Ignored for
        identity or linear layers.
      qstep_init: Initial quantization stepsize.
      min_qstep: Minimum quantization stepsize.
      model_config_thresholds: Fraction of epochs over which different
        configurations of the model should be trained. Useful in changing model
        configuration during training in order to find a better minimum.
      name: string specifying a name for the model.
      intra_dir: Directory to the pretrained intra model. If None, train the
        model on both I- and P-frames.
      video_is_420: When true assumes the input video is 420 and outputs 420
        video. When false 444 video is assumed.
      codec_proxy_is_420: When true codec proxy implements 420 processing. When
        false 444 is implemented.
      custom_postprocessor_layer: Model that is run as a custom postprocessor.
        Useful in establishing custom post-processing (instead of the default
        MLP + UNet postprocessing.)
    """
    super().__init__(name=name)

    self.model_keys = model_keys
    self.qstep_inter = tf.Variable(
        initial_value=qstep_init,
        trainable=True,
        name='qstep_inter',
        dtype=tf.float32)
    self.min_qstep = min_qstep
    self.custom_postprocessor_layer = custom_postprocessor_layer

    if downsample_factor <= 0 or not isinstance(downsample_factor, int):
      raise ValueError('downsample_factor must be a positive integer.')

    self.downsample_factor = downsample_factor

    self.qstep_intra = tf.Variable(
        initial_value=qstep_init,
        trainable=True,
        name='qstep_intra',
        dtype=tf.float32,
    )

    if preprocessor_layer is None:
      self._unet_preprocessor = lambda x, y: tf.zeros_like(x)
    else:
      self._unet_preprocessor = preprocessor_layer

    if postprocessor_layer is None:
      self._unet_postprocessor = lambda x, y: tf.zeros_like(x)
    else:
      self._unet_postprocessor = postprocessor_layer

    self.num_mlp_layers = tf.Variable(
        initial_value=num_mlp_layers,
        trainable=False,
        name='num_mlp_layers',
        dtype=tf.int32)

    unet_scalers_are_trainable = True
    unet_scalers_initializer = 0.0
    if self.num_mlp_layers > 0:
      self._mlp_preprocessor = compress_intra_model.create_mlp_model(
          num_layers=self.num_mlp_layers.numpy(),
          num_channels=num_mlp_nodes,
          output_channels=bottleneck_channels,
          activation=tf.math.sin,
          name='MLPPreprocessor')
      self._mlp_postprocessor = compress_intra_model.create_mlp_model(
          num_layers=self.num_mlp_layers.numpy(),
          num_channels=num_mlp_nodes,
          output_channels=output_channels,
          activation=tf.math.sin,
          name='MLPPostprocessor')
    else:
      self._mlp_preprocessor = lambda x, y: x
      self._mlp_postprocessor = lambda x, y: x
      unet_scalers_are_trainable = False
      unet_scalers_initializer = 1.0

    # Scales the preprocessors so that
    # bottlenecks = self._unet_preprocessor_scaler * self._unet_preprocessor(..)
    #    + self._mlp_preprocessor(..)
    # The scaling value is trained.
    self._unet_preprocessor_scaler = self.add_weight(
        initializer=tf.constant_initializer(unet_scalers_initializer),
        trainable=unet_scalers_are_trainable,
        name='preprocessor_scaler',
        dtype=tf.float32)

    # Same as above but for the posptrocessors and final output of the model.
    self._unet_postprocessor_scaler = self.add_weight(
        initializer=tf.constant_initializer(unet_scalers_initializer),
        trainable=unet_scalers_are_trainable,
        name='postprocessor_scaler',
        dtype=tf.float32)

    # On/off switching of the unet preprocessor and postprocessors. Useful
    # when shifting training regiments.
    self._unet_preprocessor_switch = self.add_weight(
        initializer=tf.constant_initializer(1.0),
        trainable=False,
        name='unet_preprocessor_switch',
        dtype=tf.float32)

    self._unet_postprocessor_switch = self.add_weight(
        initializer=tf.constant_initializer(1.0),
        trainable=False,
        name='unet_postprocessor_switch',
        dtype=tf.float32)

    self.video_is_420 = video_is_420
    self.codec_proxy_is_420 = codec_proxy_is_420
    if intra_compression_layer is None:
      intra_compression_layer = encode_decode_intra_lib.EncodeDecodeIntra(
          rounding_fn=compress_intra_model.differentiable_round,
          train_qstep=False,
          use_jpeg_rate_model=True,
          jpeg_clip_to_image_max=False,
          downsample_chroma=self.codec_proxy_is_420,
      )

    self._intra_compression_layer = intra_compression_layer

    tf.debugging.assert_equal(
        self._intra_compression_layer.is_codec_proxy_420(),
        self.codec_proxy_is_420,
    )

    if loop_filter_layer is None:
      self._loop_filter_layer = tf.zeros_like
    else:
      self._loop_filter_layer = loop_filter_layer

    if gamma is not None:
      self.gamma = self.add_weight(
          initializer=tf.constant_initializer(gamma),
          trainable=False,
          name='gamma',
          dtype=tf.float32)
    else:
      self.gamma = self.add_weight(
          trainable=False, name='gamma', dtype=tf.float32)

    self._configuration = self.add_weight(
        initializer=tf.constant_initializer(ModelConfiguration.FULL_MODEL),
        trainable=False,
        name='configuration',
        dtype=tf.int32)

    # Useful in changing model configuration: For the first
    #   self._train_postprocessor_only_threshold * 100
    # percent of epochs use the mlps and the unet_postprocessor to run/train the
    # respective models. Then shift to regular operation where all models are
    # run/trained together. This finds better local minima in some cases. Set
    # to zero to turn the functionality off.
    self._train_mlps_only_threshold = model_config_thresholds.mlps_only
    self._train_postprocessor_only_threshold = (
        model_config_thresholds.mlps_and_unet_post
    )

    # Adjust mean/scale to improve training. Adjustment are made at the input of
    # networks and inverted at the output.
    self.mean_adjust = tf.constant(128, dtype=tf.float32)
    self.scale_adjust = tf.constant(255, dtype=tf.float32)

  def _dict_to_model_inputs(self, input_dict: Dict[str, tf.Tensor],
                            model_keys: Tuple[str, ...]) -> tf.Tensor:
    """Returns a tensor formed using model_keys from input_dict."""
    return tf.concat(
        [tf.cast(input_dict[key], dtype=tf.float32) for key in model_keys],
        axis=-1)

  def _model_predictions_to_dict(
      self, predictions: tf.Tensor, input_dict: Dict[str, tf.Tensor],
      model_keys: Tuple[str, ...]) -> Dict[str, tf.Tensor]:
    """Returns a dict of tensors by slicing predictions."""
    example = {}
    begin_channel = 0
    for key in model_keys:
      end_channel = begin_channel + input_dict[key].shape[-1]
      example[key] = predictions[..., begin_channel:end_channel]
      begin_channel = end_channel
    return example

  def _configure_model(self):
    if self._configuration == ModelConfiguration.MLPS_ONLY:
      self._unet_preprocessor_switch.assign(0)
      self._unet_postprocessor_switch.assign(0)
    elif self._configuration == ModelConfiguration.MLPS_AND_UNET_POST:
      self._unet_preprocessor_switch.assign(0)
      self._unet_postprocessor_switch.assign(1)
    else:
      self._unet_preprocessor_switch.assign(1)
      self._unet_postprocessor_switch.assign(1)

  def _permute(self, inputs) -> tf.Tensor:
    """Permutes the input frames and flows.

      Use this to permute frames/flowswith size [b,f,n,m,c] to [f,b,n,m,c] for
      parallel computation, where b is batch_size, f is number of frames/flows,
      [n, m] is the frame size, c is the number of channels.

    Args:
      inputs: input frames or flows.

    Returns:
      outputs: permuted frames or flows.
    """
    return tf.transpose(inputs, [1, 0, 2, 3, 4])

  def get_loop_filter_layer(self) -> Any:
    return self._loop_filter_layer

  def get_gamma(self) -> tf.float32:
    """Returns the value of gamma used in the optimization of this model."""
    return self.gamma

  def set_gamma(self, gamma: float):
    """Sets the gamma so that the model can be evaluated at different gamma."""
    self.gamma.assign(gamma)

  def get_qstep_inter(self) -> tf.Variable:
    """Returns the qstep used by the codec proxy. Useful for logging."""
    return self.qstep_inter

  def get_qstep_intra(self) -> tf.Variable:
    """Returns the qstep used by the codec proxy. Useful for logging."""
    return self.qstep_intra

  def get_pre_post_scalers(self) -> Tuple[tf.Tensor, tf.Tensor]:
    """Returns the scalers used for the networks. Useful for logging."""
    pre = self._unet_preprocessor_switch * self._unet_preprocessor_scaler
    post = self._unet_postprocessor_switch * self._unet_postprocessor_scaler
    return pre, post

  def set_configuration(self, epoch: int, num_epochs: int) -> bool:
    """Sets the training sequence to postprocessor first then both."""
    if epoch < 0 or num_epochs < epoch:
      raise ValueError(f'Got epoch {epoch}, num_epochs {num_epochs}. '
                       'Need 0 <= epoch <= num_epochs.')
    training_state = epoch / num_epochs if num_epochs else 1

    def is_config_changed(next_config: ModelConfiguration) -> bool:
      return True if self._configuration != next_config else False

    if training_state < self._train_mlps_only_threshold:
      # Configure to run/train only the mlps.
      config_changed = is_config_changed(ModelConfiguration.MLPS_ONLY)
      self._configuration.assign(ModelConfiguration.MLPS_ONLY)
    elif training_state < self._train_postprocessor_only_threshold:
      # Configure to run/train the mlps and the unet postprocessor.
      config_changed = is_config_changed(ModelConfiguration.MLPS_AND_UNET_POST)
      self._configuration.assign(ModelConfiguration.MLPS_AND_UNET_POST)
    else:
      # Configure to run/train all.
      config_changed = is_config_changed(ModelConfiguration.FULL_MODEL)
      self._configuration.assign(ModelConfiguration.FULL_MODEL)

    if config_changed:
      self._configure_model()
    return config_changed

  def run_preprocessor(self, inputs: tf.Tensor, training: bool) -> tf.Tensor:
    """Runs the preprocessor and generates the bottleneck."""
    tf.debugging.assert_rank(inputs, 4)
    adjusted_inputs = tf.math.divide(
        inputs - self.mean_adjust, self.scale_adjust
    )
    scale = self._unet_preprocessor_switch * self._unet_preprocessor_scaler
    output = (
        tf.math.multiply(
            self.scale_adjust,
            self._mlp_preprocessor(adjusted_inputs, training)
            + scale * self._unet_preprocessor(adjusted_inputs, training),
        )
        + self.mean_adjust
    )
    if self.downsample_factor > 1:
      output = compress_intra_model.downsample_tensor(
          output,
          self.downsample_factor,
          method=tf.image.ResizeMethod.BICUBIC,
      )
    return output

  def run_postprocessor(self, compressed_bottleneck: tf.Tensor,
                        training: bool) -> tf.Tensor:
    """Runs the postprocessor on the compressed bottleneck."""
    if self.custom_postprocessor_layer:
      return self.custom_postprocessor_layer(compressed_bottleneck)
    inputs = compressed_bottleneck
    if self.downsample_factor > 1:
      inputs = compress_intra_model.upsample_tensor(
          inputs,
          self.downsample_factor,
          method=tf.image.ResizeMethod.LANCZOS3,
      )
    tf.debugging.assert_rank(inputs, 4)
    adjusted_inputs = tf.math.divide(
        inputs - self.mean_adjust, self.scale_adjust
    )
    scale = self._unet_postprocessor_switch * self._unet_postprocessor_scaler
    return (
        tf.math.multiply(
            self.scale_adjust,
            self._mlp_postprocessor(adjusted_inputs, training)
            + scale * self._unet_postprocessor(adjusted_inputs, training),
        )
        + self.mean_adjust
    )

  def apply_loop_filter(self, compressed_bottleneck: tf.Tensor) -> tf.Tensor:
    """Runs the loop-filtering proxy on the compressed bottleneck."""
    filtered = tf.concat(
        [
            self._loop_filter_layer(compressed_bottleneck[..., ch : ch + 1])
            for ch in range(compressed_bottleneck.shape[-1])
        ],
        axis=-1,
    )

    return compressed_bottleneck + filtered

  def set_bit_depth(self, bit_depth: tf.Tensor):
    """Sets the bit depth of the input/output video."""
    # Tensors throughout to allow for different bit depths in a batch of clips.
    tf.debugging.assert_greater(tf.math.reduce_min(bit_depth), 0.0)
    tf.debugging.assert_less_equal(tf.math.reduce_max(bit_depth), 12.0)

    # Reshape to rank-4 so that broadcasting works when needed.
    self.scale_adjust = tf.reshape(
        tf.math.pow(2.0, bit_depth) - 1.0, [bit_depth.shape[0], 1, 1, 1]
    )
    self.mean_adjust = tf.math.round(self.scale_adjust / 2)

  def get_warp(self, flow: tf.Tensor) -> tf.Tensor:
    warp = _flow_to_warp(flow)
    if self.downsample_factor > 1:
      warp = compress_intra_model.downsample_tensor(
          warp,
          self.downsample_factor,
          method=tf.image.ResizeMethod.NEAREST_NEIGHBOR,
      )
    return warp

  def call(self,
           input_dict: Dict[str, tf.Tensor],
           training: Optional[bool] = None) -> Dict[str, Any]:
    """Forward-pass of the PreprocessCompressPostprocess model.

    Args:
      input_dict: Dictionary containing tensors that will be used to derive a
        model input tensor of shape [b,f,n,m,c] where b is batch size, f is the
        number of frames, [n,m] is the frame shape, and c is the number of
        channels.
      training: bool that defines whether the call should be executed as a
        training or an inference call.

    Returns:
      outputs: Dictionary containing the prediction, rate, and bottleneck
        tensors. When the model is configured for 420 video, prediction will
        be 420 video with chroma zero-padded to full size. All other output
        tensors remain 444.
    """
    inputs = self._dict_to_model_inputs(input_dict, model_keys=self.model_keys)
    self.set_bit_depth(tf.cast(input_dict.get('bit_depth', [8]), tf.float32))

    # Convert 420 inputs to 444.
    if self.video_is_420:
      inputs = encode_decode_intra_lib.convert_420_to_444(
          inputs, method=tf.image.ResizeMethod.LANCZOS3
      )

    inputs = self._permute(inputs)
    backward_flows = self._dict_to_model_inputs(
        input_dict, model_keys=('backward_flow',))
    flows = self._permute(backward_flows)

    # First, compress the intra frame.
    bottleneck = self.run_preprocessor(inputs[0], training)

    # Use intra qstep.
    compressed_bottleneck, rate = self._intra_compression_layer(
        tf.clip_by_value(bottleneck, 0.0, self.scale_adjust),
        self.get_qstep_intra(),
        image_max=self.scale_adjust,
    )
    compressed_bottleneck = self.apply_loop_filter(compressed_bottleneck)
    prev_recons_bottleneck = compressed_bottleneck
    prediction = self.run_postprocessor(compressed_bottleneck, training)

    total_rate = rate
    bottlenecks = [bottleneck]
    errors = [bottleneck]
    compressed_errors = [compressed_bottleneck]
    recons_bottlenecks = [compressed_bottleneck]
    predictions = [prediction]

    def compression_fn(x: tf.Tensor) -> tf.Tensor:
      return self._intra_compression_layer(
          x, self.get_qstep_intra(), image_max=self.scale_adjust
      )[0]

    # Now, compress the predicted frames.
    for frame_id in range(1, inputs.shape[0]):
      bottleneck = self.run_preprocessor(inputs[frame_id], training=training)
      bottleneck = tf.clip_by_value(bottleneck, 0.0, self.scale_adjust)
      bottlenecks.append(bottleneck)

      # Generate the warp for motion compensation.
      warp = self.get_warp(flows[frame_id - 1])

      # Perform motion compensation.
      compensated_bottleneck = _resample_at_coords(
          prev_recons_bottleneck, warp
      )
      _, compensated_bottleneck = derive_inter_vs_intra_mask(
          bottleneck, compensated_bottleneck, compression_fn=compression_fn)

      # Calculate displaced frame difference.
      bottleneck -= compensated_bottleneck

      # Shift the error frame to [0, self.scale_adjust] range (approximately)
      # to be able to encode it with the jpeg proxy.
      bottleneck += self.mean_adjust

      errors.append(bottleneck)
      # Use inter qstep.
      compressed_bottleneck, rate = self._intra_compression_layer(
          bottleneck, self.get_qstep_inter(), image_max=self.scale_adjust
      )

      total_rate += rate
      compressed_errors.append(compressed_bottleneck)

      # Shift the compressed error frame back.
      compressed_bottleneck -= self.mean_adjust

      # Add the compensated bottleneck to the compressed bottleneck.
      compressed_bottleneck += compensated_bottleneck
      compressed_bottleneck = self.apply_loop_filter(compressed_bottleneck)

      prev_recons_bottleneck = compressed_bottleneck
      recons_bottlenecks.append(compressed_bottleneck)

      prediction = self.run_postprocessor(compressed_bottleneck, training)
      predictions.append(prediction)

    predictions = self._permute(
        tf.convert_to_tensor(predictions, dtype=tf.float32))
    bottlenecks = self._permute(
        tf.convert_to_tensor(bottlenecks, dtype=tf.float32))
    recons_bottlenecks = self._permute(
        tf.convert_to_tensor(recons_bottlenecks, dtype=tf.float32))
    errors = self._permute(tf.convert_to_tensor(errors, dtype=tf.float32))
    compressed_errors = self._permute(
        tf.convert_to_tensor(compressed_errors, dtype=tf.float32))

    # If custom layer is not set, convert 444 predictions to 420
    # (chroma zero-padded). Otheriwse assume custom layer will handle any needed
    # conversion to 420.
    # Note that given D_sum = Dy + Du + Dv,
    #   with 444, per-pixel D = D_sum / (3 N).
    #   with 420 (zero-padded), per-pixel D = D_sum / ( 3 N / 2).
    # Hence distortion needs to be scaled by 2 when calculating the loss.
    if self.video_is_420 and self.custom_postprocessor_layer is None:
      predictions = encode_decode_intra_lib.convert_444_to_420(predictions)

    output_dict = self._model_predictions_to_dict(
        predictions=predictions,
        input_dict=input_dict,
        model_keys=self.model_keys)
    output_dict.update({
        'prediction': predictions,
        'rate': total_rate,
        'bottleneck': bottlenecks,
        'recons_bottleneck': recons_bottlenecks,
        'errors': errors,
        'compressed_errors': compressed_errors
    })
    return output_dict


def create_basic_model(
    model_keys: Tuple[str, ...] = ('clip',),
    bottleneck_channels: int = 1,
    output_channels: int = 3,
    num_mlp_layers: int = 2,
    use_video_codec_rate_model: bool = False,
    downsample_factor: int = 1,
    gamma: float = 1,
    model_config_thresholds: _ModelConfigurationThresholds = _ModelConfigurationThresholds(
        mlps_only=0, mlps_and_unet_post=0
    ),
    intra_dir: Optional[str] = None,
    loop_filter_folder: Optional[str] = None,
    use_unet_preprocessor: bool = True,
    use_unet_postprocessor: bool = True,
    video_is_420: bool = False,
    codec_proxy_is_420: bool = False,
) -> tf.keras.Model:
  """Constructs the Keras model for PreprocessCompressPostprocessInter.

  Args:
    model_keys: Keys used to extract model relevant tensors from the model
      dictionary input.
    bottleneck_channels: Number of channels in the bottleneck that undergo
      compression.
    output_channels: Number of channels at the output of the network.
    num_mlp_layers: Number of layers in mlp pre and postprocessors
      (num_mlp_layers=0 for identity, num_mlp_layers=1 for linear layers.)
    use_video_codec_rate_model: True for JPEG rate model, False for Gaussian
      rate model.
    downsample_factor: Amount by which bottleneck channels should be spatially
      downsampled for wrapping around LR standard codecs to transport HR content
      (downsample_factor = 1 for no downsampling).
    gamma: float that establishes the Lagrange multiplier in the optimized
      function "distortion + gamma * rate".
    model_config_thresholds: Fraction of epochs over which different
      configurations of the model should be trained. Useful in changing model
      configuration during training in order to find a better minimum.
    intra_dir: Directory to the pretrained intra model. If None, train the model
      on both I- and P-frames.
    loop_filter_folder: Folder containing the model for loop filtering proxy.
    use_unet_preprocessor: Whether preprocessor unet is on.
    use_unet_postprocessor: Whether postprocessor unet is on.
    video_is_420: Whether the input is YUV420.
    codec_proxy_is_420: When true codec proxy implements 420 processing. When
      false 444 is implemented.

  Returns:
    Constructed Keras model.
  """

  if gamma is not None:
    qstep_init = max(math.sqrt(abs(gamma)), 1.0)
  else:
    qstep_init = 1.0

  loop_filter_compound_model = compress_intra_model.create_loop_filter_model(
      base_model=PreprocessCompressPostprocessInter, model_keys=model_keys
  )

  if loop_filter_folder:
    model = serialization.load_checkpoint(
        loop_filter_folder, loop_filter_compound_model
    )
    loop_filter_layer = model.get_loop_filter_layer()
    loop_filter_layer.trainable = False
    logging.info(
        'Loop filter trainable weights: %d',
        len(loop_filter_layer.trainable_weights),
    )
  else:
    loop_filter_layer = None

  logging.info('Preprocessor unet: %d', use_unet_preprocessor)
  logging.info('Postprocessor unet: %d', use_unet_postprocessor)
  if use_unet_preprocessor:
    preprocessor_layer = compress_intra_model.basic_preprocessor_layer(
        output_channels=bottleneck_channels,
        encoder_filters_sequence=(32, 64, 128, 256),
        decoder_filters_sequence=(512, 256, 128, 64, 32),
    )
  else:
    preprocessor_layer = None

  if use_unet_postprocessor:
    postprocessor_layer = compress_intra_model.basic_postprocessor_layer(
        output_channels=output_channels,
        encoder_filters_sequence=(32, 64, 128, 256),
        decoder_filters_sequence=(512, 256, 128, 64, 32),
    )
  else:
    postprocessor_layer = None

  # UNet for pre and postprocessing.
  return PreprocessCompressPostprocessInter(
      model_keys=model_keys,
      preprocessor_layer=preprocessor_layer,
      postprocessor_layer=postprocessor_layer,
      intra_compression_layer=encode_decode_intra_lib.EncodeDecodeIntra(
          rounding_fn=compress_intra_model.differentiable_round,
          use_jpeg_rate_model=use_video_codec_rate_model,
          qstep_init=qstep_init,
          train_qstep=False,
          min_qstep=1,
          jpeg_clip_to_image_max=False,
          downsample_chroma=codec_proxy_is_420,
      ),
      loop_filter_layer=loop_filter_layer,
      downsample_factor=downsample_factor,
      gamma=gamma,
      bottleneck_channels=bottleneck_channels,
      output_channels=output_channels,
      num_mlp_layers=num_mlp_layers,
      qstep_init=qstep_init,
      min_qstep=1,
      model_config_thresholds=model_config_thresholds,
      intra_dir=intra_dir,
      video_is_420=video_is_420,
      codec_proxy_is_420=codec_proxy_is_420,
  )
