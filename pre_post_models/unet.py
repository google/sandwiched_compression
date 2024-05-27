"""A UNet model for pre and postprocessing in the sandwich architecture."""

from typing import Callable, Optional, Sequence, Union
import tensorflow as tf


class EncoderBlock(tf.keras.Model):
  """Layers of Conv2D/ReLU followed by downsampling/pooling.

  The encoder block passes the input tensor through a series of Conv2D/ReLU
  layers followed by a downsampling/pooling layer which reduces spatial
  dimensions by half.
  """

  def __init__(
      self,
      num_convs: int,
      num_filters: int,
      name: str = 'encoder_block',
  ):
    """Initializes the EncoderBlock model.

    Args:
      num_convs: Integer specifying the number of convolutional layers.
      num_filters: Integer specifying the number of filters in each
        convolutional layer.
      name: Name of the model.
    """
    super().__init__(name=name)

    if num_convs <= 0 or num_filters <= 0:
      raise ValueError(
          'Incompatible convolutional layer parameters:'
          f' ({num_convs}, {num_filters})'
      )

    self._block_layers = []
    for conv_idx in range(num_convs):
      conv_layer = tf.keras.layers.Conv2D(
          filters=num_filters,
          kernel_size=3,
          strides=1,
          padding='same',
          activation='relu',
          name='conv_{}'.format(conv_idx),
      )
      self._block_layers.append(conv_layer)

    assert len(self._block_layers) == num_convs

    def pool_tensor(inputs: tf.Tensor) -> tf.Tensor:
      """Returns a pooled tensor containing every other sample."""
      input_shape = tf.shape(inputs)
      height = input_shape[1]
      width = input_shape[2]
      return inputs[:, 1:height:2, 1:width:2, :]

    self._pool_layer = pool_tensor

  def call(self, inputs: tf.Tensor) -> Sequence[tf.Tensor]:
    """Forward-pass of the EncoderBlock model.

    Args:
      inputs: Tensor of shape [b, n, m, c] where b is batch size, [n, m] is the
        image shape, and c is the number of channels.

    Returns:
      outputs: The pooled features and the features prior to pooling.
    """
    tf.debugging.assert_equal(
        tf.rank(inputs),
        4,
        message=f'inputs must have rank 4 not {tf.rank(inputs)}.',
    )
    model_outputs = inputs
    for layer in self._block_layers:
      model_outputs = layer(model_outputs)

    # model_outputs are at full spatial resolution. They can be directly fed
    # into the decoder network. Add downsampled/pooled versions for the next
    # level of the multiresolution hierarchy.
    return self._pool_layer(model_outputs), model_outputs


class DecoderBlock(tf.keras.Model):
  """A default decoder block (n Conv2D/ReLU followed by UpSampling2D).

  The decoder block passes the first input tensor through a series of
  Conv2D/ReLU layers and finally increases its spatial dimension by a
  factor of two via an UpSampling2D layer. The upsampled input features are then
  concatenated with the second input tensor (skips). The decoder block returns
  the pre-upsampled features and the concatenated ones.
  """

  def __init__(
      self, num_convs: int, num_filters: int, name: str = 'decoder_block'
  ):
    """Model initialization.

    Args:
      num_convs: Integer specifying the number of convolutions.
      num_filters: Integer specifying the number of filters in convolution.
      name: Name of the model.
    """
    super().__init__(name=name)

    self._block_layers = []
    for conv_idx in range(num_convs):
      conv_layer = tf.keras.layers.Conv2D(
          num_filters,
          kernel_size=3,
          strides=1,
          padding='same',
          activation='relu',
          name='conv_{}'.format(conv_idx),
      )
      self._block_layers.append(conv_layer)

    assert len(self._block_layers) == num_convs

    self._upsample_layer = tf.keras.layers.UpSampling2D(
        size=2, interpolation='bilinear', name='upsample'
    )

    self._concatenation_layer = tf.keras.layers.Concatenate(
        axis=-1, name='concatenate'
    )

  def call(
      self,
      inputs: tf.Tensor,
      skips: Optional[tf.Tensor] = None,
  ) -> Sequence[tf.Tensor]:
    """Forward-pass of the DecoderBlock model.

    Args:
      inputs: A tensor of shape [b, n, m, c] where b is batch size, [n, m] is
        the image shape, and c is the number of channels.
      skips: A tensor of shape [b, 2n, 2m, k]. (Setting skips to None
        faciliatates a decoding block which need not concatenate nor upsample.)

    Returns:
      outputs: When skips is not None a tensor of size
        [b, 2n, 2m, num_filters + k] which contains the upsampled features
        concatenated with the skips tensor. Else a tensor of size
        [b, n, m, num_filters].
    """
    outputs = inputs
    for layer in self._block_layers:
      outputs = layer(outputs)

    if skips is not None:
      outputs = self._upsample_layer(outputs)
      outputs = self._concatenation_layer([outputs, skips])

    return outputs


class UNet(tf.keras.Model):
  """A Unet-style network."""

  def __init__(
      self,
      encoder_filters_sequence: Sequence[int] = (32, 64, 128, 256),
      decoder_filters_sequence: Sequence[int] = (512, 256, 128, 64, 32),
      encoder_convolutions_per_block: Union[int, Sequence[int]] = 2,
      decoder_convolutions_per_block: Union[int, Sequence[int]] = 2,
      output_activation: Optional[Callable[[tf.Tensor], tf.Tensor]] = None,
      output_channels: int = 3,
      name: str = 'simple_unet',
  ):
    """Initializes the UNet model.

    Args:
      encoder_filters_sequence: Sequence of integers specifying the number of
        convolutional filters in each encoder block.
      decoder_filters_sequence: Sequence of integers specifying the number of
        convolutional filters in each decoder block.
      encoder_convolutions_per_block: Integer or sequence of integers specifying
        the number of convolutions in each encoder block.
      decoder_convolutions_per_block: Integer or sequence of integers specifying
        the number of convolutions in each decoder block.
      output_activation: Callable that implements the desired activation on the
        Unet outputs.
      output_channels: Number of channels for Unet outputs.
      name: Name of the model.
    """
    super().__init__(name=name)

    num_encoder_blocks = len(encoder_filters_sequence)
    if isinstance(encoder_convolutions_per_block, int):
      convolutions_sequence = [
          encoder_convolutions_per_block
      ] * num_encoder_blocks
    else:
      convolutions_sequence = encoder_convolutions_per_block

    # Build the analysis multiresolution ladder, i.e., the left side of the U.
    self._encoder_blocks = []
    for encoder_idx in range(num_encoder_blocks - 1):
      encoder_block = EncoderBlock(
          num_convs=convolutions_sequence[encoder_idx],
          num_filters=encoder_filters_sequence[encoder_idx],
          name='encoder_block_{}'.format(encoder_idx),
      )
      self._encoder_blocks.append(encoder_block)

    num_decoder_blocks = num_encoder_blocks + 1
    assert num_decoder_blocks == len(decoder_filters_sequence)
    if isinstance(decoder_convolutions_per_block, int):
      convolutions_sequence = [
          decoder_convolutions_per_block
      ] * num_decoder_blocks
    else:
      convolutions_sequence = decoder_convolutions_per_block

    # Build the synthesis multiresolution ladder, i.e., the right side of the U.
    self._decoder_blocks = []
    for decoder_idx in range(num_decoder_blocks):
      decoder_block = DecoderBlock(
          num_convs=convolutions_sequence[decoder_idx],
          num_filters=decoder_filters_sequence[decoder_idx],
          name='decoder_block_{}'.format(decoder_idx),
      )
      self._decoder_blocks.append(decoder_block)

    # Model to generate the final outputs of the Unet.
    self._output_layer = tf.keras.layers.Conv2D(
        filters=output_channels,
        kernel_size=3,
        strides=1,
        padding='same',
        activation=output_activation,
        name='output',
    )

  def call(self, inputs: tf.Tensor, *ignored_args) -> tf.Tensor:
    """Forward-pass of the UNet model.

    Args:
      inputs: Tensor of shape [b, n, m, c] where b is batch size, [n, m] is the
        image shape, and c is the number of channels.
      *ignored_args: Ignored arguments to maintain compatibility with other
        unets.

    Returns:
      outputs: Tensor of shape [b, n, m, output_channels].
    """

    model_outputs, skip_connections = inputs, []
    for encoder_block in self._encoder_blocks:
      model_outputs, skips = encoder_block(model_outputs)
      skip_connections.append(skips)

    # No skips/upsampling when synthesizing the finest layer.
    skip_connections.append(None)
    num_skips = len(skip_connections)

    for block_idx, decoder_block in enumerate(self._decoder_blocks):
      model_outputs = decoder_block(
          model_outputs, skip_connections[num_skips - 1 - block_idx]
      )

    return self._output_layer(model_outputs)
