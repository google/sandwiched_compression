# Sandwiched Compression

Copyright 2024 Google LLC

### Color over Grayscale Codec
![Grayscale Codec](images/400_video.gif)
### Hires over Lowres Codec
![Lowres Codec](images/lrhr.gif)
### MSE-Codec Optimized for LPIPS Loss
![Lpips loss](images/lpips.gif)


## Overview
Sandwiched compression augments a standards-based codec with pre-processor and post-processor neural networks. The primary goal is to adapt the codec to data and use-cases that are outside of the codec’s design targets. Examples include:

* Transporting high-resolution images/video over codecs that can only transport low-resolution.
* Transporting high-bit-depth (10, 12-bit) data over codecs that can only transport 8-bit.
* Catering to applications where the data will be used to satisfy a sophisticated metric different from the codec’s native metric (typically PSNR):
    LPIPS, VMAF, SSIM
* Transporting texture maps that will be used to render graphical models with view/lighting dependent metrics imposed. 
* Compressing data that will be used to enable further computations:  ARCore accomplishing SLAM using features it calculates on compressed images from a wearable, cases of calculating depth from stereo, …

The code uses a differentiable codec proxy for the standard codec. Pre and post-processors are standard unets. Pre and post-processors are trained jointly and typically implement a message passing strategy between them.

In image/video compression scenarios a nice property of this work is that the networks need to generate images/video, i.e., visual data, which the standard codecs transport. We can hence check the generated images/video (termed bottlenecks) to get an idea of what the networks are trying to accomplish.

The image codec proxy is included in this release.

## Manifest
* image_compression/encode_decode_intra_lib.py: Includes class EncodeDecodeIntra which contains the differentiable image proxy.
* image_compression/jpeg_proxy.py: Includes class JpegProxy which supports EncodeDecodeIntra.

## Usage
```python
def differentiable_round(x: tf.Tensor) -> tf.Tensor:
  """Differentiable rounding."""
  return x + tf.stop_gradient(tf.round(x) - x)


intra_compression_layer = encode_decode_intra_lib.EncodeDecodeIntra(
    rounding_fn=differentiable_round,
    qstep_init=30.0,
    train_qstep=False, # Set to True when training a sandwich.
    convert_to_yuv=True, # Set to False when training a sandwich.
)

# Any batch of 3-color, 8-bit (0-255) images in float.
images = tf.convert_to_tensor(test_images[0:16, ...])

compressed_images, rate = intra_compression_layer(images)
print(images.dtype, images.shape, compressed_images.shape, rate.shape)

# mediapy: https://github.com/google/mediapy
media.show_images(compressed_images[0:3] / 255)
```

![Output](images/image_proxy_output.png)

## References
[Image Compression](https://research.google/pubs/sandwiched-image-compression-wrapping-neural-networks-around-a-standard-codec/)
```bibtex
@INPROCEEDINGS{9506256,
  author={Guleryuz, Onur G. and Chou, Philip A. and Hoppe, Hugues and Tang, Danhang and Du, Ruofei and Davidson, Philip and Fanello, Sean},
  booktitle={2021 IEEE International Conference on Image Processing (ICIP)}, 
  title={Sandwiched Image Compression: Wrapping Neural Networks Around A Standard Codec}, 
  year={2021},
  volume={},
  number={},
  pages={3757-3761},
  doi={10.1109/ICIP42928.2021.9506256}}
```

[Video Compression](https://arxiv.org/abs/2303.11473)
```bibtex
@INPROCEEDINGS{10222313,
  author={Isik, Berivan and Guleryuz, Onur G. and Tang, Danhang and Taylor, Jonathan and Chou, Philip A.},
  booktitle={2023 IEEE International Conference on Image Processing (ICIP)}, 
  title={Sandwiched Video Compression: Efficiently Extending the Reach of Standard Codecs with Neural Wrappers}, 
  year={2023},
  volume={},
  number={},
  pages={2055-2059},
  doi={10.1109/ICIP49359.2023.10222313}}
```


