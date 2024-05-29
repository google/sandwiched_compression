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

## Release
The full sandwich image compression model is included in this release. (Video compression model is coming soon.)

## Manifest
* distortion/distortion_fns.py: Distortion functions to use in distortion-rate optimization.
* image_compression/encode_decode_intra_lib.py: Includes class EncodeDecodeIntra which contains the differentiable image proxy.
* image_compression/jpeg_proxy.py: Includes class JpegProxy which supports EncodeDecodeIntra.
* pre_post_models/unet.py: Simple unet model for pre-post-processors.
* utilities/serialization.py: Checkpoint management routines.
* compress_intra_model.py: Sandwich model for image compression.
* datasets.py: Basic loader for tensorflow datasets.
* image_codec_proxy.ipynb: Colab for basic image codec proxy usage.
* sandwich_image_compression_grayscale_codec.ipynb: Colab for the grayscale codec scenario with example training and results.
* sandwich_image_compression_lowres_codec.ipynb: Colab for the lowres codec scenario with example training and results.

## Usage
Please see sandwich_image_compression_lowres_codec.ipynb and sandwich_image_compression_grayscale_codec.ipynb for two scenarios discussed in the paper. Beyond the included software you will need tensorflow-datasets (with the 'clic' dataset downloaded) and mediapy. Please see the colabs for instructions.

## References
[Image and Video Compression](https://arxiv.org/abs/2402.05887)
```bibtex
@article{guleryuz2024sandwiched,
  title={Sandwiched Compression: Repurposing Standard Codecs with Neural Network Wrappers},
  author={Onur G. Guleryuz and Philip A. Chou and Berivan Isik and Hugues Hoppe and Danhang Tang and Ruofei Du and Jonathan Taylor and Philip Davidson and Sean Fanello},
  journal={arXiv preprint arXiv:2402.05887},
  year={2024}
}
```
