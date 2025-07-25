# darknet2any

## Introduction

darknet2any helps you convert darknet trained models into most modern
formats including tflite, keras, onnx, and trt. It also includes sample
prediction implementations for some of these model formats so you can
make your own translations.

## Installation

```
pipx install darknet2any
```

## Tool Listing

* darknet2onnx: converts from darknet to onnx
* darknet2torch: converts from darknet to torch
* darknet2visual: visualizes all layers within darknet models
* onnx2tf: external tool for converting from onnx to tensorflow formats
* onnx2trt: converts from onnx to tensorrt engines
* predict_darknet: predicts a directory of images with darknet weights
* predict_onnx: predicts a directory of images with onnx session
* predict_tflite: predicts a directory of images with tflite model
* predict_trt: predicts a directory of images with trt engine


## Usage

### Converting darknet to other formats

In general, any modern script has `-h` built in from the command line. If
you run into any problems, try passing in `-h`, e.g.,

```
darknet2onnx -h
```

To generate all kinds of fun formats, try:   

```
darknet2onnx -i example.weights
onnx2tf -i example.onnx -o .  -otfv1pb -okv3 -oh5
onnx2trt -i example.onnx
```

The generated formats will include:   
* onnx (op11 is default atm)
* tensorrt (pretty much most optimal format on any CUDA)
* TF v1 (.pb) format
* Keras v3
* Keras v5
* TF lite

See [onnx2tf cli options](https://github.com/PINTO0309/onnx2tf?tab=readme-ov-file#cli-parameter)
for some of the extensive options available for quant options like int8, uint8, float32, etc.

### Running your trt model on image directories

```
predict_trt -i example.trt --image-dir ~/Pictures
```

This will by default create labeled images in the local `labeled_images` directory.
Check it out to see how accurate your model is.

### darknet2visualize

This script visualizes each layer of a yolo cnn, provided some example input
image to help you see how the layers of the cnn convolve toward the boxes.

**Recommended usage**
```
darknet2visual -i {weight_path}/example.weights -o {output_image_path} --image {image_path}/my_image.jpg
```

The above would load the darknet weights at `{weight_path}/example.weights`, read the image at `{image_path}/my_image.jpg`
and save every layer of the cnn's outputs to `{output_image_path}/{image_name}_layer_{layer_id}.png`. You can start from
the minimum layer (0) by default or you can specify a starting layer such as 35 with `-l {starting_layer_id}`.

## Bugs/Issues

Feel free to make an issue on this repo if you have trouble

## Project Sponsorship

* darknet2any is maintained by the team at Koshee (https://koshee.ai)

## Additional Help with Darknet/Yolo/onnx2tf.

* Darknet project (maintained by Hank.ai): [Github repo](https://github.com/hank-ai/darknet) | [Discord](https://discord.gg/zSq8rtW)
* onnx2tf project: [Github repo](https://github.com/PINTO0309/onnx2tf)

