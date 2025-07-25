# darknet2any

## Introduction

darknet2any helps you convert darknet trained models into most modern
formats including tflite, keras, onnx, and trt. It also includes sample
prediction implementations for some of these file formats so you can
make your own translations.

## Installation

```
scripts/install.sh
source .venv/bin/activate
```

## Usage

### Converting darknet to other formats

In general, any modern script has `-h` built in from the command line. If
you run into any problems, try passing in `-h`, e.g.,

```
source .venv/bin/activate
darknet2onnx -h
```

To generate all kinds of fun formats, try:   

```
source .venv/bin/activate
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
source .venv/bin/activate
predict_trt -i example.trt --image-dir ~/Pictures
```

This will by default create labeled images in the local `labeled_images` directory.
Check it out to see how accurate your model is.

### darknet2visualize

This script visualizes each layer of a yolo cnn, provided some example input
image to help you see how the layers of the cnn convolve toward the boxes.

**Recommended usage**
```
source .venv/bin/activate
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

