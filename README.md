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
python darknet2onnx.py -h
```

To generate all kinds of fun formats, try:   

```
python darknet2onnx.py -i example.weights
onnx2tf -i example.onnx -o .  -otfv1pb -okv3 -oh5
python onnx2trt.py -i example.onnx
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
python predict_trt.py -i example.trt --image-dir ~/Pictures
```

This will by default create labeled images in the local `labeled_images` directory.
Check it out to see how accurate your model is.

### darknet2visualize.py

This script visualizes each layer of a yolo cnn, provided some example input
image to help you see how the layers of the cnn convolve toward the boxes.

**Recommended usage**
```
python darknet2visual.py -i {weight_path}/example.weights -o {output_image_path} --image {image_path}/my_image.jpg
```

The above would load the darknet weights at `{weight_path}/example.weights`, read the image at `{image_path}/my_image.jpg`
and save every layer of the cnn's outputs to `{output_image_path}/{image_name}_layer_{layer_id}.png`. You can start from
the minimum layer (0) by default or you can specify a starting layer such as 35 with `-l {starting_layer_id}`.

## Bugs/Issues

Feel free to make an issue on this repo if you have trouble

## Additional Help with Darknet/Yolo/Etc.

* Darknet project (maintained by Hank.ai): [Github repo](https://github.com/hank-ai/darknet) | [Discord](https://discord.gg/zSq8rtW)



## Docker

```bash
docker build -t darknet2any .
# copy the cfg and the names here
# cp /mystuff/LegoGears.cfg .
# cp /mystuff/LegoGears.names .

docker run --gpus all --rm -it \
  -v "$(pwd)":/workspace \
  darknet2any \
  python darknet2onnx.py \
    --input=LegoGears_best.weights \
    --image=DSCN1583_frame_000179.jpg
docker run --gpus all --rm -it \
  -v "$(pwd)":/workspace \
  darknet2any \
  python predict_onnx.py \
    --input=LegoGears.onnx \
    --image=DSCN1583_frame_000179.jpg
```

