# Pytorch-YOLOv4

## Apology / Introduction

This repository has residuals from work four years ago, but is repurposed
primarily for conversions from yolov4/darknet to onnx, pytorch, and other
formats. Though I have left various tools in place, I am only currently
supporting files that I have name changed or created such as:

* darknet2onnx.py
* darknet2visual.py

And to a lesser extent:

* darknet2torch.py

I will respond to any questions as best I can, but the majority of this
code is not mine and not maintained. If you need help with anything other
than these, it may be best to contact the original code creator through
the [parent repo](https://github.com/Tianxiaomo/pytorch-YOLOv4).

## Installation

```
scripts/install.sh
source .venv/bin/activate
```

## Usage

Usage is primarily focused on a weights file with associated names, cfg,
etc. that bears the same basename (e.g., `example.weights` should have
a corresponding names and cfg file in the same folder named `example.names`
and `example.cfg`). Though this is slightly less configurable, I find it
much easier to maintain in terms of technical support and, imo, helps with
intuitive usage.

Each supported script has been modified to use standard argparse, and so if
you need help, please pass `-h` or `--help` to see full options.

### darknet2onnx.py

I only need static onnx generation, so this script has been modified to just
generate a single batch onnx file where you would pass it one image at a time.
This is because I typically support real-time camera apps and when you are
reading from live cameras, you tend to do so one frame at a time. It would be
somewhat trivial to add arguments that allow for a different use case, but this
is the current functionality. Consequently, there are only three options worth
mentioning in implementation

**Bare minimum usage**

```
python darknet2onnx.py -i example.weights -o example.onnx
```

The above would read an example.weights file and output it in the local
directory. Note that this script can be ran anywhere and on anything, so
if you provide `-i {some_path}/example.weights`, it will search for any
required .cfg or .names files in `{some_path}` and assume they are called
`example.cfg` and `example.names`

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


