
import argparse
import os
import sys
import cv2
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import torchvision.transforms as transforms

from tool.darknet2pytorch import Darknet
from tool.torch_utils import do_detect

def parse_args(args):
  """
  parses command line arguments

  Args:
    args (list): list of arguments
  Returns:
    dict: a map of arguments as defined by the parser
  """
  parser = argparse.ArgumentParser(
    description="Visualizes a darknet forward pass",
    add_help=True
  )
  parser.add_argument('-i','--input','--weights',
    action='store', dest='input', default=None,
    help='the weights file to convert')
  parser.add_argument('--image','--test',
    action='store', dest='image', default=None,
    help='an input image')
  parser.add_argument('-l','--layer-start',
    action='store', dest='layer', default=None,
    help='the pytorch file to create (default=filename.png)')
  parser.add_argument('-o','--output','--output-dir',
    action='store', dest='output', default=".",
    help='the directory to save layer outputs to (default=.)')

  return parser.parse_args(args)

def visualize_backbone(
    output_dir, image_name,
    feature_maps, layer):
  """
  visualizes a yolo backbone. based on example provided by MR.KILLZ
  """

  downsample_factor = 1

  for idx in range(layer, len(feature_maps)):
    if idx in feature_maps:
      feature_map = feature_maps[idx]
      downsample_factor *= 2  # Adjust downsampling at each layer

      print(f"  exploring layer {idx} with shape {feature_map.shape}")

      # Feature map visualization
      num_channels = feature_map.shape[1]
      plt.figure(figsize=(18, 32))

      # Display first 16 channels of feature map
      for i in range(min(num_channels, 16)):
        plt.subplot(4, 4, i + 1)
        channel_data = feature_map[0, i].detach().cpu().numpy()
        # Normalize for display
        channel_data = (channel_data - channel_data.min()) / (channel_data.max() - channel_data.min())
        plt.imshow(channel_data, cmap='viridis')
        plt.axis('off')
        plt.title(f'Layer {idx + 1} - Channel {i + 1}')

      plt.suptitle(f'Feature Map at Layer {idx + 1} - Shape: {feature_map.shape}')
      #plt.show()


      print(f"  saving layer {idx} to {output_dir}/{image_name}_layer_{idx}.png")
      plt.savefig(f'{output_dir}/{image_name}_layer_{idx}.png')
      plt.close()

options = parse_args(sys.argv[1:])


if not options.input or not options.image:
  parse_args(["--help"])
  exit(0)

output = options.output
basename = os.path.splitext(options.input)[0]
config = f"{basename}.cfg"

if not output:
  output = f"{basename}.png"

image_basename = os.path.splitext(os.path.basename(options.image))[0]

print("darknet2visual: input parameters:")
print(f"  config: {config}")
print(f"  weights: {options.input}")
print(f"  output: {output}")
print(f"  layer: {options.layer}")

print(f"darknet2visual: exporting layer outputs to {output}:")
model = Darknet(config, inference=True)
model.load_weights(options.input)
model.print_network()
model.cuda()
img = cv2.imread(options.image)
resized = cv2.resize(img, (model.width, model.height))
resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)# Transform the image to a tensor
transform = transforms.ToTensor()
tensor = transform(resized)

boxes = do_detect(model, resized, 0.4, 0.6, use_cuda=True)
outputs = model.outputs

print(f"model layer outputs:")

layer_start = 0

if options.layer is not None:
  layer_start = int(options.layer)

if options.output != ".":
  os.makedirs(options.output, exist_ok=True)

visualize_backbone(output_dir=options.output, image_name=image_basename,
  feature_maps=outputs, layer=layer_start)

print("darknet2visual: conversion complete")
