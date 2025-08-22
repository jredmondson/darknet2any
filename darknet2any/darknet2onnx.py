############################################################################
# --------------------------------------------------------------------------
# @author James Edmondson <james@koshee.ai>
# --------------------------------------------------------------------------
############################################################################

"""
provides conversion from darknet to onnx format
"""

import sys
import onnx
import os
import argparse
import numpy as np
import cv2
import onnxruntime

from darknet2any.tool.utils import *
from darknet2any.tool.darknet2onnx import *

def parse_args(args):
  """
  parses command line arguments

  Args:
  args (list): list of arguments
  Returns:
  dict: a map of arguments as defined by the parser
  """
  parser = argparse.ArgumentParser(
  description="Converts a yolov4 weights file to onnx",
  add_help=True
  )
  parser.add_argument('-i','--input','--weights', action='store',
    dest='input', default=None,
    help='the weights file to convert')
  parser.add_argument('--image','--test', action='store',
    dest='image', default=None,
    help='the image to test the resulting onnx model on')
  parser.add_argument('--ie','--image-embeddings','--include-embeddings',
    action='store_true', dest='include_embeddings',
    help='add image feature embeddings to model output')
  parser.add_argument('-o','--output','--onnx',
    action='store', dest='output', default=None,
    help='the onnx file to create (default=filename.onnx)')
  parser.add_argument('--op','--opset',
    action='store', type=int, dest='opset', default=15,
    help='the opset to use for export. 15 is default.')

  return parser.parse_args(args)

def convert(cfg_file, weight_file, output_name, include_embeddings, opset):
  """
  converts the darknet model
  """

  transform_to_onnx(cfg_file, weight_file, 1, output_name, include_embeddings, opset)

def main():
  """
  main script entry point
  """
  options = parse_args(sys.argv[1:])

  if options.input is not None:
    prefix = os.path.splitext(options.input)[0]

    if not os.path.isfile(options.input):
      print(f"darknet2onnx: darknet weights file cannot be read. "
        "check file exists or permissions.")
      exit(1)

    weights_file, cfg_file, names_file, prefix = get_darknet_files(options.input)

    image_path = options.image
    batch_size = 1
    output_file = f"{prefix}.onnx"

    if weights_file is not None:
      print(f"darknet2onnx: converting darknet weights to onnx...")
      print(f"  weights_file={weights_file}")
      print(f"  names_file={names_file}")
      print(f"  cfg_file={cfg_file}")
      print(f"  target={output_file}")
      print(f"  opset={options.opset}")

      if options.output is not None:
        output_file = options.output

      convert(cfg_file, weights_file, output_file, options.include_embeddings, options.opset)

      print("darknet2onnx: conversion complete")
    else:
      print("darknet2onnx: unable to find appropriate names/cfg files for "
        "weights file. Ideally, name.weights should have name.cfg and name.names"
        "in the same directory")
  else:
    parse_args(["-h"])


if __name__ == '__main__':
  main()
