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
  parser.add_argument('-o','--output','--onnx',
    action='store', dest='output', default=None,
    help='the onnx file to create (default=filename.onnx)')

  return parser.parse_args(args)

def convert(cfg_file, weight_file, output_name):
  """
  converts the darknet model
  """

  transform_to_onnx(cfg_file, weight_file, 1, output_name)

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

    original = prefix
    if prefix.endswith("_best"):
      prefix = prefix.replace("_best", "")

    cfg_file = f"{prefix}.cfg"
    names_file = f"{prefix}.names"
    weight_file = f"{original}.weights"
    image_path = options.image
    batch_size = 1
    output_file = f"{prefix}.onnx"

    print(f"darknet2onnx: converting darknet weights to onnx...")
    print(f"  weights_file={weight_file}")
    print(f"  names_file={names_file}")
    print(f"  cfg_file={cfg_file}")
    print(f"  target={output_file}")

    if options.output is not None:
      output_file = options.output

    convert(cfg_file, weight_file, output_file)

    print("darknet2onnx: conversion complete")
  else:
    parse_args(["-h"])


if __name__ == '__main__':
  main()
