############################################################################
# --------------------------------------------------------------------------
# @author James Edmondson <james@koshee.ai>
# --------------------------------------------------------------------------
############################################################################

"""
provides conversion from onnx to tensort engine format
"""

import sys
import onnx
import os
import argparse
import numpy as np
import cv2
import importlib
import onnxruntime as ort

available_providers = ort.get_available_providers()

if "MIGraphXExecutionProvider" not in available_providers:
  print(f"darknet2any: this script requires darknet2any[amd] installation")
  print(f"  to fix this issue from a local install, use scripts/install_amd.sh")
  print(f"  from pipx, try pipx install darknet2any[amd]")

  exit(1)

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
  parser.add_argument('-i','--input','--onnx', action='store',
    dest='input', default=None,
    help='the weights file to convert')
  parser.add_argument('-o','--output','--trt',
    action='store', dest='output', default=None,
    help='the onnx file to create (default=filename.onnx)')

  return parser.parse_args(args)

def convert(input_file, output_file):
  """
  converts onnx to mxr format
  """
  os.environ["ORT_MIGRAPHX_SAVE_COMPILED_MODEL"] = "1"
  os.environ["ORT_MIGRAPHX_SAVE_COMPILED_PATH"] = output_file

  session = ort.InferenceSession(input_file, providers=available_providers)

  time.sleep(3.0)

def main():
  """
  main script entry point
  """

  options = parse_args(sys.argv[1:])

  if options.input is not None:
    if not os.path.isfile(options.input):
      print(f"onnx2mxr: onnx file cannot be read. "
        "check file exists or permissions.")
      exit(1)

    prefix = os.path.splitext(options.input)[0]

    input_file = f"{prefix}.onnx"
    output_file = f"{prefix}.mxr"

    if options.output is not None:
      output_file = options.output

    print(f"onnx2mxr: converting onnx to trt...")
    print(f"  source: {input_file}")
    print(f"  target: {output_file}")
    print(f"  providers: {available_providers}")

    start = time.perf_counter()
    convert(input_file, output_file)
    end = time.perf_counter()
    total = end - start

    print("onnx2mxr: conversion complete")

    print(f"onnx2mxr: built {output_file} in {total:.4f}s")

  else:
    parse_args(["-h"])

if __name__ == '__main__':
  main()
