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

tensorrt_loader = importlib.util.find_spec('tensorrt')

if not tensorrt_loader:
  print(f"darknet2any: this script requires an installation with tensorrt")
  print(f"  to fix this issue from a local install, use scripts/install_tensorrt.sh")
  print(f"  from pip, try pip install darknet2any[tensorrt]")

  exit(1)

import tensorrt as trt

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
  parser.add_argument('--fp16',
    action='store_true', dest='fp16',
    help='quantize for fp16')
  parser.add_argument('--bf16',
    action='store_true', dest='bf16',
    help='quantize for bf16')
  parser.add_argument('--int8',
    action='store_true', dest='int8',
    help='quantize for int8')

  return parser.parse_args(args)

def convert(input_file, output_file, convert_options):
  """
  converts onnx to trt format
  """

  TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
  builder = trt.Builder(TRT_LOGGER)

  network = builder.create_network(1 << int(
    trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
  parser = trt.OnnxParser(network, TRT_LOGGER)

  with open(input_file, "rb") as model_file:
    if not parser.parse(model_file.read()):
      print("ERROR: Failed to parse the ONNX file.")
      for error in range(parser.num_errors):
        print(parser.get_error(error))
      exit()

  config = builder.create_builder_config()
  config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)

  if convert_options.int8:
    print("onnx2trt: quantizing for int8")
    config.set_flag(trt.BuilderFlag.INT8)
  elif convert_options.fp16:
    print("onnx2trt: quantizing for fp16")
    config.set_flag(trt.BuilderFlag.FP16)
  elif convert_options.bf16:
    print("onnx2trt: quantizing for bf16")
    config.set_flag(trt.BuilderFlag.BF16)
  else:
    print("onnx2trt: using default quantization of fp32")
  
  serialized_engine = builder.build_serialized_network(network, config)
  with open(output_file, "wb") as f:
      f.write(serialized_engine)

def main():
  """
  main script entry point
  """

  options = parse_args(sys.argv[1:])

  if options.input is not None:
    if not os.path.isfile(options.input):
      print(f"onnx2trt: onnx file cannot be read. "
        "check file exists or permissions.")
      exit(1)

    prefix = os.path.splitext(options.input)[0]

    input_file = f"{prefix}.onnx"
    output_file = f"{prefix}.trt"

    if options.output is not None:
      output_file = options.output

    print(f"onnx2trt: converting onnx to trt...")
    print(f"  source: {input_file}")
    print(f"  target: {output_file}")

    start = time.perf_counter()
    convert(input_file, output_file, options)
    end = time.perf_counter()
    total = end - start

    print("onnx2trt: conversion complete")

    print(f"onnx2trt: built {output_file} in {total:.4f}s")

  else:
    parse_args(["-h"])

if __name__ == '__main__':
  main()
