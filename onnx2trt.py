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
import tensorrt as trt

from tool.utils import *
from tool.darknet2onnx import *

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

def main(input_file, output_file):

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
  # Optional: Enable FP16 precision
  # config.set_flag(trt.BuilderFlag.FP16)

  serialized_engine = builder.build_serialized_network(network, config)
  with open(output_file, "wb") as f:
      f.write(serialized_engine)

if __name__ == '__main__':
  options = parse_args(sys.argv[1:])

  if options.input is not None:
    prefix = os.path.splitext(options.input)[0]

    cfg_file = f"{prefix}.cfg"
    names_file = f"{prefix}.names"
    input_file = f"{prefix}.onnx"
    output_file = f"{prefix}.trt"

    if options.output is not None:
      output_file = options.output

    start = time.perf_counter()
    main(input_file, output_file)
    end = time.perf_counter()
    total = end - start

    print(f"onnx2trt: built {output_file} in {total:.4f}s")

  else:
    parse_args(["-h"])
