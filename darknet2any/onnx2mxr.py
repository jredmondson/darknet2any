############################################################################
# --------------------------------------------------------------------------
# @author James Edmondson <james@koshee.ai>
# --------------------------------------------------------------------------
############################################################################

"""
provides conversion from onnx to tensort engine format
"""

import sys
import os
import argparse

import importlib
migraphx_loader = importlib.util.find_spec('migraphx')

if not migraphx_loader:
  print(f"darknet2any: this script requires an installation with migraphx")
  print(f"  to fix this issue from a local install, use scripts/install_amd.sh")
  print(f"  from pip, try pip install darknet2any[amd]")

  exit(1)

os.environ["PYTHONPATH"] = "/opt/rocm/lib"

import migraphx

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
  description="Converts onnx files to migraphx format for AMD gpus",
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
  parser.add_argument('--exhaustive', '--exhaustive-tune',
    action='store_true', dest='exhaustive', default=False,
    help='exhaustively optimize for the target GPU')
  parser.add_argument('--cpu',
    action='store_true', dest='cpu',
    help='target a cpu instead of the gpu')

  return parser.parse_args(args)

def convert(input_file, output_file, convert_options):
  """
  converts onnx to mxr format
  """
  model = migraphx.parse_onnx(input_file)

  target = "gpu"
  if convert_options.cpu:
    target = "cpu"

  if convert_options.int8:
    print("onnx2mxr: quantizing for int8")
    migraphx.quantize_int8(model, t=migraphx.get_target(target))
  elif convert_options.fp16:
    print("onnx2mxr: quantizing for fp16")
    migraphx.quantize_fp16(model)
  elif convert_options.bf16:
    print("onnx2mxr: quantizing for bf16")
    migraphx.quantize_bf16(model)
  else:
    print("onnx2mxr: using default quantization of fp32")

  if convert_options.exhaustive:
    print("onnx2mxr: preparing for exhaustive tuning")


  print(f"onnx2mxr: compiling migraphx for {target}")
  model.compile(t=migraphx.get_target(target),
    exhaustive_tune=convert_options.exhaustive)

  print("mxr parameters:")
  print(f"{model.get_parameter_names()}")

  print("mxr parameter shapes:")
  print(f"{model.get_parameter_shapes()}")

  migraphx.save(model, output_file, format='msgpack')

  
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

    start = time.perf_counter()
    convert(input_file, output_file, options)
    end = time.perf_counter()
    total = end - start

    print("onnx2mxr: conversion complete")

    print(f"onnx2mxr: built {output_file} in {total:.4f}s")

  else:
    parse_args(["-h"])

if __name__ == '__main__':
  main()
