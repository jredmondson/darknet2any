from tool.darknet2pytorch import Darknet
import argparse
import os
import sys


def parse_args(args):
  """
  parses command line arguments

  Args:
    args (list): list of arguments
  Returns:
    dict: a map of arguments as defined by the parser
  """
  parser = argparse.ArgumentParser(
    description="Converts a yolov4 weights file to pytorch",
    add_help=True
  )
  parser.add_argument('-i','--input','--weights',
    action='store', dest='input', default=None,
    help='the weights file to convert')
  parser.add_argument('-o','--output','--pytorch',
    action='store', dest='output', default=None,
    help='the pytorch file to create (default=filename.pt)')

  return parser.parse_args(args)

options = parse_args(sys.argv[1:])


if not options.input:
  parse_args(["--help"])
  exit(0)

output = options.output
basename = os.path.splitext(options.input)[0]
config = f"{basename}.cfg"

if not output:
  output = f"{basename}.pt"

print("darknet2torch: input parameters:")
print(f"  config: {config}")
print(f"  weights: {options.input}")
print(f"  output: {output}")

print(f"darknet2torch: converting to {output}:")
weights = Darknet(config)
weights.load_weights(options.input)
weights.save_weights(output)

print("darknet2torch: conversion complete")
