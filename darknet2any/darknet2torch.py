############################################################################
# --------------------------------------------------------------------------
# @author James Edmondson <james@koshee.ai>
# --------------------------------------------------------------------------
############################################################################

"""
provides conversion from darknet to pytorch format
"""

import argparse
import os
import sys

from darknet2any.tool.darknet2pytorch import Darknet
from darknet2any.tool.utils import get_darknet_files

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
  parser.add_argument('-f','--file',
    action='store', dest='output', default=None,
    help='an image or video to test against the pytorch')

  return parser.parse_args(args)

def main():
  """
  main script entry point
  """

  options = parse_args(sys.argv[1:])

  if not options.input:
    parse_args(["--help"])
    exit(0)

  if not os.path.isfile(options.input):
    print(f"darknet2torch: darknet weights file cannot be read. "
      "check file exists or permissions.")
    exit(1)

  output = options.output
  weights_file, cfg_file, _, basename = get_darknet_files(options.input)

  if weights_file is not None:
    if not output:
      output = f"{basename}.pt"

    print("darknet2torch: input parameters:")
    print(f"  config: {cfg_file}")
    print(f"  weights: {weights_file}")
    print(f"  output: {output}")

    model = Darknet(cfg_file)
    model.load_weights(weights_file)
    
    model.save_weights(output)

    print("darknet2torch: conversion complete")
  else:
    print("darknet2torch: unable to find appropriate names/cfg files for "
      "weights file. Ideally, name.weights should have name.cfg and name.names"
      "in the same directory")

if __name__ == '__main__':
  main()
