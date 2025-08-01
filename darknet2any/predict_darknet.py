############################################################################
# --------------------------------------------------------------------------
# @author James Edmondson <james@koshee.ai>
# --------------------------------------------------------------------------
############################################################################

"""
predicts using darknet on a directory structure of images
"""

import sys
import os
import cv2
import argparse
import numpy as np
import time

from darknet2any.tool.utils import *

darknet_root = os.environ.get("DARKNET_ROOT")

if not darknet_root or not os.path.isdir(f"{darknet_root}/src-python"):
  print(f"predict_darknet: set DARKNET_ROOT to proceed")
  print(f"  for example: export DARKNET_ROOT=$HOME/projects/darknet")
  print(f"  important: modern darknet with src-python folder in root")
  print(f"  project: https://github.com/hank-ai/darknet")
  exit(1)

sys.path.append(f"{darknet_root}/src-python")

import darknet

def is_image(filename):
  """
  checks if filename is an image

  Args:
  filename (str): a filename to check extensions of
  Returns:
  bool: a map of arguments as defined by the parser
  """
  ext = os.path.splitext(filename)[1].lower()

  return ext == ".jpg" or ext == ".png"

def parse_args(args):
  """
  parses command line arguments

  Args:
  args (list): list of arguments
  Returns:
  dict: a map of arguments as defined by the parser
  """
  parser = argparse.ArgumentParser(
  description="predicts from an darknet model",
  add_help=True
  )
  parser.add_argument('-i','--input','--darknet', action='store',
    dest='input', default=None,
    help='the engine/darknet to load')
  parser.add_argument('--image', action='store',
    dest='image', default=None,
    help='the image to test the model on')
  parser.add_argument('--image-dir', action='store',
    dest='image_dir', default=None,
    help='a directory of images to test')
  parser.add_argument('-o','--output-dir', action='store',
    dest='output', default="labeled_images",
    help='a directory to place labeled images')
  # parser.add_argument('-t','--threads', action='store',
  #   dest='threads', type=int, default=1,
  #   help='the number of threads to run')
  

  return parser.parse_args(args)

def load_model(weights_file, cfg_file, names_file):
  """
  loads a darknet model
  """

  model = darknet.load_net_custom(
    cfg_file.encode("ascii"),
    weights_file.encode("ascii"), 0, 1)

  names = open(
    names_file, encoding="utf-8").read().splitlines()
  
  return model, names

def darknet_image_predict(
  model, colors, shape, classes, output, image_file):
  """
  predicts classes of an image file

  Args:
  interpreter (tf.lite.Interpreter): the darknet interpreter for a model
  input_details (list[dict[string, Any]]): result of get_input_details()
  output_details (list[dict[string, Any]]): result of get_output_details()
  image_file (str): an image file to read and predict on
  Returns:
  tuple: read_time, predict_time
  """
  print(f"darknet: Reading {image_file}")

  start = time.perf_counter()

  image = cv2.imread(image_file)
  # use OpenCV to load the image and swap OpenCV's
  # usual BGR for the RGB that Darknet requires
  image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  image_resized = cv2.resize(
    image_rgb, shape, interpolation=cv2.INTER_LINEAR)

  # create a Darknet-specific image structure with the resized image
  darknet_image = darknet.make_image(shape[0], shape[1], 3)
  darknet.copy_image_from_bytes(darknet_image, image_resized.tobytes())

  end = time.perf_counter()
  read_time = end - start


  start = time.perf_counter()

  results = darknet.detect_image(
    model, classes, darknet_image, thresh=0.4)
  darknet.free_image(darknet_image)

  end = time.perf_counter()
  predict_time = end - start

  start = time.perf_counter()
  
  end = time.perf_counter()
  process_time = end - start

  print(f"darknet: predict for {image_file}")
  print(f"  output: {results}")
  print(f"  read_time: {read_time:.4f}s")
  print(f"  predict_time: {predict_time:.4f}s")
  print(f"  post_processing: {process_time:.4f}s")

  return read_time, predict_time, process_time

def main():
  """
  main script entry point
  """

  options = parse_args(sys.argv[1:])
  has_images = options.image is not None or options.image_dir is not None

  print(f"darknet: predicting with {options.input}")

  if options.input is not None and has_images:
    
    print(f"darknet: loading {options.input}")

    if not os.path.isfile(options.input):
      print(f"predict_darknet: darknet weights cannot be read. "
        "check file exists or permissions.")
      exit(1)

    weights_file, cfg_file, names_file, basename = get_darknet_files(
      options.input)

    if weights_file is not None:
      if not os.path.isdir(options.output):
        os.makedirs(options.output)

      names_file = f"{basename}.names"

      # 1. Load the darknet model
      start = time.perf_counter()

      model, names = load_model(weights_file, cfg_file, names_file)

      end = time.perf_counter()
      load_time = end - start
      print(f"  load_time: {load_time:.4f}s")
      
      shape = None

      if model is not None:

        colors = darknet.class_colors(names)
        width = darknet.network_width(model)
        height = darknet.network_height(model)
          
        shape = (width, height)

        images = []

        if options.image is not None:
          images.append(options.image)

        if options.image_dir is not None:

          for dir, _, files in os.walk(options.image_dir):
            for file in files:
              source = f"{dir}/{file}"

              # file needs to be video extension and not already in cameras
              if is_image(file):
                images.append(source)

        total_read_time = 0
        total_predict_time = 0
        total_process_time = 0

        num_predicts = len(images)

        if num_predicts > 0:

          for image in images:
            read_time, predict_time, process_time = darknet_image_predict(
              model, colors, shape, names, options.output, image)
            
            total_read_time += read_time
            total_predict_time += predict_time
            total_process_time += process_time

          avg_read_time = total_read_time / num_predicts
          avg_predict_time = total_predict_time / num_predicts
          avg_process_time = total_process_time / num_predicts

          print(f"darknet: time for {num_predicts} predicts")
          print(f"  model_load_time: total: {load_time:.4f}s")
          print(f"  image_read_time: total: {total_read_time:.4f}s, avg: {avg_read_time:.4f}s")
          print(f"  predict_time: {total_predict_time:.4f}s, avg: {avg_predict_time:.4f}s")
          print(f"  process_time: {total_process_time:.4f}s, avg: {avg_process_time:.4f}s")

    else:
      print("predict_darknet: unable to find appropriate names/cfg files for "
        "weights file. Ideally, name.weights should have name.cfg and name.names"
        "in the same directory")
  else:

    print("No model or image specified. Printing usage and help.")
    parse_args(["-h"])
    
if __name__ == '__main__':
  main()
