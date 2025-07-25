############################################################################
# --------------------------------------------------------------------------
# @author James Edmondson <james@koshee.ai>
# --------------------------------------------------------------------------
############################################################################

"""
predicts using tflite on a directory structure of images
"""

import argparse
import collections
import os
import numpy as np
import re
import sys
import time

import cv2
import onnx

import tensorflow as tf

import importlib

litert_loader = importlib.util.find_spec('ai_edge_litert')

if not litert_loader:
  print(f"darknet2any: this script requires an installation with ai_edge_litert")
  print(f"  Windows and other operating systems may not support ai_edge_litert")

  exit(0)

from ai_edge_litert.interpreter import Interpreter

from darknet2any.tool.utils import *
from darknet2any.tool.darknet2onnx import *

FLOAT_SUFFIX = re.compile(r'(.*)_float[0-9]+')

Class = collections.namedtuple('Class', ['id', 'score'])

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
  description="predicts from a tflite model",
  add_help=True
  )
  parser.add_argument('--device', action='store',
    dest='device', default='/device:GPU:0',
    help='the device to use, defaults to /device:GPU:0')
  parser.add_argument('-i','--input','--tflite', action='store',
    dest='input', default=None,
    help='the tflite model to load')
  parser.add_argument('--image', action='store',
    dest='image', default=None,
    help='the image to test the model on')
  parser.add_argument('--image-dir', action='store',
    dest='image_dir', default=None,
    help='a directory of images to test')
  parser.add_argument('-o','--output-dir', action='store',
    dest='output', default="labeled_images",
    help='a directory to place labeled images')
  

  return parser.parse_args(args)

def tflite_image_predict(
  interpreter, input_details, classes, output_details, output, image_file):
  """
  predicts classes of an image file

  Args:
  interpreter (tf.lite.Interpreter): the tflite interpreter for a model
  input_details (list[dict[string, Any]]): result of get_input_details()
  output_details (list[dict[string, Any]]): result of get_output_details()
  image_file (str): an image file to read and predict on
  Returns:
  tuple: read_time, predict_time
  """
  print(f"tflite: Reading {image_file}")

  shape = (
    input_details[0]['shape'][2], # width
    input_details[0]['shape'][1], # height
  )

  start = time.perf_counter()
  img = cv2.imread(image_file)
  image = cv2.resize(img, shape)
  image = np.float32(image)
  image /= 255.0
  
  end = time.perf_counter()
  read_time = end - start

  print(f"tflite: predicting {image_file}")

  start = time.perf_counter()
  interpreter.set_tensor(input_details[0]['index'], [image])

  # run the inference
  interpreter.invoke()

  output_data = [
    interpreter.get_tensor(output_details[0]['index']),
    interpreter.get_tensor(output_details[1]['index'])
  ]

  output_data[0] = output_data[0].reshape(1, -1, 1, 4)
  output_data[1] = output_data[1].reshape(1, -1, len(classes))

  end = time.perf_counter()

  predict_time = end - start

  start = time.perf_counter()

  boxes = post_processing(img, 0.4, 0.6, output_data)
  end = time.perf_counter()
  process_time = end - start

  basename = os.path.basename(image_file)
  plot_boxes_cv2(img, boxes[0],
     savename=f"{output}/{basename}", class_names=classes)

  print(f"tflite: predict for {image_file}")
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

  tf.debugging.set_log_device_placement(True)
  print(tf.config.list_physical_devices('GPU'))

  if options.input is not None and has_images:

    if not os.path.isfile(options.input):
      print(f"predict_tflite: tflite file cannot be read. "
        "check file exists or permissions.")
      exit(1)

    basename = os.path.splitext(options.input)[0]
    m = FLOAT_SUFFIX.match(basename)
    if m:
      basename = m.group(1)

    if not os.path.isdir(options.output):
      os.makedirs(options.output)

    names_file = f"{basename}.names"
    classes = load_class_names(names_file)

    print(f"tflite: loading {options.input}")
    # 1. Load the TFLite model
    with tf.device(options.device):
      start = time.perf_counter()
      interpreter = Interpreter(model_path=options.input)
      end = time.perf_counter()
      load_time = end - start
      print(f"  load_time: {load_time:.4f}s")

      # 2. Allocate memory for tensors
      print(f"tflite: allocating tensors")
      interpreter.allocate_tensors()

      # Get input and output tensors details
      input_details = interpreter.get_input_details()
      output_details = interpreter.get_output_details()

      print(f"tflite: input_details:")
      print(input_details)

      print(f"tflite: output_details:")
      print(output_details)

      
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
          read_time, predict_time, process_time = tflite_image_predict(
            interpreter, input_details, classes, output_details,
            options.output, image)
          
          total_read_time += read_time
          total_predict_time += predict_time
          total_process_time += process_time

        avg_read_time = total_read_time / num_predicts
        avg_predict_time = total_predict_time / num_predicts
        avg_process_time = total_process_time / num_predicts

        print(f"tflite: time for {num_predicts} predicts")
        print(f"  model_load_time: total: {load_time:.4f}s")
        print(f"  image_read_time: total: {total_read_time:.4f}s, avg: {avg_read_time:.4f}s")
        print(f"  predict_time: {total_predict_time:.4f}s, avg: {avg_predict_time:.4f}s")
        print(f"  process_time: {total_process_time:.4f}s, avg: {avg_process_time:.4f}s")

  else:

    print("No model or image specified. Printing usage and help.")
    parse_args(["-h"])

if __name__ == '__main__':
  main()
