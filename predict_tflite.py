import sys
import onnx
import os
import cv2
import argparse
import numpy as np
import time

import tensorflow as tf
from ai_edge_litert.interpreter import Interpreter

from tool.utils import *
from tool.darknet2onnx import *

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
    help='the weights file to convert')
  parser.add_argument('--image', action='store',
    dest='image', default=None,
    help='the image to test the model on')
  parser.add_argument('--image-dir', action='store',
    dest='image_dir', default=None,
    help='a directory of images to test')
  

  return parser.parse_args(args)

def tflite_image_predict(
  interpreter, input_details, output_details, image_file):
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
  
  end = time.perf_counter()
  read_time = end - start

  print(f"tflite: predicting {image_file}")

  start = time.perf_counter()
  interpreter.set_tensor(input_details[0]['index'], [image])

  # run the inference
  interpreter.invoke()

  # output_details[0]['index'] = the index which provides the input
  output_data = interpreter.get_tensor(output_details[0]['index'])

  end = time.perf_counter()

  predict_time = end - start

  print(f"tflite: predict for {image_file}")
  print(f"  output: {output_data}")
  print(f"  read_time: {read_time:.4f}s")
  print(f"  predict_time: {predict_time:.4f}s")

  return read_time, predict_time

options = parse_args(sys.argv[1:])
has_images = options.image is not None or options.image_dir is not None

tf.debugging.set_log_device_placement(True)
print(tf.config.list_physical_devices('GPU'))

if options.input is not None and has_images:

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

    num_predicts = len(images)

    if num_predicts > 0:

      for image in images:
        read_time, predict_time = tflite_image_predict(
          interpreter, input_details, output_details, image)
        
        total_read_time += read_time
        total_predict_time += predict_time


      avg_read_time = total_read_time / num_predicts
      avg_predict_time = total_predict_time / num_predicts

      print(f"tflite: time for {num_predicts} predicts")
      print(f"  read_time: total: {total_read_time:.4f}s, avg: {avg_read_time:.4f}")
      print(f"  predict_time: {total_predict_time:.4f}s, avg: {avg_predict_time:.4f}")

else:

  print("No model or image specified. Printing usage and help.")
  parse_args(["-h"])