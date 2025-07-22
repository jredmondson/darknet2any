import sys
import onnxruntime as ort
import os
import cv2
import argparse
import numpy as np
import time

import tensorrt

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
  description="predicts from an onnx model",
  add_help=True
  )
  parser.add_argument('--cpu', action='store_true',
    dest='cpu',
    help='sets a cpu-only inference mode')
  parser.add_argument('-i','--input','--onnx', action='store',
    dest='input', default=None,
    help='the weights file to convert')
  parser.add_argument('--image', action='store',
    dest='image', default=None,
    help='the image to test the model on')
  parser.add_argument('--image-dir', action='store',
    dest='image_dir', default=None,
    help='a directory of images to test')
  

  return parser.parse_args(args)

def onnx_image_predict(
  ort_sess, shape, image_file):
  """
  predicts classes of an image file

  Args:
  interpreter (tf.lite.Interpreter): the onnx interpreter for a model
  input_details (list[dict[string, Any]]): result of get_input_details()
  output_details (list[dict[string, Any]]): result of get_output_details()
  image_file (str): an image file to read and predict on
  Returns:
  tuple: read_time, predict_time
  """
  print(f"onnx: Reading {image_file}")

  start = time.perf_counter()
  img = cv2.imread(image_file)
  image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  image_resized = cv2.resize(image_rgb, shape, interpolation=cv2.INTER_LINEAR)
  image_resized = np.transpose(image_resized, (2, 0, 1)).astype(np.float32)
  image = np.expand_dims(image_resized, axis=0)
  
  end = time.perf_counter()
  read_time = end - start


  start = time.perf_counter()
  input_name = ort_sess.get_inputs()[0].name
  outputs = ort_sess.run(None, {input_name: image})
  end = time.perf_counter()
  predict_time = end - start

  print(f"onnx: predict for {image_file}")
  print(f"  output: {outputs}")
  print(f"  read_time: {read_time:.4f}s")
  print(f"  predict_time: {predict_time:.4f}s")

  return read_time, predict_time

options = parse_args(sys.argv[1:])
has_images = options.image is not None or options.image_dir is not None

print(f"onnx executor options: {ort.get_available_providers()}")

if options.input is not None and has_images:
  print(f"onnx: loading {options.input}")

  providers = []

  if not options.cpu:
    providers.extend(
      ['TensorrtExecutionProvider',
      'CUDAExecutionProvider'])
  
  providers.append('CPUExecutionProvider')

  # 1. Load the onnx model
  start = time.perf_counter()
  ort_sess = ort.InferenceSession(options.input,
    providers=providers)
  end = time.perf_counter()
  load_time = end - start
  print(f"  load_time: {load_time:.4f}s")
  
  print(f"  provider options: {ort_sess.get_provider_options()}")
  
  shape = None

  for input_meta in ort_sess.get_inputs():
    if input_meta.name == "input":
      shape = (
        input_meta.shape[3],
        input_meta.shape[2]
      )
      break

  if shape is not None:

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
        read_time, predict_time = onnx_image_predict(
          ort_sess, shape, image)
        
        total_read_time += read_time
        total_predict_time += predict_time


      avg_read_time = total_read_time / num_predicts
      avg_predict_time = total_predict_time / num_predicts

      print(f"onnx: time for {num_predicts} predicts")
      print(f"  read_time: total: {total_read_time:.4f}s, avg: {avg_read_time:.4f}")
      print(f"  predict_time: {total_predict_time:.4f}s, avg: {avg_predict_time:.4f}")

else:

  print("No model or image specified. Printing usage and help.")
  parse_args(["-h"])