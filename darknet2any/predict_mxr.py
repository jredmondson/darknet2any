############################################################################
# --------------------------------------------------------------------------
# @author James Edmondson <james@koshee.ai>
# --------------------------------------------------------------------------
############################################################################

"""
predicts using mxr on a directory structure of images
"""

import ctypes
import sys
import os
import cv2
import argparse
import numpy as np
import time

from darknet2any.tool.utils import *

import importlib.util
migraphx_loader = importlib.util.find_spec('migraphx')

if not migraphx_loader:
  print(f"darknet2any: this script requires an installation with migraphx")
  print(f"  to fix this issue from a local install, use scripts/install_amd.sh")
  print(f"  from pip, try pip install darknet2any[amd]")

  exit(1)

os.environ["PYTHONPATH"] = "/opt/rocm/lib"

import migraphx

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
  description="predicts from a mxr model",
  add_help=True
  )
  parser.add_argument('--cpu', action='store_true',
    dest='cpu',
    help='sets a cpu-only inference mode')
  parser.add_argument('-p','--provider', '--force-provider', action='store',
    dest='provider',
    help='sets a cpu-only inference mode')
  parser.add_argument('-i','--input','--mxr', action='store',
    dest='input', default=None,
    help='the mxr model to load')
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

def mxr_image_predict(
  model, shape, classes, output, image_file):
  """
  predicts classes of an image file

  Args:
  interpreter (tf.lite.Interpreter): the mxr interpreter for a model
  input_details (list[dict[string, Any]]): result of get_input_details()
  output_details (list[dict[string, Any]]): result of get_output_details()
  image_file (str): an image file to read and predict on
  Returns:
  tuple: read_time, predict_time
  """
  print(f"mxr: Reading {image_file}")

  start = time.perf_counter()
  img = cv2.imread(image_file)
  image = cv2.dnn.blobFromImage(img, 1.0 / 255, shape, None, swapRB=True)
#   image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#   image = cv2.resize(image_rgb, shape, interpolation=cv2.INTER_LINEAR)
#   image = np.transpose(image, (2, 0, 1)).astype(np.float32)
#   image = np.expand_dims(image, axis=0)
#   image /= 255.0

  #print(f"Preprocessed image shape: {image.shape}")

  end = time.perf_counter()
  read_time = end - start

  start = time.perf_counter()
  input_name = model.get_parameter_names()[0]
  outputs = model.run({"input": image})

  for i, output in enumerate(outputs):
    if isinstance(output, migraphx.argument):
      outputs[i] = np.ndarray(shape=output.get_shape().lens(),
        buffer=np.array(output.tolist()), dtype=float)
      if False: # Migraphx provides a pointer to avoid memcpy
        print(f"converting migraphx.argument={outputs[0]}")
        addr = ctypes.cast(output.data_ptr(), ctypes.POINTER(ctypes.c_float))
        outputs[i] = np.ctypeslib.as_array(addr, shape=output.get_shape().lens())
  
  # print(f"boxes={outputs[0]}")
  # print(f"confidences={outputs[1]}")

  outputs[0] = outputs[0].reshape(1, -1, 1, 4)
  outputs[1] = outputs[1].reshape(1, -1, len(classes))

  end = time.perf_counter()
  predict_time = end - start

  start = time.perf_counter()
  boxes = post_processing(image, 0.4, 0.6, outputs)
  end = time.perf_counter()
  process_time = end - start

  basename = os.path.basename(image_file)
  plot_boxes_cv2(img, boxes[0],
    savename=f"{output}/{basename}", class_names=classes)

  print(f"mxr: predict for {image_file}")
  #print(f"  output: {outputs}")
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

  if options.input is not None and has_images:
    print(f"predict_mxr: loading {options.input}")

    if not os.path.isfile(options.input):
      print(f"predict_mxr: mxr file cannot be read. "
        "check file exists or permissions.")
      exit(1)

    basename = os.path.splitext(options.input)[0]
    names_file = f"{basename}.names"

    if not os.path.isdir(options.output):
      os.makedirs(options.output)

    providers = []

    print(f"predict_mxr: using mxr={options.input}")

    # 1. Load the mxr model
    start = time.perf_counter()
    model = migraphx.load(options.input, format='msgpack')
    end = time.perf_counter()
    load_time = end - start
    print(f"  load_time: {load_time:.4f}s")
    
    shape = None

    for param_name, param_shape in model.get_parameter_shapes().items():
      if param_name == "input":
        lens = param_shape.lens()
        shape = (
          lens[3],
          lens[2]
        )
        break

    if shape is not None:

      classes = load_class_names(names_file)
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
          read_time, predict_time, process_time = mxr_image_predict(
            model, shape, classes, options.output, image)
          
          total_read_time += read_time
          total_predict_time += predict_time
          total_process_time += process_time


        avg_read_time = total_read_time / num_predicts
        avg_predict_time = total_predict_time / num_predicts
        avg_process_time = total_process_time / num_predicts

        print(f"mxr: time for {num_predicts} predicts")
        print(f"  model_load_time: total: {load_time:.4f}s")
        print(f"  image_read_time: total: {total_read_time:.4f}s, avg: {avg_read_time:.4f}s")
        print(f"  predict_time: {total_predict_time:.4f}s, avg: {avg_predict_time:.4f}s")
        print(f"  process_time: {total_process_time:.4f}s, avg: {avg_process_time:.4f}s")

  else:

    print("No model or image specified. Printing usage and help.")
    parse_args(["-h"])

if __name__ == '__main__':
  main()
