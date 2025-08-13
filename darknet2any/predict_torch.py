############################################################################
# --------------------------------------------------------------------------
# @author James Edmondson <james@koshee.ai>
# --------------------------------------------------------------------------
############################################################################

"""
predicts using torch on a directory structure of images
"""

import sys
import os
import cv2
import argparse
import numpy as np
import time
import torch

from darknet2any.tool.torch_utils import do_detect
from darknet2any.tool.utils import *
from darknet2any.tool.darknet2pytorch import Darknet

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
  description="predicts from a weights-based torch model",
  add_help=True
  )
  parser.add_argument('-i','--input','--torch', action='store',
    dest='input', default=None,
    help='the engine/torch to load')
  parser.add_argument('--image', action='store',
    dest='image', default=None,
    help='the image to test the model on')
  parser.add_argument('--image-dir', action='store',
    dest='image_dir', default=None,
    help='a directory of images to test')
  parser.add_argument('--ie','--image-embeddings','--include-embeddings',
    action='store_true', dest='include_embeddings',
    help='add image feature embeddings to model output')
  parser.add_argument('-o','--output-dir', action='store',
    dest='output', default="labeled_images",
    help='a directory to place labeled images')
  parser.add_argument('-c','--cuda', action='store',
    dest='cuda', default=True, type=bool,
    help='a directory to place labeled images')
  # parser.add_argument('-t','--threads', action='store',
  #   dest='threads', type=int, default=1,
  #   help='the number of threads to run')
  

  return parser.parse_args(args)

def load_model(path, use_cuda, include_embeddings):
  """
  loads a torch model
  """

  base = os.path.splitext(path)[0]
  cfg_file = f"{base}.cfg"
  model_file = f"{base}.weights"
  names_file = f"{base}.names"

  model = Darknet(cfg_file, include_embeddings=include_embeddings)
  model.load_weights(model_file)
  if use_cuda:
    model.cuda()

  names = open(
    names_file, encoding="utf-8").read().splitlines()
  
  return model, names

def torch_image_predict(
  model, shape, classes, output, image_file, use_cuda):
  """
  predicts classes of an image file

  Args:
  interpreter (tf.lite.Interpreter): the torch interpreter for a model
  input_details (list[dict[string, Any]]): result of get_input_details()
  output_details (list[dict[string, Any]]): result of get_output_details()
  image_file (str): an image file to read and predict on
  Returns:
  tuple: read_time, predict_time
  """
  print(f"torch: Reading {image_file}")

  start = time.perf_counter()

  image = cv2.imread(image_file)
  basename = os.path.basename(image_file)
  embedding_path = f"{output}/{basename}.features"

  print(f"image: shape={image.shape}")

  # use OpenCV to load the image and swap OpenCV's
# usual BGR for the RGB that torch requires
  sized = cv2.resize(image, shape)
  sized = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB)

  end = time.perf_counter()
  read_time = end - start


  start = time.perf_counter()

  boxes = do_detect(model, sized, 0.4, 0.6, use_cuda,
    embedding_path=embedding_path)

  end = time.perf_counter()
  predict_time = end - start

  start = time.perf_counter()
  
  end = time.perf_counter()
  process_time = end - start

  plot_boxes_cv2(image, boxes[0],
    savename=f"{output}/{basename}", class_names=classes)

  print(f"torch: predict for {image_file}")
  print(f"  output: {boxes}")
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

  print(f"torch: predicting with {options.input}")

  if options.input is not None and has_images:
    print(f"torch: loading {options.input}")

    if not os.path.isfile(options.input):
      print(f"predict_torch: torch weights cannot be read. "
        "check file exists or permissions.")
      exit(1)

    basename = os.path.splitext(options.input)[0]

    if not os.path.isdir(options.output):
      os.makedirs(options.output)

    names_file = f"{basename}.names"
    names = load_class_names(names_file)

    # 1. Load the torch model
    start = time.perf_counter()

    model, names = load_model(options.input, options.cuda, options.include_embeddings)
    # model = torch.jit.load(options.input)
    # model.eval()
    
    #print(f"jit_model={traced_model}")
    
    end = time.perf_counter()
    load_time = end - start
    print(f"  load_time: {load_time:.4f}s")
    
    shape = None

    if model is not None:
        
      shape = (model.width, model.height)

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
          read_time, predict_time, process_time = torch_image_predict(
            model, shape, names, options.output, image, options.cuda)
          
          total_read_time += read_time
          total_predict_time += predict_time
          total_process_time += process_time

        avg_read_time = total_read_time / num_predicts
        avg_predict_time = total_predict_time / num_predicts
        avg_process_time = total_process_time / num_predicts

        print(f"torch: time for {num_predicts} predicts")
        print(f"  model_load_time: total: {load_time:.4f}s")
        print(f"  image_read_time: total: {total_read_time:.4f}s, avg: {avg_read_time:.4f}s")
        print(f"  predict_time: {total_predict_time:.4f}s, avg: {avg_predict_time:.4f}s")
        print(f"  process_time: {total_process_time:.4f}s, avg: {avg_process_time:.4f}s")

  else:

    print("No model or image specified. Printing usage and help.")
    parse_args(["-h"])
    
if __name__ == '__main__':
  main()
