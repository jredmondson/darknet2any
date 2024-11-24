import sys
import onnx
import os
import argparse
import numpy as np
import cv2
import onnxruntime

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
  parser.add_argument('-i','--input','--weights', action='store',
    dest='input', default=None,
    help='the weights file to convert')
  parser.add_argument('--image','--test', action='store',
    dest='image', default=None,
    help='the image to test the resulting onnx model on')
  parser.add_argument('-o','--output','--onnx',
    action='store', dest='output', default=None,
    help='the onnx file to create (default=filename.onnx)')

  return parser.parse_args(args)

def main(cfg_file, namesfile, weight_file, image_path, batch_size, output_name):

  if batch_size <= 0:
    onnx_path_demo = transform_to_onnx(cfg_file, weight_file, batch_size, output_name)
  else:
    # Transform to onnx as specified batch size
    transform_to_onnx(cfg_file, weight_file, batch_size)
    # Transform to onnx as demo
    onnx_path_demo = transform_to_onnx(cfg_file, weight_file, 1, output_name)

  session = onnxruntime.InferenceSession(onnx_path_demo)
  # session = onnx.load(onnx_path)
  print("The model expects input shape: ", session.get_inputs()[0].shape)

  if image_path is not None:
    image_src = cv2.imread(image_path)
    detect(session, image_src, namesfile)

def nms_cpu(boxes, confs, nms_thresh=0.5, min_mode=False):
  # print(boxes.shape)
  x1 = boxes[:, 0]
  y1 = boxes[:, 1]
  x2 = boxes[:, 2]
  y2 = boxes[:, 3]

  areas = (x2 - x1) * (y2 - y1)
  order = confs.argsort()[::-1]

  keep = []
  while order.size > 0:
    idx_self = order[0]
    idx_other = order[1:]

    keep.append(idx_self)

    xx1 = np.maximum(x1[idx_self], x1[idx_other])
    yy1 = np.maximum(y1[idx_self], y1[idx_other])
    xx2 = np.minimum(x2[idx_self], x2[idx_other])
    yy2 = np.minimum(y2[idx_self], y2[idx_other])

    w = np.maximum(0.0, xx2 - xx1)
    h = np.maximum(0.0, yy2 - yy1)
    inter = w * h

    if min_mode:
      over = inter / np.minimum(areas[order[0]], areas[order[1:]])
    else:
      over = inter / (areas[order[0]] + areas[order[1:]] - inter)

    inds = np.where(over <= nms_thresh)[0]
    order = order[inds + 1]

  return np.array(keep)


def detect(session, image_src, namesfile):
  IN_IMAGE_H = session.get_inputs()[0].shape[2]
  IN_IMAGE_W = session.get_inputs()[0].shape[3]

  # Input
  start = time.time()
  resized = cv2.resize(image_src, (IN_IMAGE_W, IN_IMAGE_H),
    interpolation=cv2.INTER_LINEAR)
  end = time.time()
  resize_time = end - start

  start = time.time()
  img_in = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
  img_in = np.transpose(img_in, (2, 0, 1)).astype(np.float32)
  img_in = np.expand_dims(img_in, axis=0)
  img_in /= 255.0
  print("Shape of the network input: ", img_in.shape)

  end = time.time()
  reshape_time = end - start

  # Compute
  input_name = session.get_inputs()[0].name

  start = time.time()
  outputs = session.run(None, {input_name: img_in})
  end = time.time()
  detect_time = end - start

  start = time.time()
  boxes = post_processing(img_in, 0.4, 0.6, outputs)
  end = time.time()
  nms_time = end - start

  total_time = resize_time + reshape_time + detect_time + nms_time

  class_names = load_class_names(namesfile)

  print(f"Attempting to plot the boxes in cv2")
  plot_boxes_cv2(image_src, boxes[0], savename='predictions_onnx.jpg', class_names=class_names)
  print(f"Done with plotting")
  print(f"Detect Timing:")

  print(f"  resize={resize_time:.4f}")
  print(f"  reshape={reshape_time:.4f}")
  print(f"  detect={detect_time:.4f}")
  print(f"  nms={nms_time:.4f}")
  print(f"  total={total_time:.4f}")



if __name__ == '__main__':
  options = parse_args(sys.argv[1:])

  if options.input is not None:
    prefix = os.path.splitext(options.input)[0]

    cfg_file = f"{prefix}.cfg"
    names_file = f"{prefix}.names"
    weight_file = options.input
    image_path = options.image
    batch_size = 1
    output_file = f"{prefix}.onnx"

    if options.output is not None:
      output_file = options.output

    main(cfg_file, names_file, weight_file, image_path, batch_size, output_file)
  else:
    parse_args(["-h"])
