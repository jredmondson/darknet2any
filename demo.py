# -*- coding: utf-8 -*-
'''
@Time      : 20/04/25 15:49
@Author    : huguanghao
@File      : demo.py
@Noice     :
@Modificattion :
  @Author  :
  @Time    :
  @Detail  :
'''

# import sys
# import time
# from PIL import Image, ImageDraw
# from models.tiny_yolo import TinyYoloNet
from tool.utils import *
from tool.torch_utils import *
from tool.darknet2pytorch import Darknet
import torch
import argparse
import os
import cv2

def detect_cv2(options):
  m = Darknet(options.config_file)

  m.print_network()
  m.load_weights(options.model)

  print(f'Loading weights from {options.model}... Done!')

  if options.use_cuda:
    m.cuda()

  class_names = load_class_names(options.names_file)

  img = cv2.imread(options.image)
  sized = cv2.resize(img, (m.width, m.height))
  sized = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB)

  for i in range(2):
    start = time.time()
    boxes = do_detect(m, sized, 0.4, 0.6, options.use_cuda)
    finish = time.time()
    if i == 1:
      print('%s: Predicted in %f seconds.' % (options.image, (finish - start)))

  plot_boxes_cv2(img, boxes[0], savename='predictions.jpg', class_names=class_names)


def detect_cv2_camera(options):
  import cv2
  m = Darknet(options.config_file)

  m.print_network()
  if options.torch:
    m.load_state_dict(torch.load(options.model))
  else:
    m.load_weights(options.model)
  print('Loading weights from %s... Done!' % (options.model))

  if options.use_cuda:
    m.cuda()

  cap = cv2.VideoCapture(0)

  cap.set(3, 1280)
  cap.set(4, 720)

  print("Starting the YOLO loop...")

  options.names_file = os.path.splitext(options.model)[0] + ".names"

  class_names = load_class_names(options.names_file)

  while True:
    ret, img = cap.read()
    sized = cv2.resize(img, (m.width, m.height))
    sized = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB)

    start = time.time()
    boxes = do_detect(m, sized, 0.4, 0.6, options.use_cuda)
    finish = time.time()
    print('Predicted in %f seconds.' % (finish - start))

    result_img = plot_boxes_cv2(img, boxes[0], savename=None, class_names=class_names)

    cv2.imshow('Yolo demo', result_img)
    cv2.waitKey(1)

  cap.release()


def detect_skimage(options):
  from skimage import io
  from skimage.transform import resize
  m = Darknet(options.config_file)

  m.print_network()
  m.load_weights(options.model)
  print('Loading weights from %s... Done!' % (options.model))

  if options.use_cuda:
    m.cuda()

  num_classes = m.num_classes
  if num_classes == 20:
    options.names_file = 'data/voc.names'
  elif num_classes == 80:
    options.names_file = 'data/coco.names'
  else:
    options.names_file = 'data/x.names'
  class_names = load_class_names(options.names_file)

  img = io.imread(options.image)
  sized = resize(img, (m.width, m.height)) * 255

  for i in range(2):
    start = time.time()
    boxes = do_detect(m, sized, 0.4, 0.4, options.use_cuda)
    finish = time.time()
    if i == 1:
      print('%s: Predicted in %f seconds.' % (options.image, (finish - start)))

  plot_boxes_cv2(img, boxes, savename='predictions.jpg', class_names=class_names)


def parse_args(args):
  parser = argparse.ArgumentParser(
    description="Runs inference with pytorch rather than yolov4 weights",
    add_help=True
  )
  parser.add_argument('--cf', '--config-file', action='store',
            help='path of cfg file', dest='config_file')
  parser.add_argument('--nf', '--names-file', action='store',
            help='path of cfg file', dest='names_file')
  parser.add_argument('-m', '--model', action='store',
            help='path of trained model.', dest='model')
  parser.add_argument('-i', '--image', action='store',
            default='./data/mscoco2017/train2017/190109_180343_00154162.jpg',
            help='path of your image file.', dest='image')
  parser.add_argument('--uc', '--use-cuda',
            action='store_true',
            help='if true, use CUDA. Otherwise, CPU.', dest='use_cuda')
  
  return parser.parse_args(args)


if __name__ == '__main__':
  options = parse_args(sys.argv[1:])

  if options.model:
    
    basename = os.path.splitext(options.model)[0]

    if not options.names_file:
      options.names_file = f"{basename}.names"
    
    if not options.config_file:
      options.config_file = f"{basename}.cfg"

  if options.image:
    detect_cv2(options)
  else:
    detect_cv2_camera(options)


