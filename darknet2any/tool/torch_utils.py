import sys
import os
import time
import math
import torch
import torch.nn as nn
import cv2
import numpy as np
from torch.autograd import Variable

import itertools
import struct  # get_image_size
import imghdr  # get_image_size

from darknet2any.tool import utils 


def bbox_ious(boxes1, boxes2, x1y1x2y2=True):
  if x1y1x2y2:
    mx = torch.min(boxes1[0], boxes2[0])
    Mx = torch.max(boxes1[2], boxes2[2])
    my = torch.min(boxes1[1], boxes2[1])
    My = torch.max(boxes1[3], boxes2[3])
    w1 = boxes1[2] - boxes1[0]
    h1 = boxes1[3] - boxes1[1]
    w2 = boxes2[2] - boxes2[0]
    h2 = boxes2[3] - boxes2[1]
  else:
    mx = torch.min(boxes1[0] - boxes1[2] / 2.0, boxes2[0] - boxes2[2] / 2.0)
    Mx = torch.max(boxes1[0] + boxes1[2] / 2.0, boxes2[0] + boxes2[2] / 2.0)
    my = torch.min(boxes1[1] - boxes1[3] / 2.0, boxes2[1] - boxes2[3] / 2.0)
    My = torch.max(boxes1[1] + boxes1[3] / 2.0, boxes2[1] + boxes2[3] / 2.0)
    w1 = boxes1[2]
    h1 = boxes1[3]
    w2 = boxes2[2]
    h2 = boxes2[3]
    uw = Mx - mx
    uh = My - my
    cw = w1 + w2 - uw
    ch = h1 + h2 - uh
    mask = ((cw <= 0) + (ch <= 0) > 0)
    area1 = w1 * h1
    area2 = w2 * h2
    carea = cw * ch
    carea[mask] = 0
    uarea = area1 + area2 - carea
  return carea / uarea


def get_region_boxes(boxes_and_confs, include_embeddings=False):

  # print('Getting boxes from boxes and confs ...')

  boxes_list = []
  confs_list = []
  features_list = []

  #print(f"get_region_boxes:boxes_and_confs:size={len(boxes_and_confs)}")
  
  for item in boxes_and_confs:
    boxes_list.append(item[0])
    confs_list.append(item[1])
    if include_embeddings:
      features_list.append(item[2])

  # boxes: [batch, num1 + num2 + num3, 1, 4]
  # confs: [batch, num1 + num2 + num3, num_classes]
  boxes = torch.cat(boxes_list, dim=1)
  confs = torch.cat(confs_list, dim=1)
  
  #print(f"get_region_boxes:features_list:shape={features_list[0].shape}")
  
  result = [boxes, confs]
  
  if include_embeddings:
    result.append(features_list[0])
  
  return result


def convert2cpu(gpu_matrix):
  return torch.FloatTensor(gpu_matrix.size()).copy_(gpu_matrix)


def convert2cpu_long(gpu_matrix):
  return torch.LongTensor(gpu_matrix.size()).copy_(gpu_matrix)



def do_detect(model, img, conf_thresh, nms_thresh, use_cuda=1, embedding_path=None, debug=False):
  model.eval()
  with torch.no_grad():
    img_prep_start = time.perf_counter()

    if type(img) == np.ndarray and len(img.shape) == 3:  # cv2 image
      img = torch.from_numpy(img.transpose(2, 0, 1)).float().div(255.0).unsqueeze(0)
    elif type(img) == np.ndarray and len(img.shape) == 4:
      img = torch.from_numpy(img.transpose(0, 3, 1, 2)).float().div(255.0)
    else:
      print("unknow image type")
      exit(-1)

    if use_cuda:
      img = img.cuda()

    img = torch.autograd.Variable(img)

    predict_start = time.perf_counter()

    output = model(img)

    embed_start = time.perf_counter()

    embedding = None
    if len(output) > 2:
      embedding = output[2]

    embed_end = time.perf_counter()

    if embedding_path is not None:
      torch.save(embedding, embedding_path)

    save_embedding_end = time.perf_counter()

    preprocess_time = predict_start - img_prep_start
    predict_time = embed_start - predict_start
    embed_time = embed_end - embed_start
    save_embedding_time = save_embedding_end - embed_end

    if debug:
      print(f"torch: embedding for img is shaped {embedding.shape}")

      print(f"torch: time for do_detect")
      print(f"  preprocess_time: {preprocess_time:.4f}s")
      print(f"  predict_time: {predict_time:.4f}s")
      print(f"  embed_time: {embed_time:.5f}s")
      print(f"  save_embedding_time: {save_embedding_time:.4f}s")

    return utils.post_processing(img, conf_thresh, nms_thresh, output)
