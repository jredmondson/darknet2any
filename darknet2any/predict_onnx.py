############################################################################
# --------------------------------------------------------------------------
# @author James Edmondson <james@koshee.ai>
# --------------------------------------------------------------------------
############################################################################

"""
predicts using onnx on a directory structure of images
"""

import argparse
import importlib
import numpy as np
import os
import sys
import time
from threading import Thread

import cv2

from darknet2any.tool.utils import *
from darknet2any.tool.darknet2onnx import *

ort_loader = importlib.util.find_spec('onnxruntime')

if not ort_loader:
  print(f"darknet2any: this script requires an installation with onnxruntime")
  print(f"  try pip install darknet2any[tensorrt] or [rocm] options")

  exit(0)

import onnxruntime as ort

available_providers = ort.get_available_providers()

print(f"onnx executor options: {available_providers}")

class Stats:
  """
  a stats class for aggregating thread metrics
  """
  def __init__(self):
    """
    constructor
    """
    self.embed_time = 0
    self.fps = 0
    self.load_time = 0
    self.read_time = 0
    self.predict_time = 0
    self.process_time = 0
    self.write_time = 0
    self.context_create_time = 0
    self.avg_embed_time = 0
    self.avg_load_time = 0
    self.avg_read_time = 0
    self.avg_predict_time = 0
    self.avg_process_time = 0
    self.avg_write_time = 0
    self.avg_context_create_time = 0
    self.num_predicts = 0
    self.num_threads = 0
  
  def add(self, thread):
    """
    adds thread stats to the aggregated stats
    """
    self.embed_time += thread.embed_time
    self.load_time += thread.load_time
    self.num_predicts += thread.num_predicts
    self.read_time += thread.read_time
    self.predict_time += thread.predict_time
    self.process_time += thread.process_time
    self.write_time += thread.write_time
    self.context_create_time += thread.context_create_time
    self.num_threads += 1

  def print(self, total_time):
    """
    prints the aggregated stats
    """
    self.fps = self.num_predicts / total_time
    self.avg_embed_time = self.embed_time / self.num_predicts
    self.avg_load_time = self.load_time / self.num_predicts
    self.avg_read_time = self.read_time / self.num_predicts
    self.avg_predict_time = self.predict_time / self.num_predicts
    self.avg_process_time = self.process_time / self.num_predicts
    self.avg_write_time = self.write_time / self.num_predicts
    self.avg_context_create_time = self.context_create_time / self.num_threads
    
    print("Overall stats:")
    print(f"  predicts: {self.num_predicts}, time: {total_time:.3f}s, "
      f"fps: {self.fps:.2f}")
    print(f"  load_time: time: {self.load_time:.3f}s, avg: {self.avg_load_time:.4f}s")
    print(f"  reads: time: {self.read_time:.3f}s, avg: {self.avg_read_time:.4f}s")
    print(f"  predicts: time: {self.predict_time:.4f}s, "
      f"avg: {self.avg_predict_time:.4f}s")
    print(f"  writes: time: {self.write_time:.4f}s, "
      f"avg: {self.avg_write_time:.4f}s")
    print(f"  embed: time: {self.embed_time:.4f}s, "
      f"avg: {self.avg_embed_time:.4f}s")

class OnnxThread(Thread):
  """
  threaded onnxruntime implementation
  """
  def __init__(self, idx, model_path, providers, classes,
    output, num_threads, images):
    """
    constructor
    """
    
    Thread.__init__(self)
    
    self.avg_embed_time = 0
    self.avg_predict_time = 0
    self.avg_process_time = 0
    self.avg_read_time = 0
    self.avg_write_time = 0
    self.bindings = list()
    self.buffers = None
    self.classes = classes
    self.context_create_time = 0
    self.embed_time = 0
    self.model_path = model_path
    self.device = None
    self.session = None
    self.fps = 0
    self.idx = idx
    self.images = self.get_images(images, num_threads)
    self.inputs = list()
    self.load_time = 0
    self.num_predicts = 0
    self.num_threads = num_threads
    self.output_dir = output
    self.outputs = list()
    self.predict_time = 0
    self.providers = providers
    self.process_time = 0
    self.read_time = 0
    self.shape = None
    self.stream = None
    self.trt_context = None
    self.write_time = 0
    self.in_run = False

  def get_images(self, images, num_threads):
    '''
    Allocates an images list based on num threads and length of images
    '''
    segment = int(len(images) / num_threads)
    start = int(segment * self.idx)
    end = int(segment) * (self.idx + 1)
    
    results = None
    
    if self.idx != num_threads - 1:
      results = images[start:end]
    else:
      results = images[start:]
    
    #print(f"get_images: images={results}")
    
    return results
  
  def load(self):
    
      print(f"predict_onnx.{self.idx}: using providers={self.providers}")
      print(f"predict_onnx.{self.idx}: loading model={self.model_path}")

      # 1. Load the onnx model
      start = time.perf_counter()
      self.session = ort.InferenceSession(self.model_path,
        providers=self.providers)
      end = time.perf_counter()
      self.load_time = end - start
      print(f"  load_time: {self.load_time:.4f}s")
      
      print(f"  provider options: {self.session.get_provider_options()}")
      
      self.shape = None

      for input_meta in self.session.get_inputs():
        if input_meta.name == "input":
          self.shape = (
            input_meta.shape[3],
            input_meta.shape[2]
          )
          break


  def predict(self, image_file):
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
    self.num_predicts += 1
    basename = os.path.basename(image_file)

    start = time.perf_counter()
    img = cv2.imread(image_file)
    image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, self.shape, interpolation=cv2.INTER_LINEAR)
    image_resized = np.transpose(image_resized, (2, 0, 1)).astype(np.float32)
    image = np.expand_dims(image_resized, axis=0)
    image /= 255.0
    
    end = time.perf_counter()
    read_time = end - start
    self.read_time += read_time

    start = time.perf_counter()
    input_name = self.session.get_inputs()[0].name
    outputs = self.session.run(None, {input_name: image})

    outputs[0] = outputs[0].reshape(1, -1, 1, 4)
    outputs[1] = outputs[1].reshape(1, -1, len(self.classes))
    embeddings = None
    if len(outputs) > 2:
      embeddings = outputs[2]
      
    embedding_path = f"{self.output_dir}/{basename}.features"

    end = time.perf_counter()
    predict_time = end - start
    self.predict_time += predict_time

    start = time.perf_counter()
    boxes = post_processing(image, 0.4, 0.6, outputs)
    end = time.perf_counter()
    process_time = end - start
    self.process_time += process_time
    
    start = time.perf_counter()
    if self.output_dir is not None:
      torch.save(embeddings, embedding_path)
    end = time.perf_counter()
    embed_time = end - start
    self.embed_time += embed_time

    start = time.perf_counter()
    if self.output_dir is not None:
      plot_boxes_cv2(img, boxes[0],
        savename=f"{self.output_dir}/{basename}", class_names=self.classes)
    write_time = time.perf_counter() - start
    self.write_time += write_time

    print(f"onnx: predict for {image_file}")
    #print(f"  output: {outputs}")
    print(f"  read_time: {read_time:.4f}s")
    print(f"  predict_time: {predict_time:.4f}s")
    print(f"  post_processing: {process_time:.4f}s")
    print(f"  write_time: {write_time:.4f}s")
    print(f"  embed_time: {embed_time:.4f}s")

  def run(self):
    """
    executes thread logic
    """
    self.load()
    
    start = time.perf_counter()
    for image in self.images:

      print(f"run: predicting on {image}")
      self.predict(image)
      
    total_time = time.perf_counter() - start
    self.fps = self.num_predicts / total_time
        
    self.avg_read_time = self.read_time / self.num_predicts
    self.avg_predict_time = self.predict_time / self.num_predicts
    self.avg_process_time = self.process_time / self.num_predicts
    self.avg_write_time = self.write_time / self.num_predicts

    print(f"thread.{self.idx}: time for {self.num_predicts} predicts")
    print(f"  total_time: {total_time:.4f}s, fps={self.fps:.2f}")
    print(f"  context_create: total: {self.context_create_time:.4f}s")
    print(f"  image_read_time: total: {self.read_time:.4f}s, avg: {self.avg_read_time:.4f}s")
    print(f"  predict_time: {self.predict_time:.4f}s, avg: {self.avg_predict_time:.4f}s")
    print(f"  process_time: {self.process_time:.4f}s, avg: {self.avg_process_time:.4f}s")
    print(f"  write_time: {self.write_time:.4f}s, avg: {self.avg_write_time:.4f}s")
      
    
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
  parser.add_argument('-p','--provider', '--force-provider', action='store',
    dest='provider',
    help='sets a cpu-only inference mode')
  parser.add_argument('-i','--input','--onnx', action='store',
    dest='input', default=None,
    help='the onnx model to load')
  parser.add_argument('--image', action='store',
    dest='image', default=None,
    help='the image to test the model on')
  parser.add_argument('--image-dir', action='store',
    dest='image_dir', default=None,
    help='a directory of images to test')
  parser.add_argument('--no','--no-output', action='store_true',
    dest='no_output', default=False, 
    help='do not write labeled files to disk (performance test)')
  parser.add_argument('-o','--output-dir', action='store',
    dest='output', default="labeled_images",
    help='a directory to place labeled images')
  parser.add_argument('-t','--threads', action='store',
    dest='threads', default=1, type=int,
    help='the number of threads to run')
  

  return parser.parse_args(args)

def main():
  """
  main script entry point
  """

  options = parse_args(sys.argv[1:])
  has_images = options.image is not None or options.image_dir is not None

  if options.no_output:
    print(f"onnx: not saving labeled files")
    options.output = None
  else:
    print(f"onnx: saving labeled files to {options.output}")

  if options.input is not None and has_images:
    print(f"onnx: loading {options.input}")

    if not os.path.isfile(options.input):
      print(f"predict_onnx: onnx file cannot be read. "
        "check file exists or permissions.")
      exit(1)

    basename = os.path.splitext(options.input)[0]
    names_file = f"{basename}.names"

    if options.output is not None and not os.path.isdir(options.output):
      os.makedirs(options.output)

    providers = []

    if options.provider is not None:
      providers.append(options.provider)
      providers.append('CPUExecutionProvider')
    elif not options.cpu:
      if 'CUDAExecutionProvider' in available_providers:
        providers.extend(['CUDAExecutionProvider', 'CPUExecutionProvider'])
      else:                   
        providers.extend(available_providers)
    else:
      providers.append('CPUExecutionProvider')

    if providers[0] == "ROCMExecutionProvider":
      os.environ["HIP_VISIBLE_DEVICES"]="0"


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

    if len(images) > 0:
      worker_threads = list()
      stats = Stats()
      
      for i in range(options.threads):
        print(f"Creating thread.{i}")
        thread = OnnxThread(i, options.input, providers, classes,
          options.output, options.threads, images)
        worker_threads.append(thread)
      
      work_start = time.perf_counter()
      
      for worker in worker_threads:
        worker.start()
        
      for worker in worker_threads:
        worker.join()
        stats.add(worker)
        
      work_time = time.perf_counter() - work_start
      
      stats.print(work_time)

  else:

    print("No model or image specified. Printing usage and help.")
    parse_args(["-h"])

if __name__ == '__main__':
  main()
