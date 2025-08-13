############################################################################
# --------------------------------------------------------------------------
# @author James Edmondson <james@koshee.ai>
# --------------------------------------------------------------------------
############################################################################

"""
predicts using tensort on a directory structure of images
"""

import sys
import os
import cv2
import argparse
import numpy as np
import time
import torch
from threading import Thread

import importlib

tensorrt_loader = importlib.util.find_spec('tensorrt')

if not tensorrt_loader:
  print(f"darknet2any: this script requires an installation with tensorrt")
  print(f"  to fix this issue from a local install, use scripts/install_tensorrt.sh")
  print(f"  from pip, try pip install darknet2any[tensorrt]")

  exit(0)

import pycuda.autoinit
import pycuda.driver as cuda
import tensorrt as trt

from darknet2any.tool.utils import plot_boxes_cv2, post_processing, load_class_names

class HostDeviceMem(object):
  def __init__(self, host_mem, device_mem):
    self.host = host_mem
    self.device = device_mem

  def __str__(self):
    return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

  def __repr__(self):
    return self.__str__()

class Stats:
  """
  a stats class for aggregating thread metrics
  """
  def __init__(self, load_time):
    """
    constructor
    """
    self.fps = 0
    self.embed_time = 0
    self.load_time = load_time
    self.read_time = 0
    self.predict_time = 0
    self.process_time = 0
    self.write_time = 0
    self.context_create_time = 0
    self.avg_embed_time = 0
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
    self.num_predicts += thread.num_predicts
    self.embed_time += thread.embed_time
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
    self.avg_read_time = self.read_time / self.num_predicts
    self.avg_predict_time = self.predict_time / self.num_predicts
    self.avg_process_time = self.process_time / self.num_predicts
    self.avg_write_time = self.write_time / self.num_predicts
    self.avg_context_create_time = self.context_create_time / self.num_threads
    
    print("Overall stats:")
    print(f"  predicts: {self.num_predicts}, time: {total_time:.3f}s, "
      f"fps: {self.fps:.2f}")
    print(f"  engine_create: time: {self.load_time:.3f}s")
    print(f"  context_create: time: {self.context_create_time:.3f}s, "
      f"avg: {self.avg_context_create_time:.2f}s")
    print(f"  reads: time: {self.read_time:.3f}s, avg: {self.avg_read_time:.4f}s")
    print(f"  predicts: time: {self.predict_time:.4f}s, "
      f"avg: {self.avg_predict_time:.4f}s")
    print(f"  writes: time: {self.write_time:.4f}s, "
      f"avg: {self.avg_write_time:.4f}s")
    print(f"  embedding: time: {self.embed_time:.4f}s, "
      f"avg: {self.avg_embed_time:.4f}s")

class TrtThread(Thread):
  """
  threaded tensorrt implementation
  """
  def __init__(self, idx, engine, cuda_context, classes,
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
    self.cuda_context = cuda_context
    self.device = None
    self.embed_time = 0
    self.engine = engine
    self.fps = 0
    self.idx = idx
    self.images = self.get_images(images, num_threads)
    self.inputs = list()
    self.num_predicts = 0
    self.num_threads = num_threads
    self.output_dir = output
    self.outputs = list()
    self.predict_time = 0
    self.process_time = 0
    self.read_time = 0
    self.shape = None
    self.stream = None
    self.trt_context = None
    self.write_time = 0
    self.in_run = False

  def allocate_buffers(self):
    '''
    Allocates all buffers required for a context, i.e., host/device inputs/outputs.
    Should be called in run()
    '''
    self.inputs.clear()
    self.outputs.clear()
    self.bindings.clear()
    self.stream = cuda.Stream()

    for i in range(self.engine.num_io_tensors):
      tensor_name = self.engine.get_tensor_name(i)
      size = trt.volume(self.engine.get_tensor_shape(tensor_name))
      dtype = trt.nptype(self.engine.get_tensor_dtype(tensor_name))

      # Allocate host and device buffers
      host_mem = cuda.pagelocked_empty(size, dtype)
      device_mem = cuda.mem_alloc(host_mem.nbytes)

      # Append the device buffer address to device bindings.
      # When cast to int, it's a linear index into the context's memory
      self.bindings.append(int(device_mem))

      # Append to the appropriate input/output list.
      if self.engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT:
        self.inputs.append(HostDeviceMem(host_mem, device_mem))
      else:
        self.outputs.append(HostDeviceMem(host_mem, device_mem))
        
    self.buffers = self.inputs, self.outputs, self.bindings, self.stream
    return self.buffers

  def close_contexts(self):
    """
    closes all contexts
    """
    self.inputs.clear()
    self.outputs.clear()
    self.bindings.clear()
    
    del self.stream
    del self.trt_context
    
    self.cuda_context.pop()
  
  # This function is generalized for multiple inputs/outputs.
  # inputs and outputs are expected to be lists of HostDeviceMem objects.
  def do_inference(self):
    """
    predicts classes from an engine
    """
    # Setup tensor address
    
    [cuda.memcpy_htod_async(inp.device, inp.host, self.stream) for inp in self.inputs]

    for i in range(self.engine.num_io_tensors):
      self.trt_context.set_tensor_address(self.engine.get_tensor_name(i), self.bindings[i])

    # Run inference
    self.trt_context.execute_async_v3(stream_handle=self.stream.handle)

    [cuda.memcpy_dtoh_async(out.host, out.device, self.stream) for out in self.outputs]

    # Synchronize the stream
    self.stream.synchronize()
    
    # Return only the host outputs.
    return [out.host for out in self.outputs]

  def get_contexts(self):
    """
    retrieves a thread context from an engine. Has to be called within run
    """
    #cuda.init()
    #if self.in_run:
    self.cuda_context.push()
    self.trt_context = self.engine.create_execution_context()
   
    shape = None

    for i in range(self.engine.num_io_tensors):
      tensor_name = self.engine.get_tensor_name(i)
      print(f"  tensor_name[{i}] = {tensor_name}")
      print(f"  tensor_name[{i}].shape = {get_binding_shape(self.engine,tensor_name)}")

    shape = get_binding_shape(self.engine, "input")

    if shape is not None:

      self.shape = (
        shape[3],
        shape[2]
      )
   
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
  
  def predict(self, image_file):
    """
    predicts classes of an image file

    Args:
    engine (tensorrt.ICudaEngine): the trt engine
    context (tensorrt.IExecutionContext]): the trt context for this thread
    buffers (tuple): result of allocate_buffers call
    shape (str): the shape of the engine input (what we're resizing to)
    classes (list): a list of class names (strings)
    output (str): directory to store labeled images to (None means don't save)
    image_file (str): path to an image
    Returns:
    tuple: read_time, predict_time, process_time, write_time
    """
    print(f"trt: Reading {image_file}")

    basename = os.path.basename(image_file)
    start = time.perf_counter()

    img = cv2.imread(image_file)
    resized = cv2.resize(img, self.shape, interpolation=cv2.INTER_LINEAR)
    img_in = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    img_in = np.transpose(img_in, (2, 0, 1)).astype(np.float32)
    img_in = np.expand_dims(img_in, axis=0)
    img_in /= 255.0
    img_in = np.ascontiguousarray(img_in)
    
    end = time.perf_counter()
    read_time = end - start

    start = time.perf_counter()
    # Allocate buffers

    # print('Length of inputs: ', len(inputs))
    self.inputs[0].host = img_in

    trt_outputs = self.do_inference()

    trt_outputs[0] = trt_outputs[0].reshape(1, -1, 1, 4)
    trt_outputs[1] = trt_outputs[1].reshape(1, -1, len(self.classes))
    embeddings = None
    if len(trt_outputs) > 2:
      embeddings = trt_outputs[2]
    embedding_path = f"{self.output_dir}/{basename}.features"

    end = time.perf_counter()
    predict_time = end - start

    start = time.perf_counter()
    boxes = post_processing(img, 0.4, 0.6, trt_outputs)
    end = time.perf_counter()
    process_time = end - start

    start = time.perf_counter()
    if self.output_dir is not None:
      torch.save(embeddings, embedding_path)
    end = time.perf_counter()
    embed_time = end - start
    
    start = time.perf_counter()
    if self.output_dir is not None:
      plot_boxes_cv2(img, boxes[0],
        savename=f"{self.output_dir}/{basename}", class_names=self.classes)
    write_time = time.perf_counter() - start

    print(f"trt: predict for {image_file}")
    print(f"  output: {boxes}")
    print(f"  read_time: {read_time:.4f}s")
    print(f"  predict_time: {predict_time:.4f}s")
    print(f"  post_processing: {process_time:.4f}s")
    print(f"  write_time: {write_time:.4f}s")
    print(f"  embed_time: {embed_time:.4f}s")

    return read_time, predict_time, process_time, write_time, embed_time

  def run(self):
    """
    executes thread logic
    """
    
    self.in_run = True
    if len(self.images) > 0:
      
      start = time.perf_counter()
      self.get_contexts()
      
      self.allocate_buffers()
      
      self.context_create_time = time.perf_counter() - start
      
      start = time.perf_counter()
      for image in self.images:

        print(f"run: predicting on {image}")
        results = self.predict(image)
        read_time, predict_time, process_time, write_time, embed_time = results
        
        self.read_time += read_time
        self.predict_time += predict_time
        self.process_time += process_time
        self.write_time += write_time
        self.embed_time += embed_time
        self.num_predicts += 1
        
      total_time = time.perf_counter() - start
      self.fps = self.num_predicts / total_time
          
      self.avg_embed_time = self.embed_time / self.num_predicts
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
      print(f"  embed_time: {self.embed_time:.4f}s, avg: {self.avg_embed_time:.4f}s")
      
      self.close_contexts()
    else:
      print(f"thread.{self.idx}: no images, so no predicts")
      
    self.in_run = False

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
  description="predicts from a trt model",
  add_help=True
  )
  parser.add_argument('-i','--input','--trt', action='store',
    dest='input', default=None,
    help='the engine/trt to load')
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

def load_engine(path):
  """
  loads the binary trt engine file
  """
  TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
  runtime = trt.Runtime(TRT_LOGGER)
  engine = None

  with open(path, "rb") as f:
    engine = runtime.deserialize_cuda_engine(f.read())
  
  return engine

def get_binding_shape(engine, name):
  """
  gets the shape of the image
  """
  return list(engine.get_tensor_shape(name))

def get_binding_dtype(engine, name):
  """
  gets the binding for the data type
  """
  return trt.nptype(engine.get_tensor_dtype(name))

def main():
  """
  main script entry point
  """

  options = parse_args(sys.argv[1:])
  has_images = options.image is not None or options.image_dir is not None
  
  if options.no_output:
    print(f"trt: not saving labeled files")
    options.output = None
  else:
    print(f"trt: saving labeled files to {options.output}")

  print(f"trt: predicting with {options.input}")

  basename, ext = os.path.splitext(options.input)
  if ext == "":
    options.input += ".trt"

  if options.input is not None and has_images:

    if not os.path.isfile(options.input):
      print(f"predict_trt: trt file cannot be read. "
        "check file exists or permissions.")
      exit(1)

    print(f"trt: loading {options.input}")

    basename = os.path.splitext(options.input)[0]

    if options.output is not None and not os.path.isdir(options.output):
      os.makedirs(options.output)

    names_file = f"{basename}.names"
    classes = load_class_names(names_file)
    
    cuda.init()
    
    cuda_context = pycuda.autoinit.context

    # 1. Load the trt model
    start = time.perf_counter()

    engine = load_engine(options.input)
    
    end = time.perf_counter()
    load_time = end - start
    print(f"  load_time: {load_time:.4f}s")

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
    
    worker_threads = list()
    stats = Stats(load_time)
    
    for i in range(options.threads):
      print(f"Creating thread.{i}")
      thread = TrtThread(i, engine, cuda_context, classes, options.output,
        options.threads, images)
      worker_threads.append(thread)
    
    work_start = time.perf_counter()
    
    for worker in worker_threads:
      worker.start()
      
    for worker in worker_threads:
      worker.join()
      stats.add(worker)
      
    work_time = time.perf_counter() - work_start
    
    stats.print(work_time)
    
    del engine

  else:

    print("No model or image specified. Printing usage and help.")
    parse_args(["-h"])

if __name__ == '__main__':
  main()
