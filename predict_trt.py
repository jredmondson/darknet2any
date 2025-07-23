import sys
import os
import cv2
import argparse
import numpy as np
import time

import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

from tool.utils import *

class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()

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
  description="predicts from an trt model",
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
  parser.add_argument('-o','--output-dir', action='store',
    dest='output', default="labeled_images",
    help='a directory to place labeled images')
  # parser.add_argument('-t','--threads', action='store_int',
  #   dest='threads', default=1,
  #   help='the number of threads to run')
  

  return parser.parse_args(args)

def allocate_buffers(engine):
  '''
  Allocates all buffers required for an engine, i.e., host/device inputs/outputs.
  '''
  inputs = []
  outputs = []
  bindings = []
  stream = cuda.Stream()

  for i in range(engine.num_io_tensors):
    tensor_name = engine.get_tensor_name(i)
    size = trt.volume(engine.get_tensor_shape(tensor_name))
    dtype = trt.nptype(engine.get_tensor_dtype(tensor_name))

    # Allocate host and device buffers
    host_mem = cuda.pagelocked_empty(size, dtype) # page-locked memory buffer (won't swap to disk)
    device_mem = cuda.mem_alloc(host_mem.nbytes)

    # Append the device buffer address to device bindings.
    # When cast to int, it's a linear index into the context's memory (like memory address).
    bindings.append(int(device_mem))

    # Append to the appropriate input/output list.
    if engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT:
      inputs.append(HostDeviceMem(host_mem, device_mem))
    else:
      outputs.append(HostDeviceMem(host_mem, device_mem))

  return inputs, outputs, bindings, stream

def load_engine(path):
  TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
  runtime = trt.Runtime(TRT_LOGGER)
  engine = None
  context = None

  with open(path, "rb") as f:
    engine = runtime.deserialize_cuda_engine(f.read())
    context = engine.create_execution_context()
  
  return engine, context

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

# This function is generalized for multiple inputs/outputs.
# inputs and outputs are expected to be lists of HostDeviceMem objects.
def do_inference(engine, context, bindings, inputs, outputs, stream):
  # Setup tensor address
  
  [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]

  for i in range(engine.num_io_tensors):
      context.set_tensor_address(engine.get_tensor_name(i), bindings[i])

  # Run inference
  context.execute_async_v3(stream_handle=stream.handle)

  [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]

  # Synchronize the stream
  stream.synchronize()

  # Return only the host outputs.
  return [out.host for out in outputs]

def trt_image_predict(
  engine, context, buffers, shape, classes, output, image_file):
  """
  predicts classes of an image file

  Args:
  interpreter (tf.lite.Interpreter): the trt interpreter for a model
  input_details (list[dict[string, Any]]): result of get_input_details()
  output_details (list[dict[string, Any]]): result of get_output_details()
  image_file (str): an image file to read and predict on
  Returns:
  tuple: read_time, predict_time
  """
  print(f"trt: Reading {image_file}")

  start = time.perf_counter()

  img = cv2.imread(image_file)
  resized = cv2.resize(img, shape, interpolation=cv2.INTER_LINEAR)
  img_in = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
  img_in = np.transpose(img_in, (2, 0, 1)).astype(np.float32)
  img_in = np.expand_dims(img_in, axis=0)
  img_in /= 255.0
  img_in = np.ascontiguousarray(img_in)
  # print("Shape of the network input: ", img_in.shape)
  # output_shape = get_binding_shape(engine, "boxes")
  # classes = get_binding_shape(engine, "confs")
  
  end = time.perf_counter()
  read_time = end - start


  start = time.perf_counter()
  # Allocate buffers

  inputs, outputs, bindings, stream = buffers
  # print('Length of inputs: ', len(inputs))
  inputs[0].host = img_in

  trt_outputs = do_inference(engine, context,
    bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
  # Copy input data to host buffer
  #h_input = image.ravel()


  print('Len of outputs: ', len(trt_outputs))

  trt_outputs[0] = trt_outputs[0].reshape(1, -1, 1, 4)
  trt_outputs[1] = trt_outputs[1].reshape(1, -1, len(classes))

  end = time.perf_counter()
  predict_time = end - start

  start = time.perf_counter()
  boxes = post_processing(image, 0.4, 0.6, trt_outputs)
  end = time.perf_counter()
  process_time = end - start

  basename = os.path.basename(image_file)
  plot_boxes_cv2(img, boxes[0],
    savename=f"{output}/{basename}", class_names=classes)

  print(f"trt: predict for {image_file}")
  print(f"  output: {boxes}")
  print(f"  read_time: {read_time:.4f}s")
  print(f"  predict_time: {predict_time:.4f}s")
  print(f"  post_processing: {process_time:.4f}s")

  return read_time, predict_time, process_time

options = parse_args(sys.argv[1:])
has_images = options.image is not None or options.image_dir is not None

print(f"trt: predicting with {options.input}")

if options.input is not None and has_images:
  print(f"trt: loading {options.input}")

  basename = os.path.splitext(options.input)[0]

  if not os.path.isdir(options.output):
    os.makedirs(options.output)

  names_file = f"{basename}.names"
  classes = load_class_names(names_file)

  # 1. Load the trt model
  start = time.perf_counter()

  engine, context = load_engine(options.input)
  buffers = allocate_buffers(engine)

  end = time.perf_counter()
  load_time = end - start
  print(f"  load_time: {load_time:.4f}s")
  
  shape = None

  for i in range(engine.num_io_tensors):
    tensor_name = engine.get_tensor_name(i)
    print(f"  tensor_name[{i}] = {tensor_name}")
    print(f"  tensor_name[{i}].shape = {get_binding_shape(engine,tensor_name)}")

  shape = get_binding_shape(engine, "input")

  if shape is not None:

    shape = (
      shape[3],
      shape[2]
    )

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
        read_time, predict_time, process_time = trt_image_predict(
          engine, context, buffers, shape, classes, options.output, image)
        
        total_read_time += read_time
        total_predict_time += predict_time
        total_process_time += process_time

      avg_read_time = total_read_time / num_predicts
      avg_predict_time = total_predict_time / num_predicts
      avg_process_time = total_process_time / num_predicts

      print(f"trt: time for {num_predicts} predicts")
      print(f"  read_time: total: {total_read_time:.4f}s, avg: {avg_read_time:.4f}")
      print(f"  predict_time: {total_predict_time:.4f}s, avg: {avg_predict_time:.4f}")
      print(f"  process_time: {total_process_time:.4f}s, avg: {avg_process_time:.4f}")

else:

  print("No model or image specified. Printing usage and help.")
  parse_args(["-h"])