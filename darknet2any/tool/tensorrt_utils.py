
############################################################################
# --------------------------------------------------------------------------
# @author James Edmondson <james@koshee.ai>
# --------------------------------------------------------------------------
############################################################################

"""
provides utilities for tensorrt
"""

import os
import pathlib
import platform

def load_tensorrt_libs(full_search=False):
  """
  searches for tensorrt libs, starting from project directory, assuming that
  the project was installed locally or via pip. Can search whole system also,
  but that takes a long time. Sets LD_LIBRARY_PATH with results.
  
  Returns:
    tuple(str,str): location of ort_providers_tensorrt and libnvinfer
  """

  this_dir = pathlib.Path(__file__).parent.resolve() #tool
  project_dir = os.path.dirname(this_dir) #darknet2any/darknet2any
  project_dir = os.path.dirname(project_dir) #darknet2any project
  home_dir = os.path.expanduser("~")
  
  search_list = []
  
  if full_search:
    search_list = ["/"]
  else:
    search_list = [project_dir, home_dir]
  
  print("load_tensorrt_libs: search paths")
  for option in search_list:
    print(f"  {option}")
    
  ort_trt_lib = "libonnxruntime_providers_tensorrt"
  libnvinfer = "libnvinfer"
  
  uname_info = platform.uname()
  # print(f"System: {uname_info.system}")
  # print(f"Node Name: {uname_info.node}")
  # print(f"Release: {uname_info.release}")
  # print(f"Version: {uname_info.version}")
  # print(f"Machine: {uname_info.machine}")
  # print(f"Processor: {uname_info.processor}")

  if uname_info.system == "Windows":
    ort_trt_lib += ".dll"
    libnvinfer += ".dll"
  else:
    ort_trt_lib += ".so"
    libnvinfer += ".so"
    
  ort_tep_dir = None
  nvinfer_dir = None
  done = False
  
  for path in search_list:
    print(f"find_tensorrt_libs: searching for {ort_trt_lib} and {libnvinfer} in {path}")
    for dir, _, files in os.walk(path):
      for file in files:
        
        if ort_tep_dir is None and file.startswith(ort_trt_lib):
          abs_path = f"{dir}/{file}"
          print(f"find_tensorrt_libs: found {ort_trt_lib} at {abs_path}")
          os.environ["LD_LIBRARY_PATH"] = f"{dir}:" + os.environ.get(
            "LD_LIBRARY_PATH")
          ort_tep_dir = dir
          
          if ort_tep_dir is not None and nvinfer_dir is not None:
            done = True
            break
        elif nvinfer_dir is None and file.startswith(libnvinfer):
          abs_path = f"{dir}/{file}"
          print(f"find_tensorrt_libs: found {libnvinfer} at {abs_path}")
          os.environ["LD_LIBRARY_PATH"] = f"{dir}:" + os.environ.get(
            "LD_LIBRARY_PATH")
          nvinfer_dir = dir
          
          if ort_tep_dir is not None and nvinfer_dir is not None:
            done = True
            break
      
      if done:
        break
  
    if done:
      break
    
  print(f"find_tensorrt_libs: LD_LIBRARY_PATH=" + os.environ.get(
    "LD_LIBRARY_PATH"))
  
  return ort_tep_dir, nvinfer_dir

    
        
      
  
