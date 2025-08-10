import sys
import onnx
from onnx_tf.backend import prepare


# tensorflow >=2.0
# 1: Thanks:github:https://github.com/onnx/onnx-tensorflow
# 2: Run git clone https://github.com/onnx/onnx-tensorflow.git && cd onnx-tensorflow
#  Run pip install -e .
# Note:
#  Errors will occur when using "pip install onnx-tf", at least for me,
#  it is recommended to use source code installation
def transform_to_tensorflow(onnx_input_path, pb_output_path):
  onnx_model = onnx.load(onnx_input_path)  # load onnx model
  tf_exp = prepare(onnx_model)  # prepare tf representation
  tf_exp.export_graph(pb_output_path)  # export the model

