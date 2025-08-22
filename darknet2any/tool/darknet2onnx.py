import sys
import torch
from darknet2any.tool.darknet2pytorch import Darknet


def transform_to_onnx(cfgfile, weightfile, batch_size=1,
  onnx_file_name=None, include_embeddings=False, opset=15):
  model = Darknet(cfgfile, include_embeddings=include_embeddings)

  model.print_network()
  model.load_weights(weightfile)
  print('Loading weights from %s... Done!' % (weightfile))

  # print('Printing model children!')
  # for child_module in model.children():
  #     print(child_module)

  # print('Printing named model children!')
  # for name, child_module in model.named_children():
  #     print(f"{name} = {child_module}")

  dynamic = False
  if batch_size <= 0:
    dynamic = True

  input_names = ["input"]
  output_names = ['boxes', 'confs']
  
  if include_embeddings:
    output_names.append('features')

  if dynamic:
    x = torch.randn((1, 3, model.height, model.width), requires_grad=True)
    if not onnx_file_name:
      onnx_file_name = "yolov4_-1_3_{}_{}_dynamic.onnx".format(model.height, model.width)
    dynamic_axes = {"input": {0: "batch_size"}, "boxes": {0: "batch_size"}, "confs": {0: "batch_size"}}
    # Export the model
    print(f'Export the onnx model to {onnx_file_name} with opset={opset}...')
    torch.onnx.export(model,
              x,
              onnx_file_name,
              export_params=True,
              opset_version=opset,
              do_constant_folding=True,
              input_names=input_names, output_names=output_names,
              dynamic_axes=dynamic_axes)

    print('Onnx model exporting done')
    return onnx_file_name

  else:
    x = torch.randn((batch_size, 3, model.height, model.width), requires_grad=True)
    if onnx_file_name is None:
      onnx_file_name = "yolov4_{}_3_{}_{}_static.onnx".format(batch_size, model.height, model.width)
    print(f'Export the onnx model to {onnx_file_name} with opset={opset}...')
    torch.onnx.export(model,
              x,
              onnx_file_name,
              export_params=True,
              opset_version=opset,
              do_constant_folding=True,
              input_names=input_names, output_names=output_names,
              dynamic_axes=None)

    print('Onnx model exporting done')
    return onnx_file_name
