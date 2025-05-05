import torch
import onnxruntime as ort
import numpy as np
from inference.utils.image_preprocessing import preprocess_image
from inference.utils.model_utils import load_onnx_model, load_torchscript_model

def run_inference(model, input_data):
    input_name = model.get_inputs()[0].name
    output_name = model.get_outputs()[0].name
    result = model.run([output_name], {input_name: input_data})
    return result

def run_torchscript_inference(model, input_data):
    with torch.no_grad():
        output = model(input_data)
    return output

def run_onnx_inference(onnx_model_path, image_path):
    model = load_onnx_model(onnx_model_path)
    input_data = preprocess_image(image_path)
    return run_inference(model, input_data)

def run_torchscript_inference(torchscript_model_path, image_path):
    model = load_torchscript_model(torchscript_model_path)
    input_data = preprocess_image(image_path)
    return run_torchscript_inference(model, input_data)

if __name__ == "__main__":
    image_path = "data/imag74.jpg"
    onnx_model_path = "model/model_quantized.onnx"
    torchscript_model_path = "model/model_quantized.pt"

    onnx_result = run_onnx_inference(onnx_model_path, image_path)
    torchscript_result = run_torchscript_inference(torchscript_model_path, image_path)
    print(f"ONNX result: {onnx_result}")
    print(f"TorchScript result: {torchscript_result}")
