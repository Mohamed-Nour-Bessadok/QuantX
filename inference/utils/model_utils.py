import onnxruntime as ort
import torch

# Load ONNX model
def load_onnx_model(model_path):
    return ort.InferenceSession(model_path)

# Load TorchScript model
def load_torchscript_model(model_path):
    return torch.jit.load(model_path)
