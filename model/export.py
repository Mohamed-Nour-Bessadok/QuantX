import torch
import torch.onnx

def export_model_to_onnx(model, export_path="model_quantized.onnx"):
    dummy_input = torch.randn(1, 3, 224, 224)  # Example input for object detection
    torch.onnx.export(model, dummy_input, export_path, opset_version=12)
    print(f"Model exported to {export_path}")

def export_model_to_torchscript(model, export_path="model_quantized.pt"):
    scripted_model = torch.jit.script(model)
    scripted_model.save(export_path)
    print(f"Model exported to {export_path}")
