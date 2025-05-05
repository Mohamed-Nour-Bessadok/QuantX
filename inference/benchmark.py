import time
import torch
import onnxruntime as ort
from inference.inference import run_onnx_inference, run_torchscript_inference

onnx_model_path = "model/model_quantized.onnx"
torchscript_model_path = "model/model_quantized.pt"
image_path = "data/imag74.jpg"

def benchmark_onnx():
    start_time = time.time()
    result = run_onnx_inference(onnx_model_path, image_path)
    elapsed_time = time.time() - start_time
    print(f"ONNX Runtime inference time: {elapsed_time:.4f} seconds")
    return elapsed_time

def benchmark_torchscript():
    start_time = time.time()
    result = run_torchscript_inference(torchscript_model_path, image_path)
    elapsed_time = time.time() - start_time
    print(f"TorchScript inference time: {elapsed_time:.4f} seconds")
    return elapsed_time

if __name__ == "__main__":
    print("Benchmarking ONNX Runtime...")
    onnx_time = benchmark_onnx()
    print("Benchmarking TorchScript...")
    torchscript_time = benchmark_torchscript()

    # Comparison of results
    print(f"ONNX Time: {onnx_time:.4f}s | TorchScript Time: {torchscript_time:.4f}s")
