# QuantX: Containerized AI Inference Pipeline

## Overview

**QuantX** is a **hardware-agnostic AI inference pipeline** optimized for **CPU-only environments**. The pipeline supports a **MobileNet-based object detection model** and benchmarks **ONNX Runtime** and **TorchScript** for efficient model inference. The project integrates post-training quantization techniques to enhance the model's size, latency, and throughput, making it ideal for scalable and reproducible deployment in production environments.

This project includes:
- **Model Export**: Convert models to **ONNX** and **TorchScript** formats.
- **Model Quantization**: Apply post-training quantization to optimize model size and speed.
- **Benchmarking**: Compare the performance of **ONNX Runtime** vs **TorchScript**.
- **Containerization**: Package the entire pipeline in a **Docker container** for reproducibility and scalability.