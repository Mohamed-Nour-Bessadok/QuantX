import unittest
from inference.benchmark import benchmark_onnx, benchmark_torchscript

class TestBenchmarking(unittest.TestCase):
    def test_benchmark_onnx(self):
        onnx_time = benchmark_onnx()
        self.assertTrue(onnx_time < 1.0, "ONNX runtime is too slow")
        
    def test_benchmark_torchscript(self):
        torchscript_time = benchmark_torchscript()
        self.assertTrue(torchscript_time < 1.0, "TorchScript runtime is too slow")

if __name__ == "__main__":
    unittest.main()
