import unittest
import torch
from models.baseline_cnn import BaselineCNN, build_baseline_cnn
from models.compressed_cnn import CompressedCNN
from evaluation.parameters import count_parameters
from evaluation.flops import count_flops
from evaluation.latency import measure_latency, get_model_size_mb

class TestModels(unittest.TestCase):
    def test_baseline_cnn_forward(self):
        model = build_baseline_cnn(channels=[64, 128, 256, 256, 512], num_classes=10)
        x = torch.randn(2, 3, 32, 32)
        out = model(x)
        self.assertEqual(out.shape, (2, 10))

    def test_dynamic_channel_cnn(self):
        model = BaselineCNN(channels=[32, 64, 128, 128, 256], num_classes=10)
        x = torch.randn(4, 3, 32, 32)
        out = model(x)
        self.assertEqual(out.shape, (4, 10))
        self.assertEqual(model.get_channel_config(), [32, 64, 128, 128, 256])

    def test_metrics(self):
        model = build_baseline_cnn(channels=[16, 32, 64, 64, 128], num_classes=10)
        params = count_parameters(model)
        flops = count_flops(model, input_size=(1, 3, 32, 32))
        lat = measure_latency(model, runs=5, warmup=2)
        size_mb = get_model_size_mb(model)

        self.assertGreater(params, 0)
        self.assertGreater(flops, 0)
        self.assertGreater(lat, 0.0)
        self.assertGreater(size_mb, 0.0)

if __name__ == "__main__":
    unittest.main()
