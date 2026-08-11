import unittest
import torch
from models.baseline_cnn import build_baseline_cnn
from pruning.channel_reduction import prune_layer_channels
from pruning.layer_removal import remove_or_bypass_layer
from pruning.quantization import quantize_model
from evaluation.parameters import count_parameters

class TestPruning(unittest.TestCase):
    def test_channel_reduction(self):
        model = build_baseline_cnn(channels=[64, 128, 256, 256, 512], num_classes=10)
        orig_params = count_parameters(model)
        
        pruned_model, new_channels = prune_layer_channels(
            model=model,
            layer_idx=2, # Prune Conv3
            reduction_ratio=0.5
        )
        new_params = count_parameters(pruned_model)
        
        self.assertEqual(new_channels, [64, 128, 128, 256, 512])
        self.assertLess(new_params, orig_params)
        
        # Verify forward pass tensor integrity
        x = torch.randn(2, 3, 32, 32)
        out = pruned_model(x)
        self.assertEqual(out.shape, (2, 10))

    def test_layer_removal(self):
        model = build_baseline_cnn(channels=[64, 128, 256, 256, 512], num_classes=10)
        bypassed_model, new_channels = remove_or_bypass_layer(model, layer_idx=2)
        self.assertEqual(new_channels[2], new_channels[1]) # Conv3 matches Conv2 channels

    def test_quantization(self):
        model = build_baseline_cnn(channels=[16, 32, 64, 64, 128], num_classes=10)
        qmodel = quantize_model(model)
        x = torch.randn(1, 3, 32, 32)
        out = qmodel(x)
        self.assertEqual(out.shape, (1, 10))

if __name__ == "__main__":
    unittest.main()
