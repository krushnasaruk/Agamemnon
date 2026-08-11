import os
import torch
import torch.nn as nn
from typing import List, Dict, Any, Tuple
from models.baseline_cnn import BaselineCNN

class CompressedCNN(BaselineCNN):
    """
    Child class of BaselineCNN extending functionality for RL architecture search,
    checkpoint saving, metadata tracking, and ONNX export.
    """
    def __init__(self, channels: List[int], num_classes: int = 10):
        super(CompressedCNN, self).__init__(channels=channels, num_classes=num_classes)
        self.metadata: Dict[str, Any] = {
            "channels": list(channels),
            "num_classes": num_classes,
            "pruning_history": []
        }

    def export_onnx(self, file_path: str, input_size: Tuple[int, int, int, int] = (1, 3, 32, 32)):
        """Exports the compressed model to ONNX format or PyTorch checkpoint if ONNX engine is unavailable."""
        self.eval()
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        dummy_input = torch.randn(*input_size, device=next(self.parameters()).device)
        
        try:
            torch.onnx.export(
                self,
                dummy_input,
                file_path,
                export_params=True,
                opset_version=11,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
            )
            print(f"[Export] Saved compressed ONNX model to: {file_path}")
            return file_path
        except Exception as e:
            pt_path = file_path.replace(".onnx", ".pt")
            torch.save({
                "model_state_dict": self.state_dict(),
                "channels": self.get_channel_config()
            }, pt_path)
            print(f"[Export Notice] ONNX exporter fallback ({e}). Saved compressed PyTorch model checkpoint to: {pt_path}")
            return pt_path
