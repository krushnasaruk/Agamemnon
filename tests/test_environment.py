import unittest
import torch
from torch.utils.data import DataLoader, TensorDataset
from models.baseline_cnn import build_baseline_cnn
from environment.rl_environment import ArchitectureSearchEnv

class TestEnvironment(unittest.TestCase):
    def setUp(self):
        # Create dummy dataloaders for fast test execution
        x_dummy = torch.randn(20, 3, 32, 32)
        y_dummy = torch.randint(0, 10, (20,))
        dataset = TensorDataset(x_dummy, y_dummy)
        self.train_loader = DataLoader(dataset, batch_size=10)
        self.val_loader = DataLoader(dataset, batch_size=10)
        
        self.model = build_baseline_cnn(channels=[16, 32, 64, 64, 128], num_classes=10)

    def test_env_reset_and_step(self):
        env = ArchitectureSearchEnv(
            baseline_model=self.model,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            device="cpu",
            max_steps=5,
            finetune_epochs=0
        )
        
        state, info = env.reset()
        self.assertEqual(state.shape, (10,))
        self.assertIn("accuracy", info)

        # Execute channel reduction action
        next_state, reward, terminated, truncated, info = env.step(action=1)
        self.assertEqual(next_state.shape, (10,))
        self.assertIsInstance(reward, float)
        self.assertIn("action_desc", info)

if __name__ == "__main__":
    unittest.main()
