import unittest
import numpy as np
from agent.dqn_agent import DQNAgent
from agent.ppo_agent import PPOAgent

class TestAgents(unittest.TestCase):
    def test_dqn_agent_step_and_update(self):
        agent = DQNAgent(state_dim=10, action_dim=10, batch_size=4)
        dummy_state = np.random.randn(10).astype(np.float32)
        
        action = agent.select_action(dummy_state)
        self.assertTrue(0 <= action < 10)

        # Store transitions
        for _ in range(5):
            s = np.random.randn(10).astype(np.float32)
            a = np.random.randint(0, 10)
            r = 1.0
            ns = np.random.randn(10).astype(np.float32)
            d = False
            agent.store_transition(s, a, r, ns, d)

        loss_info = agent.update()
        self.assertIn("loss", loss_info)

    def test_ppo_agent_step_and_update(self):
        agent = PPOAgent(state_dim=10, action_dim=10, mini_batch_size=2, ppo_epochs=2)
        dummy_state = np.random.randn(10).astype(np.float32)

        action, log_prob, value = agent.select_action(dummy_state)
        self.assertTrue(0 <= action < 10)

        # Store rollout steps
        for _ in range(4):
            s = np.random.randn(10).astype(np.float32)
            a, lp, val = agent.select_action(s)
            agent.store_transition(s, a, 1.0, s, False, log_prob=lp, value=val)

        loss_info = agent.update()
        self.assertIn("actor_loss", loss_info)
        self.assertIn("critic_loss", loss_info)

if __name__ == "__main__":
    unittest.main()
