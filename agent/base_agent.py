from abc import ABC, abstractmethod
import numpy as np

class BaseRLAgent(ABC):
    """
    Abstract interface for RL search agents.
    """
    @abstractmethod
    def select_action(self, state: np.ndarray, evaluate: bool = False):
        pass

    @abstractmethod
    def store_transition(self, state, action, reward, next_state, done, *args, **kwargs):
        pass

    @abstractmethod
    def update(self) -> dict:
        pass
