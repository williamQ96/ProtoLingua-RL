import gym
import numpy as np
from config import debug_print

class ColorMatchEnv(gym.Env):
    def __init__(self, num_candidates=4):
        super(ColorMatchEnv, self).__init__()
        self.num_candidates = num_candidates
        self.color_dim = 3  # RGB
        self.target_color = None
        self.candidates = None
        self.target_idx = None

    def reset(self):
        self.target_color = np.random.rand(self.color_dim).astype(np.float32)
        distractors = np.random.rand(self.num_candidates - 1, self.color_dim).astype(np.float32)
        candidates = np.vstack([self.target_color, distractors])
        np.random.shuffle(candidates)

        self.candidates = candidates
        self.target_idx = int(np.where(np.all(candidates == self.target_color, axis=1))[0][0])

        debug_print(f"[DEBUG] Target color: {self.target_color}")
        debug_print(f"[DEBUG] Candidates: {self.candidates}")
        debug_print(f"[DEBUG] Correct index: {self.target_idx}")

        return {
            "speaker_obs": self.target_color.copy(),
            "listener_obs": self.candidates.copy()
        }

    def step(self, listener_action):
        correct = int(listener_action == self.target_idx)
        reward = 1.0 if correct else 0.0
        done = True
        info = {
            "target_color": self.target_color.copy(),
            "candidates": self.candidates.copy(),
            "target_idx": self.target_idx,
            "correct": correct
        }

        debug_print(f"[DEBUG] Listener chose: {listener_action} | Correct: {self.target_idx} | Reward: {reward}")
        return None, reward, done, info
