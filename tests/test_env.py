import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from envs.color_match_env import ColorMatchEnv
import numpy as np


print("[DEBUG] Starting environment test...")

# Instantiate the environment
env = ColorMatchEnv(num_candidates=4)

# Reset the environment
obs = env.reset()
speaker_obs = obs["speaker_obs"]
listener_obs = obs["listener_obs"]

print("\n--- Environment Reset ---")
print(f"[DEBUG] Speaker Obs (Target Color): {speaker_obs}")
print(f"[DEBUG] Listener Obs (Candidates):\n{listener_obs}")

# Simulate the listener choosing a random index (as if untrained)
fake_listener_action = np.random.randint(0, 4)

# Step environment with fake action
_, reward, done, info = env.step(fake_listener_action)

print("\n--- Environment Step ---")
print(f"[DEBUG] Listener Action: {fake_listener_action}")
print(f"[DEBUG] Reward: {reward}")
print(f"[DEBUG] Info: {info}")
