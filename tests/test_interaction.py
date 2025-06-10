import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import numpy as np
from agents.speaker import Speaker
from agents.listener import Listener
from envs.color_match_env import ColorMatchEnv

# Debug setup
print("[DEBUG] Starting full interaction cycle test...")

# Ensure reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Instantiate agents and environment
speaker = Speaker()
listener = Listener()
env = ColorMatchEnv(num_candidates=4)

# === Step 1: Reset environment ===
obs = env.reset()
speaker_obs = torch.tensor([obs["speaker_obs"]], dtype=torch.float32)
listener_obs = torch.tensor([obs["listener_obs"]], dtype=torch.float32)  # shape [1, 4, 3]

print(f"[DEBUG] Env returned:")
print(f"  [DEBUG] Speaker sees target color: {speaker_obs}")
print(f"  [DEBUG] Listener sees candidates:\n{listener_obs}")

# === Step 2: Speaker speaks ===
message, _ = speaker(speaker_obs)
print(f"[DEBUG] Speaker generated message: {message.tolist()}")

# === Step 3: Listener listens and acts ===
action, logits = listener(message, listener_obs)
print(f"[DEBUG] Listener logits: {logits}")
print(f"[DEBUG] Listener chose index: {action.item()}")

# === Step 4: Env gives feedback ===
_, reward, done, info = env.step(action.item())
print("\n--- ENVIRONMENT FEEDBACK ---")
print(f"[DEBUG] Correct index: {info['target_idx']}")
print(f"[DEBUG] Listener correct?: {bool(info['correct'])}")
print(f"[DEBUG] Reward: {reward}")