import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
from agents.speaker import Speaker
from agents.listener import Listener


print("[DEBUG] Import successful, initializing agents...")

# Set reproducibility
torch.manual_seed(42)

# Instantiate Speaker and Listener
speaker = Speaker()
listener = Listener()

print("[DEBUG] Agents created.")

# === Step 1: Fake task input ===
target_color = torch.tensor([[0.8, 0.1, 0.1]])  # bright red
candidates = torch.tensor([
    [[0.8, 0.1, 0.1], [0.0, 0.0, 1.0], [0.1, 0.8, 0.1], [1.0, 1.0, 0.0]]
], dtype=torch.float32)  # 1 batch, 4 RGB options

print(f"[DEBUG] Target color: {target_color}")
print(f"[DEBUG] Candidates: {candidates}")

# === Step 2: Speaker generates message ===
message, _ = speaker(target_color)
print(f"[DEBUG] Message from Speaker: {message}")

# === Step 3: Listener chooses based on message ===
action, logits = listener(message, candidates)
print(f"[DEBUG] Action selected by Listener: {action}")
print(f"[DEBUG] Logits from Listener: {logits}")

# === Final Output ===
print("\n--- FINAL TEST RESULTS ---")
print(f"Target Color     : {target_color.tolist()}")
print(f"Candidates       : {candidates.tolist()}")
print(f"Speaker Message  : {message.tolist()}")
print(f"Listener Choice  : Candidate index {action.item()}")
