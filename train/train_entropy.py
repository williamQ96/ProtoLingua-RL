import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm

from config import DEBUG_MODE, debug_print
from agents.speaker import Speaker
from agents.listener import Listener
from envs.color_match_env import ColorMatchEnv
from utils.token_logger import update_token_stats, print_token_dictionary

global_log_path = "logs/global_training_log.txt"
script_name = os.path.basename(__file__)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
debug_print(f"[DEBUG] Using device: {device}")

if len(sys.argv) < 2:
    print("Usage: python train_entropy.py <num_episodes>")
    sys.exit(1)

num_episodes = int(sys.argv[1])
progress_bar = tqdm(total=num_episodes, desc="Training", dynamic_ncols=True)
log_filename = "logs/reward_log_entropy_{num_episodes}.txt"
with open(global_log_path, "a") as f:
    f.write(f"[START] {script_name} | Episodes: {num_episodes}\n")

def get_curriculum_env(episode):
    if episode < 0.05 * num_episodes:
        return ColorMatchEnv(num_candidates=2)
    elif episode < 0.1 * num_episodes:
        return ColorMatchEnv(num_candidates=3)
    else:
        return ColorMatchEnv(num_candidates=4)

speaker = Speaker().to(device)
listener = Listener().to(device)

speaker_opt = optim.Adam(speaker.parameters(), lr=1e-3)
listener_opt = optim.Adam(listener.parameters(), lr=1e-3)

debug_print("[DEBUG] Initialized environment and agents.")

start_time = time.time()
reward_history = []
consistency_weight = 0.1
entropy_weight = 0.01

for episode in range(num_episodes):
    progress_bar.update(1)

    env = get_curriculum_env(episode)
    obs = env.reset()
    target = torch.tensor(obs["speaker_obs"][None, :], dtype=torch.float32, device=device)
    candidates = torch.tensor(obs["listener_obs"][None, :], dtype=torch.float32, device=device)

    # Speaker forward
    message, msg_probs = speaker(target)
    log_probs = torch.log(torch.gather(msg_probs, 2, message.unsqueeze(-1)).squeeze(-1))
    total_msg_logprob = log_probs.sum(dim=1)
    entropy_speaker = -(msg_probs * torch.log(msg_probs + 1e-10)).sum(dim=2).mean()

    # Consistency loss
    with torch.no_grad():
        message_dup, _ = speaker(target)
    consistency_loss = torch.abs(message.float() - message_dup.float()).mean()

    # Listener forward
    action, logits = listener(message, candidates)
    logits_softmax = torch.log_softmax(logits, dim=1)
    action_logprob = logits_softmax[0, action]
    entropy_listener = -torch.sum(torch.exp(logits_softmax) * logits_softmax)

    # Environment step
    _, reward, _, info = env.step(action.item())
    update_token_stats(message, target)
    reward_tensor = torch.tensor([reward], dtype=torch.float32, device=device)

    # Compute loss
    speaker_loss = (
        -total_msg_logprob.mean() * reward_tensor
        - entropy_weight * entropy_speaker
        + consistency_weight * consistency_loss
    )
    listener_loss = -action_logprob * reward_tensor - entropy_weight * entropy_listener

    speaker_opt.zero_grad()
    listener_opt.zero_grad()
    (speaker_loss + listener_loss).backward()
    speaker_opt.step()
    listener_opt.step()

    reward_history.append(reward)

    if episode % 10 == 0:
        avg_reward = sum(reward_history[-10:]) / 10
        debug_print(f"[DEBUG] Ep {episode} | Reward: {reward:.2f} | Avg(10): {avg_reward:.2f} | Action: {action.item()} | Correct: {info['correct']}")

    if episode % 500 == 0 and episode > 0:
        interim = sum(reward_history[-50:]) / 50
        debug_print(f"[DEBUG] Interim Success Rate (last 50 eps): {interim:.2f}")

print("\nTraining complete.")
success_rate = sum(reward_history[-50:]) / 50
print(f"\nFinal 50-Episode Success Rate: {success_rate:.2f}")

os.makedirs("logs", exist_ok=True)
with open(log_filename, "w") as f:
    for i, r in enumerate(reward_history):
        f.write(f"{i},{r}\n")

print_token_dictionary()
end_time = time.time()
elapsed = end_time - start_time
debug_print(f"[DEBUG] Training time: {elapsed:.2f} seconds")
with open(global_log_path, "a") as f:
    f.write(f"[END]   {script_name} | Success Rate: {success_rate:.2f} | Time Spent: {elapsed:.2f}s\n\n")
progress_bar.close()
