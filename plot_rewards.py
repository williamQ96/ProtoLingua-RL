import os
import sys
import matplotlib.pyplot as plt

def moving_average(data, window_size=500):
    return [sum(data[max(0, i - window_size + 1):i + 1]) / min(i + 1, window_size) for i in range(len(data))]

def load_rewards(log_path):
    episodes, rewards = [], []
    with open(log_path, "r") as f:
        for line in f:
            ep, r = line.strip().split(",")
            episodes.append(int(ep))
            rewards.append(float(r))
    return episodes, rewards

def plot_reward_file(log_path, mode, label=None):
    episodes, rewards = load_rewards(log_path)
    smoothed = moving_average(rewards)
    label = label or os.path.basename(log_path)

    if mode == 0:
        plt.plot(episodes, rewards, label=f"{label} (raw)", alpha=0.3)
    plt.plot(episodes, smoothed, label=f"{label} (smoothed)")

if __name__ == "__main__":
    if len(sys.argv) < 3 or len(sys.argv) > 5:
        print("Usage:")
        print("  python plot_rewards.py MODE log1.txt [log2.txt] [log3.txt]")
        print("  MODE = 0 for raw+smoothed, 1 for smoothed only")
        sys.exit(1)

    try:
        mode = int(sys.argv[1])
        if mode not in (0, 1):
            raise ValueError()
    except ValueError:
        print("First argument (MODE) must be 0 or 1.")
        sys.exit(1)

    log_paths = sys.argv[2:]
    for log_path in log_paths:
        plot_reward_file(log_path, mode)

    plt.figure(1, figsize=(10, 5))
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Training Reward Over Time")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
