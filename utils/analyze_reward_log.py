import sys

def compute_final_success_rate(log_path, window_size=50):
    rewards = []

    try:
        with open(log_path, "r") as f:
            for line in f:
                parts = line.strip().split(",")
                if len(parts) == 2:
                    _, reward = parts
                    rewards.append(float(reward))
    except FileNotFoundError:
        print(f"File not found: {log_path}")
        return

    if len(rewards) < window_size:
        print(f"Not enough data: only {len(rewards)} episodes logged.")
        return

    final_avg = sum(rewards[-window_size:]) / window_size
    print(f"\nFile: {log_path}")
    print(f"Final {window_size}-Episode Success Rate: {final_avg:.4f}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyze_reward_log.py <reward_log_file>")
        sys.exit(1)

    log_file = sys.argv[1]
    compute_final_success_rate(log_file)
