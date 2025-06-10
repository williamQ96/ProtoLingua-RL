from collections import defaultdict
import math
from config import debug_print

token_usage = defaultdict(lambda: defaultdict(int))

def round_color(color_tensor, decimals=1):
    return str([round(float(v), decimals) for v in color_tensor])

def update_token_stats(message_tensor, target_tensor):
    message = message_tensor[0].tolist()  # assuming batch size 1
    color_key = round_color(target_tensor[0])

    for token in message:
        token_usage[token][color_key] += 1

    debug_print(f"[DEBUG] Updated token stats: {message} → {color_key}")

def print_token_dictionary(top_k=1):
    debug_print("\n=== [DEBUG] Token-to-Color Dictionary ===")
    for token in sorted(token_usage.keys()):
        color_counts = token_usage[token]
        sorted_colors = sorted(color_counts.items(), key=lambda x: -x[1])
        top_colors = [color for color, _ in sorted_colors[:top_k]]
        debug_print(f"[DEBUG] Token {token} → {top_colors}")

def get_diversity_penalty():
    """Returns the average negative entropy of token-color distributions.
       Lower value is better (more diversity)."""
    total_penalty = 0
    token_count = 0

    for token, color_counts in token_usage.items():
        counts = list(color_counts.values())
        total = sum(counts)
        if total == 0:
            continue
        probs = [count / total for count in counts]
        entropy = -sum(p * math.log(p + 1e-8) for p in probs)
        penalty = -entropy  # Negative entropy = penalty
        total_penalty += penalty
        token_count += 1

    return total_penalty / token_count if token_count > 0 else 0.0
