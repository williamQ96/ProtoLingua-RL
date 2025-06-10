from collections import defaultdict
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
