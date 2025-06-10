import torch
import torch.nn as nn
import torch.nn.functional as F
from config import debug_print  # Use shared debug

VOCAB_SIZE = 10
MESSAGE_LEN = 3
HIDDEN_SIZE = 64

class Speaker(nn.Module):
    def __init__(self, input_dim=3, hidden_size=HIDDEN_SIZE, vocab_size=VOCAB_SIZE, message_len=MESSAGE_LEN):
        super(Speaker, self).__init__()
        self.message_len = message_len
        self.fc1 = nn.Linear(input_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, vocab_size * message_len)

    def forward(self, obs):
        x = F.relu(self.fc1(obs))
        logits = self.fc2(x)
        logits = logits.view(-1, self.message_len, VOCAB_SIZE)
        probs = F.softmax(logits, dim=-1)
        message = torch.multinomial(probs.view(-1, VOCAB_SIZE).contiguous(), 1).view(-1, self.message_len)
        debug_print(f"[DEBUG] Speaker input: {obs}")
        debug_print(f"[DEBUG] Speaker message: {message}")
        return message, probs
