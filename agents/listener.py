import torch
import torch.nn as nn
import torch.nn.functional as F
from config import debug_print

VOCAB_SIZE = 10
MESSAGE_LEN = 3
HIDDEN_SIZE = 64
NUM_CANDIDATES = 4

class Listener(nn.Module):
    def __init__(self, vocab_size=VOCAB_SIZE, message_len=MESSAGE_LEN, hidden_size=HIDDEN_SIZE, candidate_dim=3, num_candidates=NUM_CANDIDATES):
        super(Listener, self).__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.msg_encoder = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.fc1 = nn.Linear(hidden_size + candidate_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)
        self.num_candidates = num_candidates

    def forward(self, message, candidates):
        embedded = self.embedding(message)                            # [B, L, H]
        _, h = self.msg_encoder(embedded)                             # h: [1, B, H]
        h = h.squeeze(0)                                              # [B, H]

        num_candidates = candidates.size(1)
        h_expanded = h.unsqueeze(1).expand(-1, num_candidates, -1)    # [B, C, H]
        joint = torch.cat([h_expanded, candidates], dim=2)            # [B, C, H+3]

        x = F.relu(self.fc1(joint))                                   # [B, C, H]
        logits = self.fc2(x).squeeze(-1)                              # [B, C]
        action = torch.argmax(logits, dim=1)                          # [B]

        debug_print(f"[DEBUG] Listener message: {message}")
        debug_print(f"[DEBUG] Listener selected action: {action}")
        return action, logits

    def generate_feedback(self, selected_candidate, true_target):
        similarity = F.cosine_similarity(selected_candidate, true_target, dim=-1)
        feedback = "Correct" if similarity.item() > 0.95 else "Almost" if similarity.item() > 0.8 else "Wrong"
        debug_print(f"[DEBUG] Listener feedback: {feedback} (similarity: {similarity.item():.2f})")
        return feedback, similarity
