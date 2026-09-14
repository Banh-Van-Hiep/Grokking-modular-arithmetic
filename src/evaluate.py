import torch
from src.data import CHAR2IDX

def create_labels(tokens, eq_id):
    labels = torch.full_like(tokens, -100)
    eq_mask = (tokens == eq_id)

    if eq_mask.any():
        eq_indices = eq_mask.int().argmax(dim=1)
        target_indices = eq_indices + 1
        valid_mask = target_indices < tokens.size(1)
        batch_idx = torch.arange(tokens.size(0), device=tokens.device)[valid_mask]
        target_idx = target_indices[valid_mask]
        labels[batch_idx, target_idx] = tokens[batch_idx, target_idx]

    return labels

@torch.no_grad()
def evaluate_accuracy(model, data_loader, device):
    model.eval()
    correct, total = 0, 0
    eq_id = CHAR2IDX['=']

    for tokens in data_loader:
        tokens = tokens.to(device)
        outputs = model(input_ids=tokens)
        preds = torch.argmax(outputs.logits, dim=-1)

        for i in range(tokens.size(0)):
            eq_pos = (tokens[i] == eq_id).nonzero(as_tuple=True)[0]
            if len(eq_pos) > 0:
                pos = eq_pos[0].item()
                if pos + 1 < tokens.size(1):
                    if preds[i, pos].item() == tokens[i, pos + 1].item():
                        correct += 1
                    total += 1

    model.train()
    return correct / total if total > 0 else 0.0