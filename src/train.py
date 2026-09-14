import time
import torch
from transformers import get_constant_schedule_with_warmup

from src.evaluate import create_labels, evaluate_accuracy


def create_optimizer(model, learning_rate=1e-3, weight_decay=1.0, betas=(0.9, 0.98)):
    decay_params = [p for n, p in model.named_parameters() if p.requires_grad and len(p.shape) > 1 and "ln" not in n and not n.endswith(".bias")]
    nodecay_params = [p for n, p in model.named_parameters() if p.requires_grad and (len(p.shape) <= 1 or "ln" in n or n.endswith(".bias"))]
    optimizer = torch.optim.AdamW([{"params": decay_params, "weight_decay": weight_decay}, {"params": nodecay_params, "weight_decay": 0.0}], lr=learning_rate, betas=betas)
    return optimizer


def create_scheduler(optimizer, warmup_steps=1000):
    return get_constant_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps)


def train_model(model, train_loader, val_loader, optimizer, scheduler, total_steps, print_every, device, eq_id, grokking_acc=99.0):
    print("Bắt đầu huấn luyện Grokking")
    step = 0
    start_time = time.time()

    model.train()

    while step < total_steps:
        for tokens in train_loader:
            step += 1
            tokens = tokens.to(device)
            labels = create_labels(tokens, eq_id)

            optimizer.zero_grad()
            outputs = model(input_ids=tokens, labels=labels)
            loss = outputs.loss
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            if step % print_every == 0:
                elapsed = time.time() - start_time
                train_acc = evaluate_accuracy(model, train_loader, device) * 100
                val_acc = evaluate_accuracy(model, val_loader, device) * 100

                print(f"Step [{step}/{total_steps}] | Loss: {loss.item():.4f} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}% | Time: {elapsed:.1f}s")

                if val_acc >= grokking_acc:
                    print(f"\nMô hình đã GROK thành công tại step {step}!")
                    return step

            if step >= total_steps:
                print("\nĐã đạt số step tối đa nhưng chưa Grokking hoàn toàn.")
                return step