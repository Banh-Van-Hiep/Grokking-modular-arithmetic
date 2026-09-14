import torch
from torch.utils.data import DataLoader

from configs.config import (
    PRIME_P, TRAIN_FRACTION, DATA_SEED, TARGET_OP,
    TOTAL_STEPS, PRINT_EVERY, BATCH_SIZE,
    LEARNING_RATE, WEIGHT_DECAY, WARMUP_STEPS,
    BETA1, BETA2, GROKKING_ACC
)

from src.data import generate_datasets, FixedDataset, VOCAB, CHAR2IDX
from src.model import create_model
from src.train import create_optimizer, create_scheduler, train_model


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_data, val_data = generate_datasets(p=PRIME_P, op_name=TARGET_OP, train_fraction=TRAIN_FRACTION, seed=DATA_SEED)

    train_loader = DataLoader(FixedDataset(train_data), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(FixedDataset(val_data), batch_size=BATCH_SIZE, shuffle=False)

    model = create_model(vocab_size=len(VOCAB), sos_id=CHAR2IDX['<SOS>'], eos_id=CHAR2IDX['<EOS>'], pad_id=CHAR2IDX['<PAD>'], device=device)

    optimizer = create_optimizer(model, learning_rate=LEARNING_RATE, weight_decay=WEIGHT_DECAY, betas=(BETA1, BETA2))
    scheduler = create_scheduler(optimizer, warmup_steps=WARMUP_STEPS)

    train_model(model=model, train_loader=train_loader, val_loader=val_loader, optimizer=optimizer, scheduler=scheduler, total_steps=TOTAL_STEPS, print_every=PRINT_EVERY, device=device, eq_id=CHAR2IDX['='], grokking_acc=GROKKING_ACC)


if __name__ == "__main__":
    main()