from transformers import GPT2Config, GPT2LMHeadModel

def create_model(vocab_size, sos_id, eos_id, pad_id, device):
    config = GPT2Config(
        vocab_size=vocab_size,
        n_positions=32,
        n_embd=128,
        n_layer=2,
        n_head=4,
        n_inner=512,
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
        summary_first_dropout=0.0,
        bos_token_id=sos_id,
        eos_token_id=eos_id,
        pad_token_id=pad_id
    )

    return GPT2LMHeadModel(config).to(device)