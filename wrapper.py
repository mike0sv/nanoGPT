"""
Sample from a trained model
"""
import os
import pickle
import sys

import torch

from model import GPT, GPTConfig

ptdtype = torch.float32


def convert(checkpoint_path, target_path):
    ckpt_path = os.path.join(checkpoint_path, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    model.eval()

    # look for the meta pickle in case it is available in the dataset folder
    load_meta = False
    if 'config' in checkpoint and 'dataset' in checkpoint[
        'config']:  # older checkpoints might not have these...
        meta_path = os.path.join('data', checkpoint['config']['dataset'],
                                 'meta.pkl')
        load_meta = os.path.exists(meta_path)
    if load_meta:
        print(f"Loading meta from {meta_path}...")
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)
        # TODO want to make this more general to arbitrary encoder/decoder schemes
        stoi, itos = meta['stoi'], meta['itos']
        encode = lambda s: [stoi[c] for c in s]
        decode = lambda l: ''.join([itos[i] for i in l])
    else:
        # ok let's assume gpt-2 encodings by default
        import tiktoken
        print("No meta.pkl found, assuming GPT-2 encodings...")
        enc = tiktoken.get_encoding("gpt2")
        encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
        decode = lambda l: enc.decode(l)

    def run(start: str, max_new_tokens: int = 50, temperature: float = 0.8,
            top_k: int = 200, num_samples=1, seed: int = 1337):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        start_ids = encode(start)
        x = (torch.tensor(start_ids, dtype=torch.long, device="cpu")[None, ...])
        res = []
        with torch.no_grad():
            for _ in range(num_samples):
                y = model.generate(x, max_new_tokens,
                                   temperature=temperature,
                                   top_k=top_k)
                res.append(decode(y[0].tolist()))
        return ("\n" + "-" * 10 + "\n").join(res)

    from mlem.api import save
    # from tiktoken_wrapper import TiktokenEncoder
    save(run, target_path, sample_data="\n")


def main():
    if len(sys.argv) < 3:
        print(
            f"Usage: {sys.argv[0]} <path to checkpoint dir> <model name to save>")
        sys.exit(1)
    convert(sys.argv[1], sys.argv[2])


if __name__ == '__main__':
    main()
