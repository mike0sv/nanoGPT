import glob
import os
import pickle
import shutil

import git
import requests
import tiktoken
import numpy as np

FILE_NAME = 'input.txt'
input_file_path = os.path.join(os.path.dirname(__file__), FILE_NAME)
repo_path = os.path.join(os.path.dirname(__file__), "mlem.ai")
def download():
    if os.path.exists(input_file_path):
        return
    if not os.path.exists(repo_path):
        git.Repo.clone_from("https://github.com/iterative/mlem.ai/", "mlem.ai")
    else:
        git.Repo(repo_path).remote("origin").pull()

    with open(input_file_path, 'w') as f:
        for filename in glob.glob(repo_path + "/content/docs/**/*.md"):
            print(os.path.relpath(filename, repo_path))
            f.write(os.path.relpath(filename, repo_path) + "\n")
            with open(filename ,"r") as docfile:
                shutil.copyfileobj(docfile, f)


def tokenize():
    with open(input_file_path, 'r') as f:
        data = f.read()
    n = len(data)
    train_data = data[:int(n*0.9)]
    val_data = data[int(n*0.9):]

    # encode with tiktoken gpt2 bpe
    enc = tiktoken.get_encoding("gpt2")
    train_ids = enc.encode_ordinary(train_data)
    val_ids = enc.encode_ordinary(val_data)
    print(f"train has {len(train_ids):,} tokens")
    print(f"val has {len(val_ids):,} tokens")

    # export to bin files
    train_ids = np.array(train_ids, dtype=np.uint16)
    val_ids = np.array(val_ids, dtype=np.uint16)
    train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
    val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))

    # train has 37,672 tokens
    # val has 4,226 tokens


def tokenize_char():
    with open(input_file_path, 'r') as f:
        data = f.read()
    print(f"length of dataset in characters: {len(data):,}")

    # get all the unique characters that occur in this text
    chars = sorted(list(set(data)))
    vocab_size = len(chars)
    print("all the unique characters:", ''.join(chars))
    print(f"vocab size: {vocab_size:,}")

    # create a mapping from characters to integers
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}

    def encode(s):
        return [stoi[c] for c in
                s]  # encoder: take a string, output a list of integers

    def decode(l):
        ''.join([itos[i] for i in
                 l])  # decoder: take a list of integers, output a string

    # create the train and test splits
    n = len(data)
    train_data = data[:int(n * 0.9)]
    val_data = data[int(n * 0.9):]

    # encode both to integers
    train_ids = encode(train_data)
    val_ids = encode(val_data)
    print(f"train has {len(train_ids):,} tokens")
    print(f"val has {len(val_ids):,} tokens")

    # export to bin files
    train_ids = np.array(train_ids, dtype=np.uint16)
    val_ids = np.array(val_ids, dtype=np.uint16)
    train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
    val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))

    # save the meta information as well, to help us encode/decode later
    meta = {
        'vocab_size': vocab_size,
        'itos': itos,
        'stoi': stoi,
    }
    with open(os.path.join(os.path.dirname(__file__), 'meta.pkl'), 'wb') as f:
        pickle.dump(meta, f)

def main():
    download()
    tokenize()
    # tokenize_char()


if __name__ == '__main__':
    main()
