import importlib
import os
import sys

import modal

DATA_DIR = "./data/mlem-docs"
OUTPUT_DIR = "out-mlemai-char"
stub = modal.Stub("nano-gpt-train")

src_mount = modal.Mount("/root/", local_dir=".",
                        condition=lambda x: x.endswith(".py"))
data_mount = modal.Mount(os.path.join("/root", DATA_DIR), local_dir=DATA_DIR,
                         condition=lambda x: any(x.endswith(e) for e in
                                                 ["train.bin", "val.bin",
                                                  "meta.pkl"]))
checkpoint_volume = modal.SharedVolume().persist("nanogpt-ckpt")


@stub.function(
    mounts=[src_mount, data_mount],
    gpu="A10G",
    image=(
            modal.Image.debian_slim()
            .run_commands(
                "pip install torch --extra-index-url https://download.pytorch.org/whl/cu117"
            )
            .pip_install("tiktoken", "numpy", "transformers")
    ),
    shared_volumes={os.path.join("/root", OUTPUT_DIR): checkpoint_volume},
    timeout=86400
)
def train(config_file: str = None, **kwargs):
    kwargs["out_dir"] = OUTPUT_DIR
    # prepare argv for train.py
    sys.argv = [sys.argv[0]] + (
        [] if config_file is None else [config_file]) + [f"--{name}={value}" for
                                                         name, value in
                                                         kwargs.items()]
    # run train.py
    importlib.import_module("train")


def main():
    with stub.run():
        train.call("config/train_mlemai.py", device="cuda", dtype="float32",
                   max_iters=5000, init_from="scratch")
        print("Done, downloading checkpoint")
        vol = modal.lookup("nanogpt-ckpt")
        with open(os.path.join(OUTPUT_DIR, "ckpt.pt"), "wb") as fout:
            for chunk in vol.read_file("ckpt.pt"):
                fout.write(chunk)
    print("To delete shared volume with checkpoint file, run "
          "`modal volume nanogpt-ckpt remove`")


if __name__ == '__main__':
    main()
