sudo apt-get update && sudo apt-get upgrade -y && sudo apt-get install xorg -y && sudo apt-get install nvidia-driver-460 python3-pip -y && pip install torch tiktoken gitpython numpy transformers dvc-s3 mlem[flyio,fastapi] && curl -L https://fly.io/install.sh | sh && sudo reboot


git clone https://github.com/mike0sv/nanoGPT && cd nanoGPT/ && git checkout -b mlem origin/mlem

# python3 data/mlem-docs/prepare.py

# python3 train.py config/finetune_mlemai.py --dtype=float32 --init_from=gpt2

# python3 sample.py --out_dir=out-mlemai --dtype=float32

# python3 train.py config/train_mlemai.py --device=cuda --compile=False --max_iters=3000 --init_from=scratch --dtype=float32 --dataset=wolf