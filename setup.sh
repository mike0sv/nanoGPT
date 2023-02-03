# setup
sudo apt-get update && sudo apt-get upgrade -y && sudo apt-get install xorg -y && sudo apt-get install nvidia-driver-460 python3-pip -y && pip install torch tiktoken gitpython numpy transformers dvc-s3 mlem[flyio,fastapi,streamlit] && curl -L https://fly.io/install.sh | sh && sudo reboot

# clone repo
git clone https://github.com/mike0sv/nanoGPT && cd nanoGPT/ && git checkout -b mlem origin/mlem

# prepare data
python3 data/mlem-docs/prepare.py char

# train model
python3 train.py config/train_mlemai.py --device=cuda --dtype=float32 --max_iters=5000 --init_from=scratch

# sample model
python3 sample.py --out_dir=out-mlemai-char --dtype=float32

# save weights to DVC
python3 -m dvc init
python3 -m dvc remote add s3 s3://mlem-nanogpt
python3 -m dvc remote default s3
python3 -m dvc add out-mlemai-char/ckpt.pt
python3 -m dvc push


# wrap model with mlem
python3 wrapper.py out-mlemai-char mlem_char
# deploy model
export FLYCTL_INSTALL="/home/ubuntu/.fly"
export PATH="$FLYCTL_INSTALL/bin:$PATH"
flyctl auth login
mlem deploy run flyio app -m mlem_char --app_name mlem-nanogpt --server streamlit --scale_memory 1024 --server.ui_port 8080 --server.server_port 8081 --server.template app.py


