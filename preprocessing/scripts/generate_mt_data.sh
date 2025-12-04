#!/bin/bash

source /path/to/envs/ssu_train/bin/activate

export TRANSFORMERS_VERBOSITY=debug
export HF_HOME="/path/to/cache/"
export HF_HUB_CACHE="/path/to/cache/"
export HF_DATASETS_CACHE="/path/to/cache/"
export HF_DATASETS_TRUST_REMOTE_CODE=true

cd ~/src/ssu/preprocessing/src
mkdir -p /path/to/outputs/

python generate_mt_data.py \
    --output_dir "/path/to/outputs" \
    --cache_dir "/path/to/cache/" \
    --repo_id your-hf-id/flores-ssu
