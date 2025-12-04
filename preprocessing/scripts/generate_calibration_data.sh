#!/bin/bash

source /path/to/envs/ssu_train/bin/activate

export TRANSFORMERS_VERBOSITY=debug
export HF_HOME="/path/to/cache/"
export HF_HUB_CACHE="/path/to/cache/"
export HF_DATASETS_CACHE="/path/to/cache/"
export HF_DATASETS_TRUST_REMOTE_CODE=true

model_name=$1
if [ -z "$model_name" ]; then
    echo "Usage: $0 <model_name>"
    echo "Example: $0 allenai/OLMo-2-1124-7B-Instruct"
    exit 1
fi
if [ "$model_name" == "allenai/OLMo-2-1124-7B-Instruct" ]; then
    model_abbrev="OLMo-2-1124-7B-Instruct"
else
    echo "Unsupported model name: $model_name"
    exit 1
fi

cd ~/src/ssu/preprocessing/src
python generate_calibration_data.py \
    --output_dir "/path/to/processed/data/${model_abbrev}_calib" \
    --cache_dir "/path/to/cache/" \
    --dataset_name allenai/tulu-3-sft-olmo-2-mixture \
    --split train \
    --num_samples 2000 \
    --tokenizer_name_or_path "${model_name}" \
    --num_workers 8 \
    --block_size 2048 \
    --shuffle \
    --streaming
