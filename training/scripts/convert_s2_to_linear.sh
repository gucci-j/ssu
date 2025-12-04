#!/bin/bash

source /path/to/envs/ssu_train/bin/activate

# Configs
export TRANSFORMERS_VERBOSITY=debug
export HF_HOME="/path/to/cache"
export HF_HUB_CACHE="/path/to/cache"
export HF_DATASETS_CACHE="/path/to/cache"
export HF_DATASETS_TRUST_REMOTE_CODE=true
model_abbrev="OLMo-2-1124-7B-Instruct"
lang_code="$1"
approach="$2"
output_dir="/path/to/models/${model_abbrev}-${lang_code}-${approach}/checkpoint-12208"
model_name_or_path="allenai/OLMo-2-1124-7B-Instruct"

cd ~/src/ssu/training/src
python utils/convert_s2_to_linear.py \
    --input "${output_dir}" \
    --output "${output_dir}_converted/" \
    --dtype bf16 \
    --reconstruct-s2 \
    --selections ~/src/ssu/training/src/${approach}.json
