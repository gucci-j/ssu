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
dataset_dir="/path/to/processed/data/${model_abbrev}_${lang_code}/"
output_dir="/path/to/models/${model_abbrev}-${lang_code}-s2ft_down_16/"
logging_dir="/path/to/logs/${model_abbrev}-${lang_code}-s2ft_down_16/"
model_name_or_path="allenai/OLMo-2-1124-7B-Instruct"

cd ~/src/ssu/training/src
python main.py \
    --dataset_path "${dataset_dir}" \
    --output_dir "${output_dir}" \
    --logging_dir "${logging_dir}" \
    --model_name_or_path "${model_name_or_path}" \
    --tokenizer_name_or_path "${model_name_or_path}" \
    --optim adamw_apex_fused \
    --seed 42 \
    --eval_strategy no \
    --logging_steps 0.001 \
    --learning_rate 5e-5 \
    --weight_decay 0.01 \
    --warmup_ratio 0.05 \
    --max_steps 12208 \
    --per_device_train_batch_size 32 \
    --gradient_accumulation_steps 1 \
    --prediction_loss_only \
    --overwrite_output_dir \
    --do_train \
    --lr_scheduler_type cosine \
    --disable_tqdm True \
    --label_names labels \
    --remove_unused_columns True \
    --save_strategy steps \
    --save_steps 0.5 \
    --bf16 \
    --gradient_checkpointing True \
    --use_s2ft \
    --s2ft_ratio 0.03 \
    --s2ft_strategy random
