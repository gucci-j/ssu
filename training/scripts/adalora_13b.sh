#!/bin/bash

source /path/to/envs/ssu_train/bin/activate

# Configs
export TRANSFORMERS_VERBOSITY=debug
export HF_HOME="/path/to/cache"
export HF_HUB_CACHE="/path/to/cache"
export HF_DATASETS_CACHE="/path/to/cache"
export HF_DATASETS_TRUST_REMOTE_CODE=true
model_abbrev="OLMo-2-1124-13B-Instruct"
data_model_abbrev="OLMo-2-1124-7B-Instruct"
lang_code="$1"
dataset_dir="/path/to/processed/data/${data_model_abbrev}_${lang_code}/"
output_dir="/path/to/models/${model_abbrev}-${lang_code}-adalora/"
logging_dir="/path/to/logs/${model_abbrev}-${lang_code}-adalora/"
model_name_or_path="allenai/OLMo-2-1124-13B-Instruct"

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
    --peft_method adalora \
    --lora_r 12 \
    --adalora_target_r 8 \
    --lora_alpha 32 \
    --lora_dropout 0.05 \
    --lora_target_modules q_proj,k_proj,v_proj,o_proj,up_proj,down_proj,gate_proj \
    --adalora_tinit 1000 \
    --adalora_total_step 12208 \
    --adalora_tfinal 8546 \
    --adalora_delta_t 20 \
    --adalora_beta1 0.85 \
    --adalora_beta2 0.85 \
    --adalora_orth_reg_weight 0.5 \
    --peft_bias none
