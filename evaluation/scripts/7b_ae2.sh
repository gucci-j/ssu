#!/bin/bash

source /path/to/envs/ssu_lighteval/bin/activate

# Configs
export OPENAI_API_KEY="your_openai_api_key"
export TRANSFORMERS_VERBOSITY=debug
export HF_HOME="/path/to/cache"
export HF_HUB_CACHE="/path/to/cache"
export HF_DATASETS_CACHE="/path/to/cache"
export HF_DATASETS_TRUST_REMOTE_CODE=true

lang_code=$1
approach=$2
checkpoint_steps=$3
postfix=$4
if [ -z "$lang_code" ] || [ -z "$approach" ] || [ -z "$checkpoint_steps" ] || [ -z "$postfix" ]; then
    echo "Usage: $0 <lang_code> <approach> <checkpoint_steps> <postfix>"
    exit 1
fi

model_name="/path/to/models/OLMo-2-1124-7B-Instruct-${lang_code}-${approach}/checkpoint-${checkpoint_steps}"
model_abbrev="OLMo-2-1124-7B-Instruct-${lang_code}-${approach}__checkpoint-${checkpoint_steps}"

# Run evaluation
cd ~/src/ssu/evaluation/src
python ae2.py \
    --model_name_or_path $model_name \
    --model_abbrev $model_abbrev \
    --annotators_config "alpaca_eval_gpt4.1-nano.yml" \
    --output_dir "~/src/ssu/evaluation/logs_ae2/adapted/$model_abbrev" \
    --postfix "$postfix" \
    --batch_size 4 \
    --skip_inference \
    --do_eval
