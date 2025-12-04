#!/bin/bash

source /path/to/envs/ssu_lighteval/bin/activate

# Configs
export OPENAI_API_KEY="your_openai_api_key"
export TRANSFORMERS_VERBOSITY=debug
export HF_HOME="/path/to/cache"
export HF_HUB_CACHE="/path/to/cache"
export HF_DATASETS_CACHE="/path/to/cache"
export HF_DATASETS_TRUST_REMOTE_CODE=true

model_name=$1
postfix=$2
if [ -z "$model_name" ] || [ -z "$postfix" ]; then
    echo "Usage: $0 <model_name> <postfix>"
    exit 1
fi
if [ "$model_name" == "allenai/OLMo-2-1124-7B-Instruct" ]; then
    model_abbrev="OLMo-2-1124-7B-Instruct"
elif [ "$model_name" == "allenai/OLMo-2-1124-13B-Instruct" ]; then
    model_abbrev="OLMo-2-1124-13B-Instruct"
else
    echo "Unsupported model name: $model_name"
    exit 1
fi

# Run evaluation
cd ~/src/ssu/evaluation/src
python ae2.py \
    --model_name_or_path $model_name \
    --model_abbrev $model_abbrev \
    --annotators_config "alpaca_eval_gpt4.1-nano.yml" \
    --output_dir "~/src/ssu/evaluation/logs_ae2/post/$model_abbrev" \
    --batch_size 4 \
    --skip_inference \
    --do_eval \
    --postfix $postfix
