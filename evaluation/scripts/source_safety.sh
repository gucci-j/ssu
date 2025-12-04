#!/bin/bash

source /path/to/envs/ssu_vllm/bin/activate

# Configs
export OPENAI_API_KEY="your_openai_api_key"
export TRANSFORMERS_VERBOSITY=debug
export HF_HOME="/path/to/cache"
export HF_HUB_CACHE="/path/to/cache"
export HF_DATASETS_CACHE="/path/to/cache"
export HF_DATASETS_TRUST_REMOTE_CODE=true

model_name=$1
if [ -z "$model_name" ]; then
    echo "Usage: $0 <model_name>"
    echo "Example: $0 allenai/OLMo-2-1124-7B-Instruct"
    exit 1
fi
model_abbrev=$(cut -d'/' -f2 <<< $model_name)

log_dir="~/src/ssu/evaluation/logs_safety/post/${model_abbrev}"
mkdir -p $log_dir

cd ~/src/safety-eval-fork
python evaluation/eval.py generators \
  --use_vllm \
  --model_name_or_path $model_name \
  --model_input_template_path_or_name tulu2 \
  --tasks wildguardtest,harmbench,xstest,trustllm_jailbreaktrigger,do_anything_now,wildjailbreak:benign,wildjailbreak:harmful \
  --report_output_path $log_dir/metrics.json
