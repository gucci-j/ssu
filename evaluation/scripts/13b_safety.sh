#!/bin/bash

source /path/to/envs/ssu_vllm/bin/activate

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
if [ -z "$lang_code" ] || [ -z "$approach" ] || [ -z "$checkpoint_steps" ]; then
    echo "Usage: $0 <lang_code> <approach> <checkpoint_steps>"
    exit 1
fi
# if approach is adalora, append -merged to checkpoint_steps
if [ "$approach" == "adalora" ]; then
  model_name="/path/to/models/OLMo-2-1124-13B-Instruct-${lang_code}-${approach}/checkpoint-${checkpoint_steps}-merged"
else
  model_name="/path/to/models/OLMo-2-1124-13B-Instruct-${lang_code}-${approach}/checkpoint-${checkpoint_steps}"
fi
echo "Evaluating model: $model_name"
model_abbrev="OLMo-2-1124-13B-Instruct-${lang_code}-${approach}__checkpoint-${checkpoint_steps}"

log_dir="~/src/ssu/evaluation/logs_safety/adapted/${model_abbrev}"
mkdir -p $log_dir

cd ~/src/safety-eval-fork
python evaluation/eval.py generators \
  --use_vllm \
  --model_name_or_path $model_name \
  --model_input_template_path_or_name tulu2 \
  --tasks wildguardtest,harmbench,xstest,trustllm_jailbreaktrigger,do_anything_now,wildjailbreak:benign,wildjailbreak:harmful \
  --report_output_path $log_dir/metrics.json
