#!/bin/bash

source /path/to/envs/ssu_lighteval/bin/activate

# Configs
export TRANSFORMERS_VERBOSITY=debug
export HF_HOME="/path/to/cache/"
export HF_HUB_CACHE="/path/to/cache/"
export HF_DATASETS_CACHE="/path/to/cache/"
export HF_DATASETS_TRUST_REMOTE_CODE=true
custom_task_script_dir="~/src/ssu/evaluation/src"
log_base_dir="~/src/ssu/evaluation/logs/post"
mkdir -p "${log_base_dir}"
model_name=$1
postfix=$2
if [[ -z "$model_name" || -z "$postfix" ]]; then
    echo "Model name and postfix are required arguments."
    echo "Usage: $0 <model_name> <postfix>"
    exit 1
fi
model_abbrev=$(cut -d'/' -f2 <<< $model_name)
lang_codes=(
    "am"
    "ha"
    "ig"
    "ky"
    "ne"
)
declare -A lang_code_to_belebele_lang_code=(
    ["ne"]="npi_Deva"
    ["am"]="amh_Ethi"
    ["ha"]="hau_Latn"
    ["ig"]="ibo_Latn"
    ["ky"]="kir_Cyrl"
)
declare -A iso639_3_lang_code=(
    ["ne"]="npi"
    ["am"]="amh"
    ["ha"]="hau"
    ["ig"]="ibo"
    ["ky"]="kir"
)

for lang_code in "${lang_codes[@]}"; do
    tasks=(
        "custom|mt:en2${lang_code}|3|0"
        "custom|mt:${lang_code}2en|3|0"
        "custom|sum:${lang_code}|0|1"
    )
    for task in "${tasks[@]}"; do
        task_name=$(echo $task | cut -d'|' -f2 | cut -d':' -f1)
        lighteval accelerate \
            "model_name=${model_name},batch_size=1,dtype=bfloat16" \
            "${task}" \
            --custom-tasks "${custom_task_script_dir}/${task_name}.py" \
            --save-details \
            --output-dir="${log_base_dir}/${model_abbrev}/${task_name}" \
            --use-chat-template
    done
done

task="custom|sum:en|0|1"
task_name=$(echo $task | cut -d'|' -f2 | cut -d':' -f1)
lighteval accelerate \
    "model_name=${model_name},batch_size=1,dtype=bfloat16" \
    "${task}" \
    --save-details \
    --custom-tasks "${custom_task_script_dir}/${task_name}.py" \
    --output-dir="${log_base_dir}/${model_abbrev}/${task_name}" \
    --use-chat-template

task="extended|mt_bench|0|0"
lighteval accelerate \
    "model_name=${model_name},batch_size=1,dtype=bfloat16" \
    "${task}" \
    --custom-tasks "${custom_task_script_dir}/mtbench.py" \
    --output-dir="${log_base_dir}/${model_abbrev}/mtbench" \
    --use-chat-template

deactivate

#####
source /path/to/envs/ssu_lmeval/bin/activate
log_base_dir="~/src/ssu/evaluation/logs_lmeval/post"
mkdir -p "${log_base_dir}"

if [ "$model_name" == "allenai/OLMo-2-1124-7B-Instruct" ]; then
    system_instruction=""
elif [ "$model_name" == "allenai/OLMo-2-1124-13B-Instruct" ]; then
    system_instruction=""
else
    echo "Unsupported model name: $model_name"
    exit 1
fi

lm-eval --model hf \
    --model_args=pretrained=${model_name},dtype=bfloat16 \
    --tasks=leaderboard_ifeval \
    --batch_size=1 \
    --output_path="${log_base_dir}/${model_abbrev}" \
    --num_fewshot 0 \
    --apply_chat_template \
    --fewshot_as_multiturn \
    --system_instruction "${system_instruction}" \
    --gen_kwargs "temperature=0.8,top_p=0.8,top_k=40,repetition_penalty=1.1,do_sample=true"

lm-eval --model hf \
    --model_args=pretrained=${model_name},dtype=bfloat16 \
    --tasks=gsm8k \
    --batch_size=1 \
    --output_path="${log_base_dir}/${model_abbrev}" \
    --num_fewshot 5 \
    --apply_chat_template \
    --fewshot_as_multiturn \
    --system_instruction "${system_instruction}" \
    --gen_kwargs "temperature=0.8,top_p=0.8,top_k=40,repetition_penalty=1.1,do_sample=true"

deactivate


#####
source /path/to/envs/ssu_lighteval/bin/activate
export OPENAI_API_KEY="your_openai_api_key"

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
mkdir -p "~/src/ssu/evaluation/logs_ae2/post/$model_abbrev"
python ae2.py \
    --model_name_or_path $model_name \
    --model_abbrev $model_abbrev \
    --annotators_config "alpaca_eval_gpt4.1-nano.yml" \
    --output_dir "~/src/ssu/evaluation/logs_ae2/post/$model_abbrev" \
    --batch_size 4 \
    --postfix "$postfix"
