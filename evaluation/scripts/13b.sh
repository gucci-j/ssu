#!/bin/bash

source /path/to/envs/ssu_lighteval/bin/activate

# Configs
export TRANSFORMERS_VERBOSITY=debug
export HF_HOME="/path/to/cache"
export HF_HUB_CACHE="/path/to/cache"
export HF_DATASETS_CACHE="/path/to/cache"
export HF_DATASETS_TRUST_REMOTE_CODE=true
custom_task_script_dir="~/src/ssu/evaluation/src"
log_base_dir="~/src/ssu/evaluation/logs/adapted"
mkdir -p $log_base_dir
lang_code=$1
approach=$2
checkpoint_steps=$3
postfix=$4
model_name="/path/to/models/OLMo-2-1124-13B-Instruct-${lang_code}-${approach}/checkpoint-${checkpoint_steps}"
model_abbrev=OLMo-2-1124-13B-Instruct
if [[ -z "$lang_code" || -z "$approach" || -z "$checkpoint_steps" || -z "$postfix" ]]; then
    echo "Language code, approach, checkpoint steps, and postfix are required arguments."
    echo "Usage: $0 <lang_code> <approach> <checkpoint_steps> <postfix>"
    exit 1
fi
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

task="custom|sum:en|0|1"
task_name=$(echo $task | cut -d'|' -f2 | cut -d':' -f1)
lighteval accelerate \
    "model_name=${model_name},batch_size=1,dtype=bfloat16" \
    "${task}" \
    --custom-tasks "${custom_task_script_dir}/${task_name}.py" \
    --output-dir="${log_base_dir}/${model_abbrev}/${task_name}" \
    --use-chat-template

task="extended|mt_bench|0|0"
lighteval accelerate \
    "model_name=${model_name},batch_size=1,dtype=bfloat16" \
    "${task}" \
    --custom-tasks "${custom_task_script_dir}/mtbench.py" \
    --save-details \
    --output-dir="${log_base_dir}/${model_abbrev}/mtbench" \
    --use-chat-template

# Check if postfix is 1 or not
if [[ "$postfix" == "1" ]]; then
    task="lighteval|belebele_${lang_code_to_belebele_lang_code[${lang_code}]}_mcf|3|0"
    task_name=$(echo $task | cut -d'|' -f2 | cut -d'_' -f1)
    lighteval accelerate \
        "model_name=${model_name},batch_size=1,dtype=bfloat16" \
        "${task}" \
        --custom-tasks "${custom_task_script_dir}/belebele.py" \
        --output-dir "${log_base_dir}/${model_abbrev}/${task_name}" \
        --use-chat-template

    task="lighteval|gmmlu_${iso639_3_lang_code[${lang_code}]}_mcf:abstract_algebra|5|0,lighteval|gmmlu_${iso639_3_lang_code[${lang_code}]}_mcf:anatomy|5|0,lighteval|gmmlu_${iso639_3_lang_code[${lang_code}]}_mcf:astronomy|5|0,lighteval|gmmlu_${iso639_3_lang_code[${lang_code}]}_mcf:business_ethics|5|0,lighteval|gmmlu_${iso639_3_lang_code[${lang_code}]}_mcf:clinical_knowledge|5|0,lighteval|gmmlu_${iso639_3_lang_code[${lang_code}]}_mcf:college_biology|5|0,lighteval|gmmlu_${iso639_3_lang_code[${lang_code}]}_mcf:college_chemistry|5|0,lighteval|gmmlu_${iso639_3_lang_code[${lang_code}]}_mcf:college_computer_science|5|0,lighteval|gmmlu_${iso639_3_lang_code[${lang_code}]}_mcf:college_mathematics|5|0,lighteval|gmmlu_${iso639_3_lang_code[${lang_code}]}_mcf:college_medicine|5|0,lighteval|gmmlu_${iso639_3_lang_code[${lang_code}]}_mcf:college_physics|5|0,lighteval|gmmlu_${iso639_3_lang_code[${lang_code}]}_mcf:computer_security|5|0,lighteval|gmmlu_${iso639_3_lang_code[${lang_code}]}_mcf:conceptual_physics|5|0,lighteval|gmmlu_${iso639_3_lang_code[${lang_code}]}_mcf:econometrics|5|0,lighteval|gmmlu_${iso639_3_lang_code[${lang_code}]}_mcf:electrical_engineering|5|0,lighteval|gmmlu_${iso639_3_lang_code[${lang_code}]}_mcf:elementary_mathematics|5|0,lighteval|gmmlu_${iso639_3_lang_code[${lang_code}]}_mcf:formal_logic|5|0,lighteval|gmmlu_${iso639_3_lang_code[${lang_code}]}_mcf:global_facts|5|0,lighteval|gmmlu_${iso639_3_lang_code[${lang_code}]}_mcf:high_school_biology|5|0,lighteval|gmmlu_${iso639_3_lang_code[${lang_code}]}_mcf:high_school_chemistry|5|0,lighteval|gmmlu_${iso639_3_lang_code[${lang_code}]}_mcf:high_school_computer_science|5|0,lighteval|gmmlu_${iso639_3_lang_code[${lang_code}]}_mcf:high_school_european_history|5|0,lighteval|gmmlu_${iso639_3_lang_code[${lang_code}]}_mcf:high_school_geography|5|0,lighteval|gmmlu_${iso639_3_lang_code[${lang_code}]}_mcf:high_school_government_and_politics|5|0,lighteval|gmmlu_${iso639_3_lang_code[${lang_code}]}_mcf:high_school_macroeconomics|5|0,lighteval|gmmlu_${iso639_3_lang_code[${lang_code}]}_mcf:high_school_mathematics|5|0,lighteval|gmmlu_${iso639_3_lang_code[${lang_code}]}_mcf:high_school_microeconomics|5|0,lighteval|gmmlu_${iso639_3_lang_code[${lang_code}]}_mcf:high_school_physics|5|0,lighteval|gmmlu_${iso639_3_lang_code[${lang_code}]}_mcf:high_school_psychology|5|0,lighteval|gmmlu_${iso639_3_lang_code[${lang_code}]}_mcf:high_school_statistics|5|0,lighteval|gmmlu_${iso639_3_lang_code[${lang_code}]}_mcf:high_school_us_history|5|0,lighteval|gmmlu_${iso639_3_lang_code[${lang_code}]}_mcf:high_school_world_history|5|0,lighteval|gmmlu_${iso639_3_lang_code[${lang_code}]}_mcf:human_aging|5|0,lighteval|gmmlu_${iso639_3_lang_code[${lang_code}]}_mcf:human_sexuality|5|0,lighteval|gmmlu_${iso639_3_lang_code[${lang_code}]}_mcf:international_law|5|0,lighteval|gmmlu_${iso639_3_lang_code[${lang_code}]}_mcf:jurisprudence|5|0,lighteval|gmmlu_${iso639_3_lang_code[${lang_code}]}_mcf:logical_fallacies|5|0,lighteval|gmmlu_${iso639_3_lang_code[${lang_code}]}_mcf:machine_learning|5|0,lighteval|gmmlu_${iso639_3_lang_code[${lang_code}]}_mcf:management|5|0,lighteval|gmmlu_${iso639_3_lang_code[${lang_code}]}_mcf:marketing|5|0,lighteval|gmmlu_${iso639_3_lang_code[${lang_code}]}_mcf:medical_genetics|5|0,lighteval|gmmlu_${iso639_3_lang_code[${lang_code}]}_mcf:miscellaneous|5|0,lighteval|gmmlu_${iso639_3_lang_code[${lang_code}]}_mcf:moral_disputes|5|0,lighteval|gmmlu_${iso639_3_lang_code[${lang_code}]}_mcf:moral_scenarios|5|0,lighteval|gmmlu_${iso639_3_lang_code[${lang_code}]}_mcf:nutrition|5|0,lighteval|gmmlu_${iso639_3_lang_code[${lang_code}]}_mcf:philosophy|5|0,lighteval|gmmlu_${iso639_3_lang_code[${lang_code}]}_mcf:prehistory|5|0,lighteval|gmmlu_${iso639_3_lang_code[${lang_code}]}_mcf:professional_accounting|5|0,lighteval|gmmlu_${iso639_3_lang_code[${lang_code}]}_mcf:professional_law|5|0,lighteval|gmmlu_${iso639_3_lang_code[${lang_code}]}_mcf:professional_medicine|5|0,lighteval|gmmlu_${iso639_3_lang_code[${lang_code}]}_mcf:professional_psychology|5|0,lighteval|gmmlu_${iso639_3_lang_code[${lang_code}]}_mcf:public_relations|5|0,lighteval|gmmlu_${iso639_3_lang_code[${lang_code}]}_mcf:security_studies|5|0,lighteval|gmmlu_${iso639_3_lang_code[${lang_code}]}_mcf:sociology|5|0,lighteval|gmmlu_${iso639_3_lang_code[${lang_code}]}_mcf:us_foreign_policy|5|0,lighteval|gmmlu_${iso639_3_lang_code[${lang_code}]}_mcf:virology|5|0,lighteval|gmmlu_${iso639_3_lang_code[${lang_code}]}_mcf:world_religions|5|0"
    task_name=gmmlu_${iso639_3_lang_code[${lang_code}]}_mcf
    lighteval accelerate \
        "model_name=${model_name},batch_size=1,dtype=bfloat16" \
        "${task}" \
        --custom-tasks "${custom_task_script_dir}/gmmlu.py" \
        --output-dir "${log_base_dir}/${model_abbrev}/${task_name}" \
        --use-chat-template

    task="lighteval|belebele_eng_Latn_mcf|3|0"
    task_name=$(echo $task | cut -d'|' -f2 | cut -d'_' -f1)
    lighteval accelerate \
        "model_name=${model_name},batch_size=1,dtype=bfloat16" \
        "${task}" \
        --custom-tasks "${custom_task_script_dir}/belebele.py" \
        --output-dir "${log_base_dir}/${model_abbrev}/${task_name}" \
        --use-chat-template

    task="leaderboard|mmlu:abstract_algebra|5|0,leaderboard|mmlu:anatomy|5|0,leaderboard|mmlu:astronomy|5|0,leaderboard|mmlu:business_ethics|5|0,leaderboard|mmlu:clinical_knowledge|5|0,leaderboard|mmlu:college_biology|5|0,leaderboard|mmlu:college_chemistry|5|0,leaderboard|mmlu:college_computer_science|5|0,leaderboard|mmlu:college_mathematics|5|0,leaderboard|mmlu:college_medicine|5|0,leaderboard|mmlu:college_physics|5|0,leaderboard|mmlu:computer_security|5|0,leaderboard|mmlu:conceptual_physics|5|0,leaderboard|mmlu:econometrics|5|0,leaderboard|mmlu:electrical_engineering|5|0,leaderboard|mmlu:elementary_mathematics|5|0,leaderboard|mmlu:formal_logic|5|0,leaderboard|mmlu:global_facts|5|0,leaderboard|mmlu:high_school_biology|5|0,leaderboard|mmlu:high_school_chemistry|5|0,leaderboard|mmlu:high_school_computer_science|5|0,leaderboard|mmlu:high_school_european_history|5|0,leaderboard|mmlu:high_school_geography|5|0,leaderboard|mmlu:high_school_government_and_politics|5|0,leaderboard|mmlu:high_school_macroeconomics|5|0,leaderboard|mmlu:high_school_mathematics|5|0,leaderboard|mmlu:high_school_microeconomics|5|0,leaderboard|mmlu:high_school_physics|5|0,leaderboard|mmlu:high_school_psychology|5|0,leaderboard|mmlu:high_school_statistics|5|0,leaderboard|mmlu:high_school_us_history|5|0,leaderboard|mmlu:high_school_world_history|5|0,leaderboard|mmlu:human_aging|5|0,leaderboard|mmlu:human_sexuality|5|0,leaderboard|mmlu:international_law|5|0,leaderboard|mmlu:jurisprudence|5|0,leaderboard|mmlu:logical_fallacies|5|0,leaderboard|mmlu:machine_learning|5|0,leaderboard|mmlu:management|5|0,leaderboard|mmlu:marketing|5|0,leaderboard|mmlu:medical_genetics|5|0,leaderboard|mmlu:miscellaneous|5|0,leaderboard|mmlu:moral_disputes|5|0,leaderboard|mmlu:moral_scenarios|5|0,leaderboard|mmlu:nutrition|5|0,leaderboard|mmlu:philosophy|5|0,leaderboard|mmlu:prehistory|5|0,leaderboard|mmlu:professional_accounting|5|0,leaderboard|mmlu:professional_law|5|0,leaderboard|mmlu:professional_medicine|5|0,leaderboard|mmlu:professional_psychology|5|0,leaderboard|mmlu:public_relations|5|0,leaderboard|mmlu:security_studies|5|0,leaderboard|mmlu:sociology|5|0,leaderboard|mmlu:us_foreign_policy|5|0,leaderboard|mmlu:virology|5|0,leaderboard|mmlu:world_religions|5|0"
    task_name="mmlu"
    lighteval accelerate \
        "model_name=${model_name},batch_size=1,dtype=bfloat16" \
        "${task}" \
        --output-dir "${log_base_dir}/${model_abbrev}/${task_name}" \
        --use-chat-template
fi

# Move files
cd ~/src/ssu/evaluation/src/utils
python move_result_files.py ${model_name} ${log_base_dir}

deactivate

#####
source /path/to/envs/ssu_lmeval/bin/activate
log_base_dir="~/src/ssu/evaluation/logs_lmeval/adapted"
mkdir -p $log_base_dir

if [ "$model_abbrev" == "OLMo-2-1124-7B-Instruct" ]; then
    system_instruction=""
elif [ "$model_abbrev" == "OLMo-2-1124-13B-Instruct" ]; then
    system_instruction=""
else
    echo "Unsupported model abbrev: $model_abbrev"
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

model_name="/path/to/models/OLMo-2-1124-13B-Instruct-${lang_code}-${approach}/checkpoint-${checkpoint_steps}"
model_abbrev="OLMo-2-1124-13B-Instruct-${lang_code}-${approach}__checkpoint-${checkpoint_steps}"
mkdir -p "~/src/ssu/evaluation/logs_ae2/adapted/$model_abbrev"

# Run evaluation
cd ~/src/ssu/evaluation/src
python ae2.py \
    --model_name_or_path $model_name \
    --model_abbrev $model_abbrev \
    --annotators_config "alpaca_eval_gpt4.1-nano.yml" \
    --output_dir "~/src/ssu/evaluation/logs_ae2/adapted/$model_abbrev" \
    --batch_size 4 \
    --postfix "$postfix"
