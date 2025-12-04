#!/bin/bash

source /path/to/envs/ssu_lighteval/bin/activate

# Configs
export TRANSFORMERS_VERBOSITY=debug
export HF_HOME="/path/to/cache"
export HF_HUB_CACHE="/path/to/cache"
export HF_DATASETS_CACHE="/path/to/cache"
export HF_DATASETS_TRUST_REMOTE_CODE=true
custom_task_script_dir="~/src/ssu/evaluation/src"
log_base_dir="~/src/ssu/evaluation/logs/post"
mkdir -p "${log_base_dir}"
model_name=$1
if [[ -z "$model_name" ]]; then
    echo "Usage: $0 <model_name>"
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
done

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

deactivate
