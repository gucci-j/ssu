#!/bin/bash

source /path/to/envs/ssu_train/bin/activate

export TRANSFORMERS_VERBOSITY=debug
export HF_HOME="/path/to/cache/"
export HF_HUB_CACHE="/path/to/cache/"
export HF_DATASETS_CACHE="/path/to/cache/"
export HF_DATASETS_TRUST_REMOTE_CODE=true

model_name=$1
lang_code=$2
if [ -z "$model_name" ] || [ -z "$lang_code" ]; then
    echo "Usage: $0 <model_name> <lang_code>"
    echo "Example: $0 allenai/OLMo-2-1124-7B-Instruct amh_Ethi"
    exit 1
fi
if [ "$lang_code" == "amh_Ethi" ]; then
    short_lang_code="am"
elif [ "$lang_code" == "hau_Latn" ]; then
    short_lang_code="ha"
elif [ "$lang_code" == "ibo_Latn" ]; then
    short_lang_code="ig"
elif [ "$lang_code" == "npi_Deva" ]; then
    short_lang_code="ne"
elif [ "$lang_code" == "kir_Cyrl" ]; then
    short_lang_code="ky"
else
    echo "Unsupported language code: $lang_code"
    exit 1
fi
if [ "$model_name" == "allenai/OLMo-2-1124-7B-Instruct" ]; then
    model_abbrev="OLMo-2-1124-7B-Instruct"
else
    echo "Unsupported model name: $model_name"
    exit 1
fi

cd ~/src/ssu/preprocessing/src
python generate_cpt_data.py \
    --lang_code $short_lang_code \
    --output_dir "/path/to/processed/data/${model_abbrev}_${short_lang_code}" \
    --cache_dir "/path/to/cache/" \
    --tokenizer_name_or_path "${model_name}" \
    --num_workers 31 \
    --max_length 512
