"""
Setup instructions:

!pip install fasttext
!pip install huggingface_hub
!pip install pandas
!pip install numpy==1.26.4

This script analyzes the language ratio in model outputs using GlotLID.
"""

import fasttext
from huggingface_hub import hf_hub_download
from pathlib import Path
import json
import numpy as np
import pandas as pd


class CustomLID:
    def __init__(self, model_path, languages = -1, mode='before'):
        self.model = fasttext.load_model(model_path)
        self.output_matrix = self.model.get_output_matrix()
        self.labels = self.model.get_labels()
        
        # compute language_indices
        if languages !=-1 and isinstance(languages, list):
            self.language_indices = [self.labels.index(l) for l in list(set(languages)) if l in self.labels]

        else:
            self.language_indices = list(range(len(self.labels)))

        # limit labels to language_indices
        self.labels = list(np.array(self.labels)[self.language_indices])
        
        # predict
        self.predict = self.predict_limit_after_softmax if mode=='after' else self.predict_limit_before_softmax

    
    def predict_limit_before_softmax(self, text, k=1):
        
        # sentence vector
        sentence_vector = self.model.get_sentence_vector(text)
        
        # dot
        result_vector = np.dot(self.output_matrix[self.language_indices, :], sentence_vector)

        # softmax
        softmax_result = np.exp(result_vector - np.max(result_vector)) / np.sum(np.exp(result_vector - np.max(result_vector)))

        # top k predictions
        top_k_indices = np.argsort(softmax_result)[-k:][::-1]
        top_k_labels = [self.labels[i] for i in top_k_indices]
        top_k_probs = softmax_result[top_k_indices]

        return tuple(top_k_labels), top_k_probs


    def predict_limit_after_softmax(self, text, k=1):
        
        # sentence vector
        sentence_vector = self.model.get_sentence_vector(text)
        
        # dot
        result_vector = np.dot(self.output_matrix, sentence_vector)

        # softmax
        softmax_result = np.exp(result_vector - np.max(result_vector)) / np.sum(np.exp(result_vector - np.max(result_vector)))

        # limit softmax to language_indices
        softmax_result = softmax_result[self.language_indices]

        
        # top k predictions
        top_k_indices = np.argsort(softmax_result)[-k:][::-1]
        top_k_labels = [self.labels[i] for i in top_k_indices]
        top_k_probs = softmax_result[top_k_indices]

        return tuple(top_k_labels), top_k_probs


def main(args):
    # download model
    model_path = hf_hub_download(repo_id="cis-lmu/glotlid", filename="model.bin")
    print("Model path:", model_path)

    # analyze results
    log_dir = Path(args.log_dir)
    results = []
    for result_path in log_dir.glob("adapted/**/model_outputs_*.json"):
        print("Processing:", result_path)
        with open(result_path) as f:
            data = json.load(f)
        target_lang = result_path.parent.name.split("__")[0].split("-")[-3]
        if result_path.parent.name.startswith("OLMo-2-1124-7B-Instruct"):
            base_model = "OLMo-2-1124-7B-Instruct"
        elif result_path.parent.name.startswith("OLMo-2-1124-13B-Instruct"):
            base_model = "OLMo-2-1124-13B-Instruct"
        else:
            continue
        if target_lang == "am":
            limited_languages = ['__label__eng_Latn', "__label__amh_Ethi"]
        elif target_lang == "ne":
            limited_languages = ['__label__eng_Latn', "__label__npi_Deva"]
        elif target_lang == "ky":
            limited_languages = ['__label__eng_Latn', "__label__kir_Cyrl"]
        elif target_lang == "ig":
            limited_languages = ['__label__eng_Latn', "__label__ibo_Latn"]
        elif target_lang == "ha":
            limited_languages = ['__label__eng_Latn', "__label__hau_Latn"]
        else:
            continue
        model = CustomLID(model_path, languages=limited_languages , mode='before')
        num_mixed = 0
        for sample in data:
            lang_label = model.predict(sample["output"].replace("\n", " "))
            if lang_label[0][0] == "__label__eng_Latn":
                if lang_label[1][0] < 0.9:
                    num_mixed += 1
            else:
                num_mixed += 1
        del model
        results.append({
            "approach": result_path.parent.name.split("__")[0].split("-")[-2],
            "language": result_path.parent.name.split("__")[0].split("-")[-3],
            "base_model": base_model,
            "num_mixed": num_mixed,
        })
    df = pd.DataFrame(results)

    # print results
    agg_df = df.groupby(["approach", "language", "base_model"]).agg("mean").reset_index()
    for approach in (
        "hft",
        "gmt",
        "ssu",
    ):
        print(approach, agg_df[(agg_df.base_model == "OLMo-2-1124-7B-Instruct") & (agg_df.approach == approach)]["num_mixed"].mean() / 805.0)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", type=str, default="~/src/ssu/evaluation/logs_ae2")
    args = parser.parse_args()
    main(args)
