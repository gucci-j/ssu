import pandas as pd
from datasets import load_dataset

def main(args):
    # Load the FLORES data for 5 languages
    lang_code_to_flores_code = {
        "en": "eng_Latn",
        "am": "amh_Ethi",
        "ne": "npi_Deva",
        "ha": "hau_Latn",
        "ky": "kir_Cyrl",
        "ig": "ibo_Latn",
    }

    # Create the dev dataset
    results = {}
    for lang_code, flores_code in lang_code_to_flores_code.items():
        ds = load_dataset("openlanguagedata/flores_plus", flores_code, split="dev")
        if results == {}:
            results = {line["id"]: {lang_code: line["text"]} for line in ds}
        else:
            for line in ds:
                if line["id"] in results:
                    results[line["id"]][lang_code] = line["text"]
                else:
                    results[line["id"]] = {lang_code: line["text"]}
    # Convert into DataFrame
    results = [
        translations
        for _, translations in results.items()
    ]
    dev_df = pd.DataFrame(results)
    # Save the data as jsonl
    dev_df.to_json(args.output_dir + f"/dev.jsonl", lines=True, orient="records", force_ascii=False)

    # Create the test dataset
    results = {}
    for lang_code, flores_code in lang_code_to_flores_code.items():
        ds = load_dataset("openlanguagedata/flores_plus", flores_code, split="devtest")
        if results == {}:
            results = {line["id"]: {lang_code: line["text"]} for line in ds}
        else:
            for line in ds:
                if line["id"] in results:
                    results[line["id"]][lang_code] = line["text"]
                else:
                    results[line["id"]] = {lang_code: line["text"]}
    # Convert into DataFrame
    results = [
        translations
        for _, translations in results.items()
    ]
    devtest_df = pd.DataFrame(results)
    # Sample 500 examples
    devtest_df = devtest_df.sample(n=500, random_state=42)
    # Save the data as jsonl
    devtest_df.to_json(args.output_dir + f"/test.jsonl", lines=True, orient="records", force_ascii=False)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Create HF datasets')
    parser.add_argument(
        "--cache_dir",
        type=str,
        help="The directory to save the downloaded files",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="The directory to save the output files",
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        help="The name of the repository to which the data will be uploaded",
    )
    args = parser.parse_args()
    main(args)


    from huggingface_hub import HfApi
    api = HfApi()
    try:
        api.create_repo(
            repo_id=args.repo_id, 
            private=True,
            repo_type='dataset',
        )
    except Exception:
        pass
    api.upload_folder(
        folder_path=args.output_dir,
        repo_id=args.repo_id,
        repo_type='dataset',
    )
