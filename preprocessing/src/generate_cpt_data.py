import json
from functools import partial

from datasets import Dataset, load_dataset
from transformers import AutoTokenizer

def group_texts(examples: dict, block_size=128):
    # Concatenate all texts.
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
    # customize this part to your needs.
    if total_length >= block_size:
        total_length = (total_length // block_size) * block_size
    
    # Split by chunks of block_size.
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result


def generator_from_iterable_dataset(ds):
    yield from ds


def main(args):
    # Load the dataset
    print("Loading dataset...")
    if args.lang_code != "en":
        dataset = load_dataset(
            "allenai/madlad-400",
            languages=[args.lang_code], 
            split="clean", 
            cache_dir=args.cache_dir,
            trust_remote_code=True,
        )
    else:
        dataset = load_dataset(
            "allenai/madlad-400",
            languages=[args.lang_code], 
            split="clean", 
            cache_dir=args.cache_dir,
            streaming=True,
            trust_remote_code=True,
        )

    # Take 250K examples randomly
    if args.lang_code not in ("am", "si", "ha", "ig", "ky"):
        print("Taking 250K examples randomly...")
        if args.lang_code != "en":
            dataset = dataset.shuffle(seed=42).select(range(250000))
        else:
            dataset = dataset.shuffle(seed=42).take(250000)
            dataset = Dataset.from_generator(partial(generator_from_iterable_dataset, dataset), features=dataset.features)

    # Load the tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name_or_path,
        cache_dir=args.cache_dir
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Tokenize the dataset
    print("Tokenizing the dataset...")
    dataset = dataset.map(
        lambda examples: tokenizer(examples["text"]),
        batched=True,
        num_proc=args.num_workers,
        remove_columns=dataset.column_names,
    )

    # Group the texts
    print("Grouping the texts...")
    dataset = dataset.map(
        lambda examples: group_texts(examples, args.max_length),
        batched=True, 
        num_proc=args.num_workers // 2,
    )

    # Save the tokenized dataset to a file
    print("Saving tokenized dataset...")
    dataset.save_to_disk(args.output_dir)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lang_code",
        type=str,
        required=True,
        help="Language code of the dataset"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        required=True, 
        help="Path to the output data directory"
    )
    parser.add_argument(
        "--cache_dir", 
        type=str, 
        default=None,
        help="Directory to cache datasets"
    )
    parser.add_argument(
        "--tokenizer_name_or_path", 
        type=str, 
        required=True,
        help="Name or path of the tokenizer to use"
    )
    parser.add_argument(
        "--num_workers", 
        type=int, 
        default=4, 
        help="Number of worker processes to use"
    )
    parser.add_argument(
        "--max_length", 
        type=int, 
        default=512, 
        help="Maximum length of the tokenized sequences"
    )
    args = parser.parse_args()
    main(args)
