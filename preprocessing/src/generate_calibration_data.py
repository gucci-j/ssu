import json
from functools import partial
from typing import List, Dict, Any, Iterable

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


def format_chat_with_tokenizer(tokenizer: AutoTokenizer, example: Dict[str, Any]) -> str:
    """Format a single example into a plain text prompt using the tokenizer's chat template when possible.

    Supports common schemas:
    - {"messages": [{"role": "user|assistant|system", "content": str}, ...]}
    - {"conversations": same as messages}
    - instruction-style: {instruction, input?, output/response}
    - single-field text: {text}
    """
    # 1) Chat-style data
    msgs = example.get("messages") or example.get("conversations")
    if isinstance(msgs, list) and msgs and isinstance(msgs[0], dict) and "role" in msgs[0]:
        try:
            return tokenizer.apply_chat_template(
                msgs,
                tokenize=False,
                add_generation_prompt=False,
            )
        except Exception:
            # Fallback to a simple concatenation if chat template isn't available
            parts = []
            for m in msgs:
                role = m.get("role", "user")
                content = m.get("content", "")
                parts.append(f"{role}: {content}")
            return "\n\n".join(parts)

    # 2) Instruction-style data
    instr = example.get("instruction") or example.get("prompt")
    inp = example.get("input")
    out = example.get("output") or example.get("response")
    if instr or out:
        msgs = []
        if instr:
            if inp:
                full_instr = f"{instr}\n\nInput: {inp}"
            else:
                full_instr = instr
            msgs.append({"role": "user", "content": full_instr})
        if out:
            msgs.append({"role": "assistant", "content": out})
        try:
            return tokenizer.apply_chat_template(
                msgs,
                tokenize=False,
                add_generation_prompt=False,
            )
        except Exception:
            # Fallback to a simple concatenation if chat template isn't available
            parts = []
            for m in msgs:
                role = m.get("role", "user")
                content = m.get("content", "")
                parts.append(f"{role}: {content}")
            return "\n\n".join(parts)

    # 3) Plain text
    if example.get("text"):
        return example["text"]

    # Fallback to JSON dump
    return json.dumps(example, ensure_ascii=False)


def main(args):
    # Load the tokenizer first (needed for chat templating)
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name_or_path,
        cache_dir=args.cache_dir,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load the dataset (streaming to avoid full download) and sample examples
    print(f"Loading dataset: {args.dataset_name} [{args.split}] (streaming={args.streaming}) ...")
    if args.use_raw_text:
        dataset = load_dataset(
            args.dataset_name,
            split=args.split,
            cache_dir=args.cache_dir,
            streaming=args.streaming,
            trust_remote_code=True,
        )
    else:
        dataset = load_dataset(
            args.dataset_name,
            split=args.split,
            cache_dir=args.cache_dir,
            streaming=args.streaming,
            trust_remote_code=True,
        )

    # Shuffle (streaming shuffle uses a buffer) then take N examples
    if args.shuffle:
        buffer_size = max(1000, min(100000, args.shuffle_buffer))
        try:
            dataset = dataset.shuffle(seed=args.seed, buffer_size=buffer_size)
        except Exception:
            # Non-streaming datasets may not support buffer_size
            dataset = dataset.shuffle(seed=args.seed)

    print(f"Sampling {args.num_samples} examples for calibration...")
    if args.streaming:
        iterator: Iterable[Dict[str, Any]] = dataset.take(args.num_samples)
    else:
        # Non-streaming: select a subset without materializing the full dataset in memory when possible
        if hasattr(dataset, "select") and len(dataset) > args.num_samples:
            dataset = dataset.select(range(args.num_samples))
        iterator = iter(dataset)

    texts: List[str] = []
    count = 0
    for ex in iterator:
        if args.use_raw_text:
            # Use raw text field if available
            if "text" in ex:
                texts.append(ex["text"])
            else:
                print(f"Warning: example {count} has no 'text' field, skipping.")
                continue
        else:
            texts.append(format_chat_with_tokenizer(tokenizer, ex))
        count += 1
        if count >= args.num_samples:
            break

    # Tokenize
    print("Tokenizing formatted texts...")
    # Ensure we don't truncate below block size before packing
    tok_max_length = args.max_length
    if tok_max_length is None or tok_max_length < args.block_size:
        print(f"Adjusting max_length from {tok_max_length} to block_size {args.block_size} for consistent packing.")
        tok_max_length = args.block_size
    tokenized = tokenizer(
        texts,
        padding=False,
        truncation=True,
        max_length=tok_max_length,
        return_attention_mask=True,
    )

    # Use packing into fixed blocks for calibration (contiguous chunks)
    print("Packing into fixed-length blocks...")
    tokenized_ds = Dataset.from_dict({
        "input_ids": tokenized["input_ids"],
        "attention_mask": tokenized["attention_mask"],
    })

    # Group/pack
    packed = tokenized_ds.map(
        lambda examples: group_texts(examples, args.block_size),
        batched=True,
        num_proc=1,  # deterministic
    )

    # Save the tokenized dataset to a file
    print(f"Saving packed calibration dataset to: {args.output_dir}")
    packed.save_to_disk(args.output_dir)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="allenai/tulu-3-sft-olmo-2-mixture",
        help="Hugging Face dataset repo id to sample calibration data from"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split to use"
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
        "--num_samples",
        type=int,
        default=2000,
        help="Number of raw examples to sample for calibration before packing"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for shuffling"
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle the dataset before sampling (recommended if streaming)"
    )
    parser.add_argument(
        "--shuffle_buffer",
        type=int,
        default=20000,
        help="Buffer size for streaming shuffle"
    )
    parser.add_argument(
        "--tokenizer_name_or_path", 
        type=str, 
        default="allenai/OLMo-2-1124-7B-Instruct",
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
    parser.add_argument(
        "--block_size",
        type=int,
        default=512,
        help="Fixed block size for calibration chunks (after packing)"
    )
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="Use streaming mode when loading the dataset"
    )
    parser.add_argument(
        "--use_raw_text",
        action="store_true",
        help="Use raw text field if available, otherwise format using chat template"
    )
    parser.add_argument(
        "--lang_code",
        type=str,
        default="en",
        help="Language code to filter the dataset (if applicable)"
    )
    args = parser.parse_args()
    main(args)
