from pathlib import Path
from tqdm import tqdm
import torch
from alpaca_eval import evaluate
import datasets
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import json

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def convert_examples_to_prompts(examples, tokenizer):
    """Convert a batch of examples to prompts for the model."""
    prompts = []
    for instruction in examples["instruction"]:
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": instruction + " Answer the question as concise as possible while still providing all the necessary information."},
        ]
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        prompts.append(prompt)
    return prompts


def main(args):
    if args.skip_inference is not True:
        eval_set = datasets.load_dataset(
            "tatsu-lab/alpaca_eval", "alpaca_eval"
        )["eval"]

        if args.base_model_name_or_path is not None:
            from peft import PeftModel
            base_model = AutoModelForCausalLM.from_pretrained(
                args.base_model_name_or_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )
            model = PeftModel.from_pretrained(
                base_model,
                args.model_name_or_path,
            )
            model = model.merge_and_unload()
            tokenizer = AutoTokenizer.from_pretrained(
                args.model_name_or_path
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                args.model_name_or_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )
            tokenizer = AutoTokenizer.from_pretrained(
                args.model_name_or_path
            )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        if tokenizer.padding_side != "left":
            tokenizer.padding_side = "left"
        pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer)

        batch_size = args.batch_size
        results = []
        for i in tqdm(range(0, len(eval_set), batch_size), desc="Generating responses"):
            batch = eval_set[i:i+batch_size]
            prompts = convert_examples_to_prompts(batch, tokenizer)
            outputs = pipe(
                prompts,
                max_new_tokens=512,
                temperature=0.8, # Higher temperature for more creative responses
                top_p=0.8, # Higher top_p for more diverse responses
                top_k=40, # Higher top_k for more diverse responses
                repetition_penalty=1.1,
                do_sample=True,
                num_beams=1,
                no_repeat_ngram_size=3,
                return_full_text=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                batch_size=batch_size,
            ) # -> List[List[dict[str, str]]]
            batch["output"] = [output[0]["generated_text"] for output in outputs]
            batch["generator"] = [args.model_abbrev] * len(batch)
            logger.info(f"Instruction: {batch['instruction'][-1]}")
            logger.info(f"Output: {batch['output'][-1]}")
            results.extend(
                [{"dataset": ds, "instruction": inst, "output": out, "generator": gen} 
                for ds, inst, out, gen in zip(batch["dataset"], batch["instruction"], batch["output"], batch["generator"])]
            )

        output_file = Path(args.output_dir) / f"model_outputs_{args.postfix}.json"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=4)

    if args.do_eval:
        output_path = Path(args.output_dir) / args.postfix
        output_path.mkdir(parents=True, exist_ok=True)
        output_file = Path(args.output_dir) / f"model_outputs_{args.postfix}.json"
        evaluate(
            model_outputs=output_file,
            annotators_config=args.annotators_config,
            output_path=output_path,
        )

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run Alpaca-Eval")
    parser.add_argument("--model_abbrev", type=str, default="oasst_pythia_12b")
    parser.add_argument("--model_name_or_path", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--base_model_name_or_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--skip_inference", action="store_true")
    parser.add_argument("--do_eval", action="store_true")
    parser.add_argument("--annotators_config", type=str, default="alpaca-7b/annotators_config.json")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for generation")
    parser.add_argument("--postfix", type=str, default="default", help="Postfix for output files")
    args = parser.parse_args()
    main(args)
