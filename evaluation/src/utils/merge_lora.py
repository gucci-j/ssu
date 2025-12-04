import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def main(args):
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model_name_or_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(model, args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = model.merge_and_unload()
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Merged model saved to {args.output_dir}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_name_or_path", type=str, required=True, help="Path to the base model")
    parser.add_argument("--model_name_or_path", type=str, required=True, help="Path to the LoRA model")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the merged model")
    args = parser.parse_args()
    main(args)
