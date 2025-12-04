import argparse
from transformers import HfArgumentParser, TrainingArguments

class CustomArgumentParser(argparse.ArgumentParser):
    def __init__(self):
        self.parser = argparse.ArgumentParser(
            description="Tune a language model."
        )
        self.hf_parser = HfArgumentParser(TrainingArguments)

        # Define any custom arguments using argparse
        self.parser.add_argument(
            "--dataset_path",
            type=str,
            required=True,
            help="Path to the tokenized dataset."
        )
        self.parser.add_argument(
            "--val_dataset_path",
            type=str,
            default=None,
            help="Path to the tokenized validation dataset."
        )
        self.parser.add_argument(
            "--tokenizer_name_or_path", 
            type=str, 
            required=True,
            help="Path to the tokenizer."
        )
        self.parser.add_argument(
            "--model_name_or_path", 
            type=str, 
            required=True,
            help="Path to the model."
        )
        self.parser.add_argument(
            "--cache_dir", 
            type=str, 
            default=None,
            help="Path to the cache directory."
        )
        self.parser.add_argument(
            "--do_hft",
            action="store_true",
            help="Whether to apply static selective parameter updates."
        )
        self.parser.add_argument(
            "--use_gmt", 
            action="store_true",
            help="Whether to apply Gradient-Mask Tuning (GMT) instead of static parameter updates."
        )
        self.parser.add_argument(
            "--freeze_ratio", 
            type=float, 
            default=0.5,
            help="Ratio of parameters to freeze randomly for each module or weight matrix when do_hft is True."
        )
        self.parser.add_argument(
            "--freeze_strategy", 
            type=str, 
            choices=[
                "random_based", 
                "random_elementwise", 
                "random_rowwise",
                "hft_based",
                "magnitude_based", 
                "magnitude_elementwise", 
                "magnitude_rowwise", 
                "ssu_based", 
                "ssu_elementwise", 
                "ssu_rowwise",
                "fisher_based",
                "fisher_elementwise",
                "fisher_rowwise",
                "sgpt_based",
                "sgpt_elementwise",
                "sgpt_rowwise",
            ],
            default="fine_grained",
            help=(
                "Strategy for parameter freezing: "
                "'random_based' freezes individual neurons/weights (structured by columns), "
                "'random_rowwise' freezes full rows (output neurons) randomly, "
                "'random_elementwise' freezes individual elements, "
                "'hft_based' freezes entire modules selected based on the HFT paper's criteria, "
                "'magnitude_based' freezes large magnitude weights (structured by columns), "
                "'magnitude_rowwise' freezes rows by magnitude, "
                "'magnitude_elementwise' freezes individual elements by magnitude, "
                "'ssu_based' uses SSU importance scores (structured by columns), "
                "'ssu_rowwise' freezes rows by SSU scores, "
                "'ssu_elementwise' uses SSU importance scores (element-wise), "
                "'fisher_based' uses Fisher information (structured by rows/cols auto), "
                "'fisher_rowwise' freezes rows by Fisher information, "
                "'fisher_elementwise' uses Fisher information (element-wise), "
                "'sgpt_based' uses SparseGPT input statistics E[x^2] (structured, column-wise default), "
                "'sgpt_rowwise' aggregates to rows, "
                "'sgpt_elementwise' uses E[x^2] for element-wise selection, "
            )
        )
        self.parser.add_argument(
            "--skip_embeddings_and_head", 
            action="store_true",
            help="Skip freezing embeddings and language model head parameters for all freezing strategies."
        )
        self.parser.add_argument(
            "--use_percentile",
            action="store_true",
            default=True,
            help="Use percentile-based freezing for weight differences (default). If False, requires diff_threshold."
        )
        self.parser.add_argument(
            "--calibration_dataset_path",
            type=str,
            default=None,
            help="Path to the calibration dataset for Wanda-based freezing. If not provided, uses a subset of the training dataset."
        )
        self.parser.add_argument(
            "--num_calibration_samples",
            type=int,
            default=128,
            help="Number of samples to use for calibration in Wanda-based freezing strategies."
        )
        self.parser.add_argument(
            "--calibration_max_length",
            type=int,
            default=512,
            help="Maximum sequence length for calibration samples in Wanda-based freezing."
        )
        self.parser.add_argument(
            "--freeze_chat_template_tokens",
            action="store_true",
            help="Additionally freeze chat template special tokens to preserve conversational structure. This is applied as a topping on top of the main freezing strategy."
        )
        self.parser.add_argument(
            "--chat_template_freeze_ratio",
            type=float,
            default=1.0,
            help="Ratio of chat template special tokens to freeze (0.0 to 1.0, default: 1.0 = all special tokens)."
        )

        # GMT (Gradient-Mask Tuning) options
        self.parser.add_argument(
            "--gmt_mask_ratio",
            type=float,
            default=0.2,
            help="Ratio of gradients to keep (top-k percentile by absolute value) in GMT. E.g., 0.2 means keep top 20% of gradients."
        )
        self.parser.add_argument(
            "--gmt_skip_embeddings_and_head",
            action="store_true",
            help="Skip applying GMT masking to embedding layers and language model head parameters."
        )
        
        # LoTA options
        self.parser.add_argument(
            "--use_lota",
            action="store_true",
            help="Enable Lottery Ticket Adaptation (LoTA) baseline: calibrate mask then sparse fine-tune only selected weights."
        )
        self.parser.add_argument(
            "--lota_sparsity",
            type=float,
            default=0.9,
            help="Fraction of weights to freeze in LoTA (e.g. 0.9 => keep top 10% trainable)."
        )
        self.parser.add_argument(
            "--lota_calibration_steps",
            type=int,
            default=100,
            help="Number of optimization steps for mask calibration phase (T). Can be small (even 1)."
        )
        self.parser.add_argument(
            "--lota_grad_accum_steps",
            type=int,
            default=1,
            help="Gradient accumulation steps during LoTA calibration (for large batch simulation)."
        )
        self.parser.add_argument(
            "--lota_skip_embeddings_and_head",
            action="store_true",
            help="During mask extraction treat embeddings & lm_head as always trainable (excluded from sparsity)."
        )
        self.parser.add_argument(
            "--lota_optimizer",
            type=str,
            choices=["adamw", "adam", "rmsprop"],
            default="adamw",
            help="Optimizer used in LoTA calibration phase (defaults to same as main training unless specified)."
        )
        self.parser.add_argument(
            "--lota_calibration_max_batches",
            type=int,
            default=None,
            help="Optional cap on number of batches processed during calibration (overrides steps if reached sooner)."
        )
        self.parser.add_argument(
            "--lota_verbose",
            action="store_true",
            help="Print detailed progress logs during LoTA calibration & mask building." 
        )
        
        # S2FT options
        self.parser.add_argument(
            "--use_s2ft",
            action="store_true",
            help="Enable S2FT baseline: select FFN channels, permute coupled weights, and fine-tune only the connected submatrices."
        )
        self.parser.add_argument(
            "--s2ft_ratio",
            type=float,
            default=0.01,
            help="Fraction of FFN channels (per layer, uniform allocation overall) to fine-tune under S2FT."
        )
        self.parser.add_argument(
            "--s2ft_strategy",
            type=str,
            choices=[
                "random",
            ],
            default="random",
            help=(
                "Channel selection strategy for S2FT. Supported: random."
            )
        )
        self.parser.add_argument(
            "--s2ft_include_attn_output",
            action="store_true",
            help=(
                "If set, S2FT also prioritizes and fine-tunes attention output (o_proj) weights "
                "in addition to selected FFN Down projection columns."
            )
        )

        # PEFT / LoRA / AdaLoRA options
        self.parser.add_argument(
            "--peft_method",
            type=str,
            choices=["none", "lora", "adalora"],
            default="none",
            help="PEFT method to use. Set to 'adalora' to enable AdaLoRA via PEFT, 'lora' for standard LoRA, or 'none' to disable."
        )
        self.parser.add_argument(
            "--lora_r",
            type=int,
            default=8,
            help="LoRA/AdaLoRA initial rank r. For AdaLoRA this is the initial low-rank (init_r)."
        )
        self.parser.add_argument(
            "--lora_alpha",
            type=int,
            default=16,
            help="LoRA alpha (scaling)."
        )
        self.parser.add_argument(
            "--lora_dropout",
            type=float,
            default=0.0,
            help="LoRA dropout."
        )
        self.parser.add_argument(
            "--lora_target_modules",
            type=str,
            default="q_proj,k_proj,v_proj,o_proj,up_proj,down_proj,gate_proj",
            help="Comma-separated list of module name substrings to target with (Ada)LoRA (e.g. 'q_proj,k_proj,v_proj,o_proj'). If omitted, PEFT will try to auto-detect."
        )
        self.parser.add_argument(
            "--peft_bias",
            type=str,
            default="none",
            choices=["none", "all", "lora_only"],
            help="Bias handling in PEFT."
        )
        
        # AdaLoRA-specific knobs
        self.parser.add_argument(
            "--adalora_target_r",
            type=int,
            default=8,
            help="AdaLoRA target rank (final)."
        )
        self.parser.add_argument(
            "--adalora_tinit",
            type=int,
            default=0,
            help="AdaLoRA tinit: step to start allocation."
        )
        self.parser.add_argument(
            "--adalora_tfinal",
            type=int,
            default=None,
            help="AdaLoRA tfinal: step to end allocation. If None, will be inferred from total steps."
        )
        self.parser.add_argument(
            "--adalora_delta_t",
            type=int,
            default=1,
            help="AdaLoRA deltaT: interval (in steps) between rank allocations."
        )
        self.parser.add_argument(
            "--adalora_beta1",
            type=float,
            default=0.85,
            help="AdaLoRA EMA beta1."
        )
        self.parser.add_argument(
            "--adalora_beta2",
            type=float,
            default=0.85,
            help="AdaLoRA EMA beta2."
        )
        self.parser.add_argument(
            "--adalora_orth_reg_weight",
            type=float,
            default=0.5,
            help="Orthogonality regularization weight for AdaLoRA."
        )
        self.parser.add_argument(
            "--adalora_total_step",
            type=int,
            default=None,
            help="Total training steps for AdaLoRA scheduling. If None, inferred from dataset and TrainingArguments."
        )

    def parse_args(self):
        args, extras = self.parser.parse_known_args()
        print(extras)
        training_args = self.hf_parser.parse_args_into_dataclasses(extras)[0]
        return args, training_args
    