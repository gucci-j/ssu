import datasets
import torch
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          DataCollatorForLanguageModeling, Trainer)
from typing import Optional, List
try:
    from peft import (
        LoraConfig,
        AdaLoraConfig,
        TaskType,
        get_peft_model,
    )
    _PEFT_AVAILABLE = True
except Exception:
    _PEFT_AVAILABLE = False

from utils import (CustomArgumentParser,
                   freeze_random_parameters,
                   create_calibration_dataloader,
                   create_gmt_trainer,
                   # LoTA
                   lota_calibrate_mask, lota_prepare_sparse_training, lota_parameter_summary,
                   # S2FT
                   s2ft_enable,)


def main(args, training_args):
    #####
    # Load the dataset
    #####
    train_dataset = datasets.load_from_disk(args.dataset_path)
    train_dataset = train_dataset.shuffle(seed=training_args.seed)
    if args.val_dataset_path is not None:
        val_dataset = datasets.load_from_disk(args.val_dataset_path)
    else:
        val_dataset = None

    #####
    # Load the tokenizer
    #####
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name_or_path,
        cache_dir=args.cache_dir
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    #####
    # Set up the data collator
    #####
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    #####
    # Load the model
    #####
    # Detect if FSDP is enabled via TrainingArguments; if so, avoid device_map and let Trainer/FSDP place shards
    fsdp_enabled = bool(getattr(training_args, "fsdp", None)) and str(getattr(training_args, "fsdp")).strip().lower() not in ("", "none")

    if fsdp_enabled:
        # Load on CPU (or default device) and let FSDP handle placement/sharding
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            cache_dir=args.cache_dir,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            low_cpu_mem_usage=True,
        )
        # Ensure FSDP uses original parameters so param names remain stable for GMT and freezing logic
        fsdp_cfg = getattr(training_args, "fsdp_config", None)
        if fsdp_cfg is None:
            fsdp_cfg = {}
            training_args.fsdp_config = fsdp_cfg
        # Handle dict-like vs object-like config containers
        try:
            # dict path
            fsdp_cfg.setdefault("use_orig_params", True)
            fsdp_cfg.setdefault("state_dict_type", "FULL_STATE_DICT")
        except AttributeError:
            # object path
            if not hasattr(fsdp_cfg, "use_orig_params") or getattr(fsdp_cfg, "use_orig_params") is None:
                try:
                    setattr(fsdp_cfg, "use_orig_params", True)
                except Exception:
                    pass
            if not hasattr(fsdp_cfg, "state_dict_type") or getattr(fsdp_cfg, "state_dict_type") is None:
                try:
                    setattr(fsdp_cfg, "state_dict_type", "FULL_STATE_DICT")
                except Exception:
                    pass
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            cache_dir=args.cache_dir,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="cuda" if torch.cuda.is_available() else "cpu"
        )
    
    # Quick exclusivity enforcement: LoTA vs other baseline mechanisms
    if getattr(args, 'use_lota', False):
        # Disable conflicting methods
        if getattr(args, 'peft_method', 'none') != 'none':
            print("[LoTA] Disabling PEFT (LoRA/AdaLoRA) because --use_lota is set.")
            args.peft_method = 'none'
        if getattr(args, 'do_hft', False):
            print("[LoTA] Disabling HFT freezing strategies because --use_lota is set.")
            args.do_hft = False
        if getattr(args, 'use_gmt', False):
            print("[LoTA] Disabling GMT because --use_lota is set.")
            args.use_gmt = False
        if getattr(args, 'use_s2ft', False):
            print("[LoTA] Disabling S2FT because --use_lota is set.")
            args.use_s2ft = False
    
    # Quick exclusivity enforcement: S2FT vs other baseline mechanisms
    if getattr(args, 'use_s2ft', False):
        if getattr(args, 'peft_method', 'none') != 'none':
            print("[S2FT] Disabling PEFT (LoRA/AdaLoRA) because --use_s2ft is set.")
            args.peft_method = 'none'
        if getattr(args, 'do_hft', False):
            print("[S2FT] Disabling HFT freezing because --use_s2ft is set.")
            args.do_hft = False
        if getattr(args, 'use_gmt', False):
            print("[S2FT] Disabling GMT because --use_s2ft is set.")
            args.use_gmt = False
        if getattr(args, 'use_lota', False):
            print("[S2FT] Disabling LoTA because --use_s2ft is set.")
            args.use_lota = False
    
    # Optionally wrap with PEFT (LoRA/AdaLoRA) before any selective freezing
    if getattr(args, 'peft_method', 'none') != 'none':
        if not _PEFT_AVAILABLE:
            raise RuntimeError("peft library is not installed but peft_method was set. Install `peft`.")
        target_modules: Optional[List[str]] = None
        if args.lora_target_modules:
            target_modules = [m.strip() for m in args.lora_target_modules.split(',') if m.strip()]
        bias = args.peft_bias
        # Ensure embeddings and lm_head are tuned with PEFT by saving these modules (kept trainable)
        modules_to_save = [
            'lm_head', 'embed_tokens', 'wte', 'word_embeddings', 'embeddings', 'token_embedding', 'output_projection'
        ]
        if args.peft_method == 'lora':
            lora_cfg = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                target_modules=target_modules,
                bias=bias,
                modules_to_save=modules_to_save,
            )
            model = get_peft_model(model, lora_cfg)
            print("Wrapped model with standard LoRA (PEFT)")
        elif args.peft_method == 'adalora':
            adalora_cfg = AdaLoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                target_modules=target_modules,
                bias=bias,
                modules_to_save=modules_to_save,
                init_r=args.lora_r,
                target_r=args.adalora_target_r,
                tinit=args.adalora_tinit,
                tfinal=args.adalora_tfinal,
                deltaT=args.adalora_delta_t,
                beta1=args.adalora_beta1,
                beta2=args.adalora_beta2,
                orth_reg_weight=args.adalora_orth_reg_weight,
                total_step=args.adalora_total_step,
            )
            model = get_peft_model(model, adalora_cfg)
            print("Wrapped model with AdaLoRA (PEFT)")

        # PEFT is a baseline: do not combine with HFT, LSFT, or GMT
        if getattr(args, 'do_hft', False):
            print("PEFT baseline selected: disabling HFT freezing.")
            args.do_hft = False
        if getattr(args, 'use_gmt', False):
            print("PEFT baseline selected: disabling Gradient-Mask Tuning (GMT).")
            args.use_gmt = False

    # Check for mutual exclusivity between HFT and GMT
    if getattr(args, 'do_hft', False) and getattr(args, 'use_gmt', False):
        raise ValueError("Cannot use both HFT (--do_hft) and GMT (--use_gmt) simultaneously. Please choose one approach.")

    # Optionally set up Lottery Ticket Adaptation (LoTA)
    lota_state = None
    if getattr(args, 'use_lota', False):
        print("=== Lottery Ticket Adaptation (LoTA) Enabled ===")
        print("[LoTA] Starting mask calibration phase...")
        
        # Build simple calibration dataloader (reuse train dataset; random shuffle)
        from torch.utils.data import DataLoader
        calib_batch_size = training_args.per_device_train_batch_size
        calib_loader = DataLoader(
            train_dataset,
            batch_size=calib_batch_size,
            shuffle=True,
            collate_fn=data_collator,
            drop_last=False,
        )
        
        # Optimizer selection
        lr = training_args.learning_rate
        weight_decay = training_args.weight_decay
        if args.lota_optimizer == 'adamw':
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        elif args.lota_optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        elif args.lota_optimizer == 'rmsprop':
            optimizer = torch.optim.RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            raise ValueError(f"Unsupported LoTA optimizer: {args.lota_optimizer}")
        lota_state = lota_calibrate_mask(
            model,
            calib_loader,
            optimizer,
            sparsity=args.lota_sparsity,
            calibration_steps=args.lota_calibration_steps,
            device=None,
            skip_embeddings_and_head=getattr(args, 'lota_skip_embeddings_and_head', False),
            grad_accum_steps=args.lota_grad_accum_steps,
            max_batches=args.lota_calibration_max_batches,
            verbose=args.lota_verbose,
        )
        
        # Prepare sparse adaptation phase
        lota_prepare_sparse_training(model, lota_state, verbose=True)
        
        # Display summary
        print(lota_parameter_summary(model))
    
    # Optionally set up S2FT
    if getattr(args, 'use_s2ft', False):
        print("=== S2FT (Structured Sparse Fine-Tuning) Enabled ===")
        print(f"S2FT config: ratio={args.s2ft_ratio:.2%}, strategy={args.s2ft_strategy}")
        o_ratio = args.s2ft_ratio if getattr(args, 's2ft_include_attn_output', False) else 0.0
        if o_ratio > 0.0:
            print("[S2FT] Including attention output heads (o_proj) with same ratio as FFN down.")
        model, selections = s2ft_enable(
            model,
            v_ratio=0.0,            # not selected in baseline
            o_ratio=o_ratio,        # optional heads
            u_ratio=0.0,            # only down_proj channels in baseline
            d_ratio=args.s2ft_ratio,
            seed=training_args.seed,
            gradient_checkpointing=getattr(training_args, 'gradient_checkpointing', False),
            make_gc_compatible_fn=None,
            freeze_bias=True,
            verbose=True,
        )
        print("[S2FT] Model conversion complete.")
        print(selections)
    
    # Decide which parameters to freeze or train for each module (HFT)
    if args.do_hft:
        # Prepare calibration data for strategies that need it
        calibration_data = None
        if args.freeze_strategy in ["ssu_based", "ssu_elementwise", "ssu_rowwise"]:
            print("Preparing calibration data for SSU-based freezing...")
            calibration_data = create_calibration_dataloader(
                args.calibration_dataset_path,
                args.num_calibration_samples,
                train_dataset, tokenizer
            )
        
        # Apply chosen strategy
        if args.freeze_strategy == "random_based":
            strategy_desc = "random (neuron-level, structured)"
        elif args.freeze_strategy == "random_elementwise":
            strategy_desc = "random (element-wise)"
        elif args.freeze_strategy == "random_rowwise":
            strategy_desc = "random (row-wise, structured)"
        elif args.freeze_strategy == "hft_based":
            strategy_desc = "HFT-based (structured, using activation importance)"
        elif args.freeze_strategy == "magnitude_based":
            strategy_desc = "magnitude-based (freeze large weights, structured)"
        elif args.freeze_strategy == "magnitude_elementwise":
            strategy_desc = "magnitude-based (freeze large weights, element-wise)"
        elif args.freeze_strategy == "magnitude_rowwise":
            strategy_desc = "magnitude-based (row-wise, large rows frozen)"
        elif args.freeze_strategy == "ssu_based":
            strategy_desc = "SSU-based (structured, using activation importance)"
        elif args.freeze_strategy == "ssu_elementwise":
            strategy_desc = "SSU-based (element-wise, using activation importance)"
        elif args.freeze_strategy == "ssu_rowwise":
            strategy_desc = "SSU-based (row-wise, using activation importance)"
        elif args.freeze_strategy == "fisher_based":
            strategy_desc = "Fisher-based (structured, using gradient Fisher information)"
        elif args.freeze_strategy == "fisher_rowwise":
            strategy_desc = "Fisher-based (row-wise, using gradient Fisher information)"
        elif args.freeze_strategy == "fisher_elementwise":
            strategy_desc = "Fisher-based (element-wise, using gradient Fisher information)"
        elif args.freeze_strategy == "sgpt_based":
            strategy_desc = "SparseGPT-based (structured, E[x^2] input statistics)"
        elif args.freeze_strategy == "sgpt_rowwise":
            strategy_desc = "SparseGPT-based (row-wise aggregation of E[x^2])"
        elif args.freeze_strategy == "sgpt_elementwise":
            strategy_desc = "SparseGPT-based (element-wise, E[x^2])"
        else:
            strategy_desc = args.freeze_strategy

        if args.freeze_chat_template_tokens:
            strategy_desc += f" + chat template tokens (ratio: {args.chat_template_freeze_ratio})"

        print(f"Applying Half Fine-Tuning (HFT) with {args.freeze_ratio:.1%} {strategy_desc} parameter freezing...")
        freeze_random_parameters(
            model=model,
            freeze_ratio=args.freeze_ratio,
            seed=training_args.seed,
            strategy=args.freeze_strategy,
            skip_embeddings_and_head=args.skip_embeddings_and_head,
            calibration_data=calibration_data,
            num_calibration_samples=args.num_calibration_samples,
            tokenizer=tokenizer,
            freeze_chat_template_tokens=args.freeze_chat_template_tokens,
            chat_template_freeze_ratio=args.chat_template_freeze_ratio,
        )

    else:
        print("Training all model parameters...")
        
    #####
    # Set up the trainer
    #####
    if getattr(args, 'use_gmt', False):
        # Use GMT Trainer
        print("=== Using Gradient-Mask Tuning (GMT) ===")
        print(f"GMT configuration:")
        print(f"  - Mask ratio: {args.gmt_mask_ratio:.1%} (keep top {args.gmt_mask_ratio:.1%} of gradients)")
        print(f"  - Skip embeddings and head: {args.gmt_skip_embeddings_and_head}")

        trainer = create_gmt_trainer(
            model,
            training_args,
            train_dataset,
            data_collator,
            gmt_mask_ratio=args.gmt_mask_ratio,
            gmt_skip_embeddings_and_head=args.gmt_skip_embeddings_and_head,
        )
    else:
        # Use standard Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=data_collator,
            eval_dataset=val_dataset,
        )
    
    #####
    # Train the model
    #####
    trainer.train()

    #####
    # Save the model
    #####
    trainer.save_model(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    parser = CustomArgumentParser()
    args, training_args = parser.parse_args()
    main(args, training_args)
