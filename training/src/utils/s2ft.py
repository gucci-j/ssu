"""Extended S2FT utilities that mirror the original authors' enable block.

This module provides helper functions to:
1. Randomly select attention head indices for `v_proj` / `o_proj` (MHA part).
2. Randomly select FFN channel indices for `up_proj` / `down_proj` (FFN part).
3. Convert selected linear layers to `S2ColumnLinear` / `S2RowLinear` via the
    existing implementations in `s2_utils.py`.
4. Freeze all non-S2 parameters, exposing only the structured sparse (`s2`) tensors.
5. Optionally make the model GC compatible (the caller passes a helper).

Metadata stored on the (possibly wrapped) model after invocation:
     _s2ft_total_params       -- total parameters of underlying model
     _s2ft_trainable_params   -- number of trainable parameters (just S2 + optional bias)
     _s2ft_strategy           -- description of applied S2FT conversion
     _s2ft_selected_mha       -- dict with selected head indices per layer
     _s2ft_selected_ffn       -- dict with selected channel indices per layer
"""

import copy
import random
from typing import Dict, List, Optional, Tuple

import torch

from .s2 import S2RowLinear, S2ColumnLinear
from .s2_utils import (
    convert_ffn_layer_to_s2,
    convert_mha_layer_to_s2,
    convert_s2_to_linear_layer,
    only_optimize_s2_parameters,
)

__all__ = [
    "S2RowLinear",
    "S2ColumnLinear",
    "s2ft_select_mha_heads",
    "s2ft_select_ffn_up_down",
    "s2ft_enable",
    "convert_s2_to_linear_layer",
]


def _default_make_model_gradient_checkpointing_compatible(model):
    """Default helper to make models GC-compatible, matching authors' snippet.

    - If the model exposes `enable_input_require_grads`, call it.
    - Else, register a forward hook on input embeddings to require grad.
    """
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    elif hasattr(model, "get_input_embeddings"):
        def make_inputs_require_grad(module, input, output):
            output.requires_grad_(True)
        model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
    return model

def s2ft_select_mha_heads(
    model,
    v_ratio: float,
    o_ratio: float,
    seed: int = 42,
    strategy: str = "random",
) -> Dict[str, Dict[int, List[int]]]:
    """Select attention head indices for v_proj and o_proj per layer.

    Returns a dict: {"v_proj": {layer: [heads]}, "o_proj": {layer: [heads]}}.
    Mirrors the original authors' global sampling approach.
    """
    if not hasattr(model, "config"):
        raise ValueError("Model missing .config; expected HF causal LM.")
    L = int(getattr(model.config, "num_hidden_layers"))
    H = int(getattr(model.config, "num_attention_heads"))
    total_heads = L * H
    v_ratio = max(0.0, min(float(v_ratio), 1.0))
    o_ratio = max(0.0, min(float(o_ratio), 1.0))
    rng = random.Random(seed)
    if strategy != "random":
        raise NotImplementedError(f"Strategy '{strategy}' not implemented; only 'random'.")

    k_v = int(total_heads * v_ratio)
    k_o = int(total_heads * o_ratio)
    if k_v == 0 and v_ratio > 0.0:
        k_v = 1
    if k_o == 0 and o_ratio > 0.0:
        k_o = 1
    global_indices = list(range(total_heads))
    select_v = sorted(rng.sample(global_indices, k_v)) if k_v > 0 else []
    select_o = sorted(rng.sample(global_indices, k_o)) if k_o > 0 else []

    parameters_v: Dict[int, List[int]] = {i: [] for i in range(L)}
    parameters_o: Dict[int, List[int]] = {i: [] for i in range(L)}
    for v in select_v:
        parameters_v[v // H].append(v % H)
    for o in select_o:
        parameters_o[o // H].append(o % H)
    return {"v_proj": parameters_v, "o_proj": parameters_o}


def s2ft_select_ffn_up_down(
    model,
    u_ratio: float,
    d_ratio: float,
    seed: int = 42,
    strategy: str = "random",
) -> Dict[str, Dict[int, List[int]]]:
    """Select FFN channel indices for up_proj and down_proj per layer.

    Returns a dict: {"up_proj": {layer: [channels]}, "down_proj": {layer: [channels]}}.
    Follows the original authors' global sampling procedure.
    """
    if not hasattr(model, "config"):
        raise ValueError("Model missing .config; expected HF causal LM.")
    L = int(getattr(model.config, "num_hidden_layers"))
    D = int(getattr(model.config, "intermediate_size"))
    total = L * D
    u_ratio = max(0.0, min(float(u_ratio), 1.0))
    d_ratio = max(0.0, min(float(d_ratio), 1.0))
    rng = random.Random(seed)
    if strategy != "random":
        raise NotImplementedError(f"Strategy '{strategy}' not implemented; only 'random'.")
    k_u = int(total * u_ratio)
    k_d = int(total * d_ratio)
    if k_u == 0 and u_ratio > 0.0:
        k_u = 1
    if k_d == 0 and d_ratio > 0.0:
        k_d = 1
    indices = list(range(total))
    select_u = sorted(rng.sample(indices, k_u)) if k_u > 0 else []
    select_d = sorted(rng.sample(indices, k_d)) if k_d > 0 else []
    parameters_u: Dict[int, List[int]] = {i: [] for i in range(L)}
    parameters_d: Dict[int, List[int]] = {i: [] for i in range(L)}
    for u in select_u:
        parameters_u[u // D].append(u % D)
    for d in select_d:
        parameters_d[d // D].append(d % D)
    return {"up_proj": parameters_u, "down_proj": parameters_d}


def s2ft_enable(
    model,
    v_ratio: float = 0.0,
    o_ratio: float = 0.0,
    u_ratio: float = 0.0,
    d_ratio: float = 0.0,
    seed: int = 42,
    gradient_checkpointing: bool = False,
    make_gc_compatible_fn: Optional[callable] = None,
    freeze_bias: bool = True,
    verbose: bool = False,
) -> Tuple[object, Dict[str, Dict[int, List[int]]]]:
    """Apply S2FT conversion to a model, matching original training script logic.

    Steps:
      1. (Optional) Select & convert MHA heads for v_proj / o_proj.
      2. (Optional) Select & convert FFN channels for up_proj / down_proj.
      3. Freeze all parameters except S2.
      4. (Optional) Make model gradient checkpointing compatible.
      5. Return updated model and a consolidated selection dictionary.

    The consolidated selection dict has keys: 'mha' and/or 'ffn'.
    """
    selections: Dict[str, Dict[int, List[int]]] = {}

    # Attention conversion
    if v_ratio > 0.0 or o_ratio > 0.0:
        sel_mha = s2ft_select_mha_heads(model, v_ratio, o_ratio, seed=seed)
        convert_mha_layer_to_s2(model, sel_mha)
        selections["mha"] = sel_mha  # nested dict {'v_proj':..., 'o_proj':...}
        if verbose:
            v_count = sum(len(v) for v in sel_mha["v_proj"].values())
            o_count = sum(len(o) for o in sel_mha["o_proj"].values())
            print(f"[S2FT] Selected MHA heads -> v:{v_count} o:{o_count}")

    # FFN conversion
    if u_ratio > 0.0 or d_ratio > 0.0:
        sel_ffn = s2ft_select_ffn_up_down(model, u_ratio, d_ratio, seed=seed)
        convert_ffn_layer_to_s2(model, sel_ffn)
        selections["ffn"] = sel_ffn  # nested dict {'up_proj':..., 'down_proj':...}
        if verbose:
            u_count = sum(len(u) for u in sel_ffn["up_proj"].values())
            d_count = sum(len(d) for d in sel_ffn["down_proj"].values())
            print(f"[S2FT] Selected FFN channels -> up:{u_count} down:{d_count}")

    # Freeze & expose only S2 params
    if selections:
        model = only_optimize_s2_parameters(model)
        # Use provided GC helper or default implementation
        if make_gc_compatible_fn is None:
            make_gc_compatible_fn = _default_make_model_gradient_checkpointing_compatible
        try:
            model = make_gc_compatible_fn(model)
        except Exception:
            if verbose:
                print("[S2FT] Warning: make_model_gradient_checkpointing_compatible failed.")
        
        # Accounting metadata
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        setattr(model, "_s2ft_total_params", total_params)
        setattr(model, "_s2ft_trainable_params", trainable_params)
        setattr(model, "_s2ft_strategy", "original_enable_block")
        if "mha" in selections:
            setattr(model, "_s2ft_selected_mha", selections["mha"])
        if "ffn" in selections:
            setattr(model, "_s2ft_selected_ffn", selections["ffn"])
        
        # Unfreeze embeddings and LM head by default
        if hasattr(model, "get_input_embeddings") and model.get_input_embeddings() is not None:
            for param in model.get_input_embeddings().parameters():
                param.requires_grad = True
        if hasattr(model, "get_output_embeddings") and model.get_output_embeddings() is not None:
            for param in model.get_output_embeddings().parameters():
                param.requires_grad = True

        if verbose:
            ratio = trainable_params / total_params if total_params else 0.0
            print(f"[S2FT] Trainable params after enable: {trainable_params:,} / {total_params:,} ({ratio:.2%})")

    return model, selections
