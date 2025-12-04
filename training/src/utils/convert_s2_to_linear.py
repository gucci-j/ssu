#!/usr/bin/env python
"""Convert S2FT checkpoints to standard Linear layers for vLLM.

This script loads a Hugging Face causal LM checkpoint that may contain
S2ColumnLinear / S2RowLinear modules (Structured Sparse Fine-Tuning), fuses the
S2 weights, replaces those modules with standard torch.nn.Linear, and saves the
converted model to a new output directory.

Example:
    python training/scripts/convert_s2_to_linear.py \
        --input /path/to/ckpt \
        --output /path/to/converted \
        --dtype bfloat16 \
        --trust-remote-code

Notes:
- By default, loads on CPU with low memory usage.
- If your model requires custom code (e.g., OLMo2), pass --trust-remote-code.
- If tokenizer isn't in the input path, pass --tokenizer /path/to/tokenizer.
"""

import argparse
import os
import json
from typing import Optional, Dict
from transformers import AutoModelForCausalLM, AutoTokenizer

#############################################################################
# The following code is adapted from https://github.com/Infini-AI-Lab/S2FT
# @inproceedings{yang2024s2ft,
#  title={S2FT: Efficient, Scalable and Generalizable LLM Fine-tuning by Structured Sparsity},
#  author={Yang, Xinyu and Leng, Jixuan and Guo, Geyang and Zhao, Jiawei and Nakada, Ryumei and Zhang, Linjun and Yao, Huaxiu and Chen, Beidi},
#  booktitle={The 38th Conference on Neural Information Processing Systems (NeurIPS)},
#  year={2024}
#}
import copy
import torch
from torch import nn

import torch
import math

from torch import nn
from torch.nn.modules import Module
from torch import Tensor
from torch.nn import init
from torch.nn.parameter import Parameter


class S2ColumnLinear(Module):

    __constants__ = ["in_features", "out_features"]
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        start=None,
        end=None,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(
            torch.empty((out_features, in_features), **factory_kwargs),
            requires_grad=True,
        )
        if bias:
            self.bias = Parameter(
                torch.empty(out_features, **factory_kwargs), requires_grad=True
            )
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()
        self.start = start
        self.end = end

        self.s2 = nn.Parameter(
            torch.zeros(end - start, in_features), requires_grad=True
        )
        self.weight.requires_grad = False
        self.fused = False

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def fuse_s2_weight(self):
        if self.fused == True:
            return
        self.weight.data[self.start : self.end, :] += self.s2
        self.fused = True

    def unfuse_s2_weight(self):
        if self.fused == False:
            return
        self.weight[self.start : self.end, :] -= self.s2
        self.fused = False

    def forward(self, input: Tensor) -> Tensor:
        base_output = torch.nn.functional.linear(input, self.weight, self.bias)
        if self.fused:
            return base_output
        else:
            s2_output = torch.nn.functional.linear(input, self.s2, None)
            base_output[:, :, self.start : self.end] += s2_output
            return base_output

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}"


class S2RowLinear(Module):

    __constants__ = ["in_features", "out_features"]
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        start=None,
        end=None,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(
            torch.empty((out_features, in_features), **factory_kwargs),
            requires_grad=True,
        )
        if bias:
            self.bias = Parameter(
                torch.empty(out_features, **factory_kwargs), requires_grad=True
            )
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()
        self.start = start
        self.end = end

        self.s2 = nn.Parameter(
            torch.zeros(out_features, end - start), requires_grad=True
        )
        self.weight.requires_grad = False
        self.fused = False

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def fuse_s2_weight(self):
        if self.fused == True:
            return
        self.weight.data[:, self.start : self.end] += self.s2
        self.fused = True

    def unfuse_s2_weight(self):
        if self.fused == False:
            return
        self.weight[:, self.start : self.end] -= self.s2
        self.fused = False

    def forward(self, input: Tensor) -> Tensor:
        base_output = torch.nn.functional.linear(input, self.weight, self.bias)
        if self.fused:
            return base_output
        else:
            s2_output = torch.nn.functional.linear(
                input[:, :, self.start : self.end], self.s2, None
            )
            base_output += s2_output
            return base_output

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}"


def only_optimize_s2_parameters(model):
    # Turn off the gradient of all the parameters except the S2 parameters
    for name, param in model.named_parameters():
        if "s2" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
    return model


# convert the linear layers in the MHA module to S2
def convert_mha_layer_to_s2(model, selected_parameters):
    head_dim = model.config.hidden_size // model.config.num_attention_heads
    for i in range(model.config.num_hidden_layers):
        layer = model.model.layers[i]
        only_v = list(
            set(selected_parameters["v_proj"][i])
            - set(selected_parameters["o_proj"][i])
        )
        only_o = list(
            set(selected_parameters["o_proj"][i])
            - set(selected_parameters["v_proj"][i])
        )
        vo = list(
            set(selected_parameters["o_proj"][i])
            & set(selected_parameters["v_proj"][i])
        )
        order = only_v + vo + only_o
        for j in range(model.config.num_attention_heads):
            if j not in order:
                order.append(j)
        if len(only_v) + len(vo) > 0:
            module = layer.self_attn.v_proj
            checkpoint = copy.deepcopy(module.state_dict())
            layer.self_attn.v_proj = S2ColumnLinear(
                in_features=module.in_features,
                out_features=module.out_features,
                bias=module.bias,
                start=0,
                end=(len(only_v) + len(vo)) * head_dim,
                device=next(module.parameters()).device,
                dtype=next(module.parameters()).dtype,
            )
            layer.self_attn.v_proj.load_state_dict(checkpoint, strict=False)
            del module
            del checkpoint
        if len(only_o) + len(vo) > 0:
            module = layer.self_attn.o_proj
            checkpoint = copy.deepcopy(module.state_dict())
            layer.self_attn.o_proj = S2RowLinear(
                in_features=module.in_features,
                out_features=module.out_features,
                bias=module.bias,
                start=(len(only_v)) * head_dim,
                end=(len(only_v) + len(vo) + len(only_o)) * head_dim,
                device=next(module.parameters()).device,
                dtype=next(module.parameters()).dtype,
            )
            layer.self_attn.o_proj.load_state_dict(checkpoint, strict=False)
            del module
            del checkpoint

        q_weight = layer.self_attn.q_proj.weight.data
        q_weight = q_weight.reshape(
            model.config.num_key_value_heads, -1, q_weight.shape[-1]
        )
        layer.self_attn.q_proj.weight.data = q_weight[order, :, :].reshape(
            -1, q_weight.shape[-1]
        )
        k_weight = layer.self_attn.k_proj.weight.data
        k_weight = k_weight.reshape(
            model.config.num_key_value_heads, -1, k_weight.shape[-1]
        )
        layer.self_attn.k_proj.weight.data = k_weight[order, :, :].reshape(
            -1, k_weight.shape[-1]
        )
        v_weight = layer.self_attn.v_proj.weight.data
        v_weight = v_weight.reshape(
            model.config.num_key_value_heads, -1, v_weight.shape[-1]
        )
        layer.self_attn.v_proj.weight.data = v_weight[order, :, :].reshape(
            -1, v_weight.shape[-1]
        )
        o_weight = layer.self_attn.o_proj.weight.data
        o_weight = o_weight.reshape(
            o_weight.shape[0], model.config.num_attention_heads, -1
        )
        layer.self_attn.o_proj.weight.data = o_weight[:, order, :].reshape(
            o_weight.shape[0], -1
        )

        del v_weight, o_weight
    return model


# convert the linear layers in the FFN module to S2
def convert_ffn_layer_to_s2(model, selected_parameters):
    for i in range(model.config.num_hidden_layers):
        layer = model.model.layers[i]
        only_u = [
            j
            for j in selected_parameters["up_proj"][i]
            if j not in selected_parameters["down_proj"][i]
        ]
        only_d = [
            j
            for j in selected_parameters["down_proj"][i]
            if j not in selected_parameters["up_proj"][i]
        ]
        ud = [
            j
            for j in selected_parameters["up_proj"][i]
            if j in selected_parameters["down_proj"][i]
        ]
        order = only_u + ud + only_d
        for j in range(model.config.intermediate_size):
            if j not in order:
                order.append(j)
        if len(only_u) + len(ud) > 0:
            module = layer.mlp.up_proj
            checkpoint = copy.deepcopy(module.state_dict())
            layer.mlp.up_proj = S2ColumnLinear(
                in_features=module.in_features,
                out_features=module.out_features,
                bias=module.bias,
                start=0,
                end=(len(only_u) + len(ud)),
                device=next(module.parameters()).device,
                dtype=next(module.parameters()).dtype,
            )
            layer.mlp.up_proj.load_state_dict(checkpoint, strict=False)
            del module
            del checkpoint

        if len(ud) + len(only_d) > 0:
            module = layer.mlp.down_proj
            checkpoint = copy.deepcopy(module.state_dict())
            layer.mlp.down_proj = S2RowLinear(
                in_features=module.in_features,
                out_features=module.out_features,
                bias=module.bias,
                start=len(only_u),
                end=(len(only_u) + len(ud) + len(only_d)),
                device=next(module.parameters()).device,
                dtype=next(module.parameters()).dtype,
            )
            layer.mlp.down_proj.load_state_dict(checkpoint, strict=False)
            del module
            del checkpoint
        u_weight = layer.mlp.up_proj.weight.data
        layer.mlp.up_proj.weight.data = u_weight[order, :]
        g_weight = layer.mlp.gate_proj.weight.data
        layer.mlp.gate_proj.weight.data = g_weight[order, :]
        d_weight = layer.mlp.down_proj.weight.data
        layer.mlp.down_proj.weight.data = d_weight[:, order]


# convert the S2FT layer to linear layer
def convert_s2_to_linear_layer(model):
    for module in model.modules():
        if isinstance(module, S2ColumnLinear) or isinstance(module, S2RowLinear):
            module.fuse_s2_weight()
    return model



#############################################################################
###### The following utility is newly added by the SSU authors

def _get_parent_and_attr(root, module_name: str):
    """Return (parent_module, last_attr) for a dotted module path.

    Handles numeric indices for ModuleList/Sequential (e.g., 'layers.3.self_attn.v_proj').
    """
    parts = module_name.split(".")
    parent = root
    for p in parts[:-1]:
        # Handle ModuleList/Sequential index
        if p.isdigit():
            parent = parent[int(p)]
        else:
            parent = getattr(parent, p)
    return parent, parts[-1]


def convert_s2_modules_to_linear(model) -> int:
    """Fuse S2 weights and replace S2*Linear modules with nn.Linear in-place.

    This utility walks the module tree, and for each S2ColumnLinear / S2RowLinear:
    - Computes the fused weight (adding the structured-sparse `s2` slice into `weight`).
    - Constructs a standard `torch.nn.Linear` with identical shapes/dtypes/devices.
    - Copies fused weights and bias, then replaces the module in its parent.

    Returns the number of modules replaced.

    Call this right before saving the model used with S2FT so that the saved
    checkpoint contains only standard Linear layers (compatible with vLLM).
    """
    # Collect names first to avoid mutating while iterating
    s2_module_names = []
    for name, module in model.named_modules():
        if isinstance(module, (S2ColumnLinear, S2RowLinear)):
            s2_module_names.append(name)

    replaced = 0
    for name in s2_module_names:
        module = dict(model.named_modules())[name]
        assert isinstance(module, (S2ColumnLinear, S2RowLinear))

        # Compute fused weight without modifying original in case of reuse
        with torch.no_grad():
            fused_w = module.weight.detach().clone()
            if isinstance(module, S2ColumnLinear):
                # Add s2 to selected output rows
                if not module.fused:
                    fused_w[module.start:module.end, :] += module.s2.detach()
            else:  # S2RowLinear
                if not module.fused:
                    fused_w[:, module.start:module.end] += module.s2.detach()

            bias = module.bias.detach().clone() if module.bias is not None else None

            # Build replacement Linear with same shapes/dtype/device
            device = fused_w.device
            dtype = fused_w.dtype
            new_linear = nn.Linear(
                in_features=module.in_features,
                out_features=module.out_features,
                bias=(bias is not None),
                device=device,
                dtype=dtype,
            )
            new_linear.weight.copy_(fused_w)
            if bias is not None:
                new_linear.bias.copy_(bias)

        parent, attr = _get_parent_and_attr(model, name)
        setattr(parent, attr, new_linear)
        replaced += 1

    return replaced


# ---- New: fuse S2 tensors directly from checkpoint (no custom modules required) ----
try:
    from safetensors.torch import load_file as _safe_load_file
except Exception:
    _safe_load_file = None


def _load_s2_tensors_from_ckpt(ckpt_dir: str) -> Dict[str, torch.Tensor]:
    """Load all '.s2' tensors from a HF checkpoint directory with safetensors.

    Supports both single-file ('model.safetensors') and sharded checkpoints
    (with 'model.safetensors.index.json'). Returns a dict of key -> tensor.
    """
    if _safe_load_file is None:
        return {}
    index_path = os.path.join(ckpt_dir, "model.safetensors.index.json")
    s2_tensors: Dict[str, torch.Tensor] = {}
    if os.path.isfile(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            idx = json.load(f)
        weight_map = idx.get("weight_map", {})
        shard_files = sorted(set(weight_map.values()))
        for shard in shard_files:
            shard_path = os.path.join(ckpt_dir, shard)
            if not os.path.isfile(shard_path):
                continue
            data = _safe_load_file(shard_path, device="cpu")
            for k, v in data.items():
                if k.endswith(".s2"):
                    s2_tensors[k] = v
    else:
        single = os.path.join(ckpt_dir, "model.safetensors")
        if os.path.isfile(single):
            data = _safe_load_file(single, device="cpu")
            for k, v in data.items():
                if k.endswith(".s2"):
                    s2_tensors[k] = v
    return s2_tensors


def _get_submodule(root: nn.Module, name: str) -> nn.Module:
    """Get submodule by dotted path supporting numeric indices (ModuleList)."""
    parts = name.split(".")
    mod = root
    for p in parts:
        if p.isdigit():
            mod = mod[int(p)]
        else:
            mod = getattr(mod, p)
    return mod


def fuse_s2_from_checkpoint(model: nn.Module, ckpt_dir: str, assume_start_zero: bool = True) -> int:
    """Fuse S2 tensors from checkpoint into the currently loaded standard model.

    This is a fallback path when the runtime model doesn't have S2 modules.
    It reads '.s2' tensors from the checkpoint, then adds them to the correct
    slices of the corresponding Linear weights under the assumption that the
    training pipeline re-ordered channels/heads to make the selected span
    contiguous starting at column/row index 0 (true for the common FFN-down only case).

    Returns number of tensors fused.
    """
    s2_tensors = _load_s2_tensors_from_ckpt(ckpt_dir)
    if not s2_tensors:
        return 0
    fused = 0
    for key, s2 in s2_tensors.items():
        # Expect keys like 'model.layers.X.mlp.down_proj.s2' or '...v_proj.s2'
        module_path = key[:-3]  # strip '.s2'
        try:
            mod = _get_submodule(model, module_path)
        except Exception:
            # If module path doesn't exist on the base model, skip
            continue
        if not isinstance(mod, nn.Linear):
            # Only support standard Linear targets here
            continue
        w = mod.weight
        s2 = s2.to(device=w.device, dtype=w.dtype)

        # Infer shape pattern: Column (k, in_features) vs Row (out_features, k)
        if s2.shape == (w.shape[0], s2.shape[1]) and s2.shape[0] == w.shape[0]:
            # Could be Row: [out_features, k]
            k = s2.shape[1]
            if k > w.shape[1]:
                continue
            # If we cannot infer non-zero start, require assume_start_zero
            if not assume_start_zero:
                raise ValueError(
                    f"Cannot infer start offset for row-S2 '{key}'. Pass --assume-start-zero to force start=0."
                )
            with torch.no_grad():
                w[:, :k] += s2
            fused += 1
        elif s2.shape == (s2.shape[0], w.shape[1]) and s2.shape[1] == w.shape[1]:
            # Could be Column: [k, in_features]
            k = s2.shape[0]
            if k > w.shape[0]:
                continue
            if not assume_start_zero:
                raise ValueError(
                    f"Cannot infer start offset for col-S2 '{key}'. Pass --assume-start-zero to force start=0."
                )
            with torch.no_grad():
                w[:k, :] += s2
            fused += 1
        else:
            # Shape mismatch; skip
            continue
    return fused


def reconstruct_s2_modules_from_ckpt(
    model: nn.Module,
    ckpt_dir: str,
    start_map: Optional[Dict[str, int]] = None,
    assume_start_zero: bool = True,
) -> int:
    """Rebuild S2 modules in-place from checkpoint `.s2` tensors, then load them.

    This constructs `S2ColumnLinear`/`S2RowLinear` wrappers at the same module paths
    where `.s2` tensors exist in the checkpoint. It copies over the existing Linear
    weights/bias into the new S2 modules, sets their `s2` tensors from the checkpoint,
    and replaces modules in the model tree.

    Returns the number of modules reconstructed.

    Use with `convert_s2_modules_to_linear(model)` afterwards to fuse and replace
    them with vanilla `nn.Linear` for vLLM.
    """
    s2_tensors = _load_s2_tensors_from_ckpt(ckpt_dir)
    if not s2_tensors:
        return 0
    start_map = start_map or {}
    reconstructed = 0
    for key, s2 in s2_tensors.items():
        module_path = key[:-3]  # strip '.s2'
        try:
            mod = _get_submodule(model, module_path)
        except Exception:
            continue
        if not isinstance(mod, nn.Linear):
            continue
        # Determine row vs column S2 shape
        w = mod.weight
        in_f, out_f = mod.in_features, mod.out_features
        device, dtype = w.device, w.dtype
        s2 = s2.to(device=device, dtype=dtype)

        # Row case: [out_features, k] added across input slice
        if s2.shape[0] == out_f and s2.shape[1] <= in_f:
            k = int(s2.shape[1])
            start = start_map.get(module_path, 0 if assume_start_zero else None)
            if start is None:
                raise ValueError(
                    f"Need start offset for row-S2 '{module_path}'. Provide --start-map or use --assume-start-zero."
                )
            end = start + k
            if start < 0 or end > in_f:
                raise ValueError(
                    f"Invalid start={start} for '{module_path}': k={k}, in_features={in_f}"
                )
            # Build S2RowLinear and transplant weights
            new_mod = S2RowLinear(
                in_features=in_f,
                out_features=out_f,
                bias=(mod.bias is not None),
                start=start,
                end=end,
                device=device,
                dtype=dtype,
            )
            with torch.no_grad():
                new_mod.weight.copy_(w)
                if mod.bias is not None:
                    new_mod.bias.copy_(mod.bias)
                new_mod.s2.copy_(s2)
                new_mod.weight.requires_grad_(False)
            parent, attr = _get_parent_and_attr(model, module_path)
            setattr(parent, attr, new_mod)
            reconstructed += 1
        # Column case: [k, in_features] added to output rows
        elif s2.shape[1] == in_f and s2.shape[0] <= out_f:
            k = int(s2.shape[0])
            start = start_map.get(module_path, 0 if assume_start_zero else None)
            if start is None:
                raise ValueError(
                    f"Need start offset for col-S2 '{module_path}'. Provide --start-map or use --assume-start-zero."
                )
            end = start + k
            if start < 0 or end > out_f:
                raise ValueError(
                    f"Invalid start={start} for '{module_path}': k={k}, out_features={out_f}"
                )
            new_mod = S2ColumnLinear(
                in_features=in_f,
                out_features=out_f,
                bias=(mod.bias is not None),
                start=start,
                end=end,
                device=device,
                dtype=dtype,
            )
            with torch.no_grad():
                new_mod.weight.copy_(w)
                if mod.bias is not None:
                    new_mod.bias.copy_(mod.bias)
                new_mod.s2.copy_(s2)
                new_mod.weight.requires_grad_(False)
            parent, attr = _get_parent_and_attr(model, module_path)
            setattr(parent, attr, new_mod)
            reconstructed += 1
        else:
            # Unsupported shape mismatch
            continue
    return reconstructed


_DTYPE_MAP = {
    "float32": torch.float32,
    "fp32": torch.float32,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float16": torch.float16,
    "fp16": torch.float16,
}


def _sel_for_layer(d: Dict, i: int):
    """Fetch selection list for a layer from dict with int-or-str keys."""
    if d is None:
        return []
    return d.get(i, d.get(str(i), []))


def compute_start_map_from_selections(model: nn.Module, selections: Dict) -> Dict[str, int]:
    """Derive S2 start offsets for each module path from training selections.

    Mirrors convert_mha_layer_to_s2 / convert_ffn_layer_to_s2 logic:
    - v_proj start = 0 when any V selected (size = (|only_v|+|vo|)*head_dim)
    - o_proj start = |only_v| * head_dim when any O selected
    - up_proj start = 0 when any U selected
    - down_proj start = |only_u| when any D selected
    Returns mapping of module path (no .s2) -> start offset.
    """
    start_map: Dict[str, int] = {}
    cfg = getattr(model, "config", None)
    if cfg is None:
        return start_map
    L = int(getattr(cfg, "num_hidden_layers"))
    H = int(getattr(cfg, "num_attention_heads"))
    hidden_size = int(getattr(cfg, "hidden_size"))
    head_dim = hidden_size // H if H > 0 else 0

    # MHA
    mha = selections.get("mha", {}) if isinstance(selections, dict) else {}
    sel_v = mha.get("v_proj") if isinstance(mha, dict) else None
    sel_o = mha.get("o_proj") if isinstance(mha, dict) else None
    for i in range(L):
        V = set(_sel_for_layer(sel_v, i) or [])
        O = set(_sel_for_layer(sel_o, i) or [])
        if V:
            # v_proj start is always 0 when present
            start_map[f"model.layers.{i}.self_attn.v_proj"] = 0
        if O:
            only_v = V - O
            start_o = len(only_v) * head_dim
            start_map[f"model.layers.{i}.self_attn.o_proj"] = start_o

    # FFN
    ffn = selections.get("ffn", {}) if isinstance(selections, dict) else {}
    sel_u = ffn.get("up_proj") if isinstance(ffn, dict) else None
    sel_d = ffn.get("down_proj") if isinstance(ffn, dict) else None
    for i in range(L):
        U = set(_sel_for_layer(sel_u, i) or [])
        D = set(_sel_for_layer(sel_d, i) or [])
        if U:
            start_map[f"model.layers.{i}.mlp.up_proj"] = 0
        if D:
            only_u = U - D
            start_d = len(only_u)
            start_map[f"model.layers.{i}.mlp.down_proj"] = start_d

    return start_map


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Convert S2FT checkpoint to standard Linear layers for vLLM")
    p.add_argument("--input", required=True, help="Path to HF model directory (checkpoint to convert)")
    p.add_argument("--output", required=True, help="Path to save converted model")
    p.add_argument("--tokenizer", default=None, help="Optional tokenizer path; defaults to --input")
    p.add_argument("--dtype", default="bfloat16", choices=list(_DTYPE_MAP.keys()), help="Model dtype for loading/saving")
    p.add_argument("--assume-start-zero", action="store_true", help="Assume selected span starts at 0 when fusing S2 tensors from checkpoint (recommended for FFN-down only)")
    p.add_argument("--reconstruct-s2", action="store_true", help="Rebuild S2 modules from checkpoint tensors before fusing (robust for non-zero starts when combined with --start-map)")
    p.add_argument("--start-map", type=str, default=None, help="Path to JSON mapping of module paths (without .s2) to integer start offsets")
    p.add_argument("--selections", type=str, default=None, help="Path to JSON selections dict from training; used to derive start offsets automatically")
    p.add_argument("--no-safe-tensors", action="store_true", help="Disable safetensors when saving")
    p.add_argument("--max-shard-size", default="5GB", help="Max shard size for save_pretrained")
    p.add_argument("--attn-impl", default=None, choices=[None, "flash_attention_2", "sdpa", "eager"], help="Optional attention implementation hint")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)

    torch_dtype = _DTYPE_MAP[args.dtype]

    print(f"[S2-Convert] Loading model from: {args.input}")
    model = AutoModelForCausalLM.from_pretrained(
        args.input,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        attn_implementation=args.attn_impl if args.attn_impl is not None else None,
        device_map=None,  # load on CPU by default
    )

    # Run conversion
    print("[S2-Convert] Preparing to convert S2 to vanilla Linear ...")
    replaced = convert_s2_modules_to_linear(model)
    if replaced > 0:
        print(f"[S2-Convert] Replaced S2 modules: {replaced}")
    else:
        # Optional reconstruction path
        start_map: Optional[Dict[str, int]] = None
        if args.start_map:
            with open(args.start_map, "r", encoding="utf-8") as f:
                start_map = json.load(f)
        # Derive start_map from selections if provided
        if args.selections:
            try:
                with open(args.selections, "r", encoding="utf-8") as f:
                    selections = json.load(f)
                derived_map = compute_start_map_from_selections(model, selections)
                if start_map is None:
                    start_map = derived_map
                else:
                    # Merge, prefer explicit start_map values
                    for k, v in derived_map.items():
                        start_map.setdefault(k, v)
                print(f"[S2-Convert] Loaded start_map: {start_map}")
                print(f"[S2-Convert] Derived start_map from selections for {len(derived_map)} modules.")
            except Exception as e:
                print(f"[S2-Convert] Warning: failed to load/derive selections: {e}")
        if args.reconstruct_s2:
            rebuilt = reconstruct_s2_modules_from_ckpt(
                model,
                args.input,
                start_map=start_map,
                assume_start_zero=args.assume_start_zero,
            )
            print(f"[S2-Convert] Reconstructed S2 modules from checkpoint: {rebuilt}")
            if rebuilt > 0:
                replaced = convert_s2_modules_to_linear(model)
                print(f"[S2-Convert] Replaced reconstructed S2 modules: {replaced}")
            elif rebuilt == 0:
                print("[S2-Convert] No .s2 tensors found to reconstruct; attempting direct fusion fallback ...")
                fused = fuse_s2_from_checkpoint(model, args.input, assume_start_zero=args.assume_start_zero)
                print(f"[S2-Convert] Fused S2 tensors from checkpoint: {fused}")
                if fused == 0:
                    print("[S2-Convert] No S2 modules or tensors detected; saving as-is.")
        else:
            # Simple fusion fallback (works for start=0 cases like Down-only / Output-only)
            fused = fuse_s2_from_checkpoint(model, args.input, assume_start_zero=args.assume_start_zero)
            print(f"[S2-Convert] Fused S2 tensors from checkpoint: {fused}")
            if fused == 0:
                print("[S2-Convert] No S2 modules or tensors detected; saving as-is.")

    # Save converted model
    print(f"[S2-Convert] Saving converted model to: {args.output}")
    model.save_pretrained(
        args.output,
        safe_serialization=not args.no_safe_tensors,
        max_shard_size=args.max_shard_size,
        from_pt=True,
        state_dict=None,  # use internal state
        safe_weights=not args.no_safe_tensors,
    )

    # Save tokenizer
    tok_path: Optional[str] = args.tokenizer or args.input
    try:
        tok = AutoTokenizer.from_pretrained(tok_path, trust_remote_code=True)
        tok.save_pretrained(args.output)
        print("[S2-Convert] Tokenizer saved.")
    except Exception as e:
        print(f"[S2-Convert] Warning: failed to load/save tokenizer from '{tok_path}': {e}")

    print("[S2-Convert] Done. The output should be vLLM-compatible.")


if __name__ == "__main__":
    main()
