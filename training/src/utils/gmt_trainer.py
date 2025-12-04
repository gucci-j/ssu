"""
Gradient-Mask Tuning (GMT) Trainer implementation.

This module implements the GMT algorithm from "Gradient-Mask Tuning Elevates the Upper Limits of LLM Performance"
(https://arxiv.org/html/2406.15330v2) from AAAI 2025.

GMT selectively updates parameters based on gradient magnitude, keeping only the top-k percentile
of gradients by absolute value during training.
"""

import torch
from transformers import Trainer
from typing import Dict, Any, Optional


class GMTTrainer(Trainer):
    """
    Custom Trainer that implements Gradient-Mask Tuning (GMT).
    
    GMT masks gradients with small absolute values during training, keeping only
    the top-k percentile of gradients for parameter updates.
    """
    
    def __init__(
        self,
        gmt_mask_ratio: float = 0.2,
        gmt_skip_embeddings_and_head: bool = False,
        **kwargs
    ):
        """
        Initialize GMT Trainer.
        
        Args:
            gmt_mask_ratio: Ratio of gradients to keep (0.0 to 1.0). E.g., 0.2 = keep top 20%
            gmt_skip_embeddings_and_head: Whether to skip GMT for embeddings and head layers
            **kwargs: Additional arguments passed to the base Trainer
        """
        super().__init__(**kwargs)
        # Core config
        self.gmt_mask_ratio = float(gmt_mask_ratio)
        self.gmt_skip_embeddings_and_head = bool(gmt_skip_embeddings_and_head)
        self.trainingsteps = 0

        # Official GMT-style mask ratio alias; prefer args.mask_ratio if present
        self.mask_ratio = float(getattr(self.args, "mask_ratio", self.gmt_mask_ratio))

        print(
            f"Initialized GMT Trainer with mask_ratio={self.mask_ratio:.1%}, "
            f"skip_embeddings_and_head={self.gmt_skip_embeddings_and_head}"
        )

    def apply_mask_to_gradient(self, param: torch.nn.Parameter, ratio: float) -> None:
        """Apply in-place top-k magnitude masking to a single parameter's gradient.

        Keeps the top (ratio) fraction of elements by absolute value and zeros the rest.
        Edge cases:
        - ratio <= 0: zero the entire gradient
        - ratio >= 1: keep gradient unchanged
        """
        # Calculate gradient mask  
        grad_abs = param.grad.data.abs()  
        grad_abs_flattened = grad_abs.view(-1)  
        num_to_update = int(len(grad_abs_flattened) * ratio)  

        threshold_value, _ = torch.topk(grad_abs_flattened, num_to_update, largest=True)  
        grad_mask = grad_abs >= threshold_value[-1]
        param.grad.data.mul_(grad_mask.view_as(param.grad.data))


    def _should_skip_module_gmt(self, param_name: str) -> bool:
        """
        Check if a parameter should be skipped from GMT masking.
        
        Args:
            param_name: Name of the parameter
            
        Returns:
            bool: True if parameter should be skipped, False otherwise
        """
        if not self.gmt_skip_embeddings_and_head:
            return False
        
        name_lower = param_name.lower()
        
        # Common embedding layer names
        embedding_keywords = ['embed', 'embedding', 'wte', 'wpe', 'embed_tokens', 'token_embedding', 'word_embedding']
        
        # Common lm_head layer names  
        lm_head_keywords = ['lm_head', 'lm_head_layer', 'output_layer', 'head', 'classifier', 'output_projection']
        
        # Check if parameter name contains any embedding or lm_head keywords
        for keyword in embedding_keywords + lm_head_keywords:
            if keyword in name_lower:
                return True
        
        return False
    

    def training_step(self, model: torch.nn.Module, inputs: Dict[str, Any], num_items_in_batch: Optional[int] = None) -> torch.Tensor:
        """GMT training step with per-microstep top-k gradient masking.

        - Compute loss and call accelerator.backward(loss)
        - Immediately mask gradients parameter-wise to keep only top-k by magnitude
        - Return detached loss; HF Trainer handles accumulation and optimizer stepping
        """
        self.trainingsteps += 1
        model.train()
        if hasattr(self.optimizer, "train") and callable(self.optimizer.train):
            self.optimizer.train()
        inputs = self._prepare_inputs(inputs)

        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs, num_items_in_batch=num_items_in_batch)
        del inputs

        self.accelerator.backward(loss)

        # Apply mask to gradients after backward pass and before accumulation step
        for name, param in model.named_parameters():
            if self._should_skip_module_gmt(name):
                continue
            if param.requires_grad and param.grad is not None:
                self.apply_mask_to_gradient(param, ratio=self.mask_ratio)

        # Update parameters if it's time, otherwise keep accumulating
        if self.trainingsteps % self.args.gradient_accumulation_steps == 0:  
            self.optimizer.step()  
            self.optimizer.zero_grad()  

        # Finally we need to normalize the loss for reporting if GA loss bug is not fixed during compute loss
        if not self.model_accepts_loss_kwargs and self.compute_loss_func is None:
            loss = loss / self.args.gradient_accumulation_steps

        return loss.detach()
    

def create_gmt_trainer(
    model,
    training_args,
    train_dataset,
    data_collator,
    eval_dataset=None,
    gmt_mask_ratio: float = 0.2,
    gmt_skip_embeddings_and_head: bool = False,
    **kwargs
) -> GMTTrainer:
    """
    Create a GMT Trainer with the specified configuration.
    
    Args:
        model: The model to train
        training_args: HuggingFace TrainingArguments
        train_dataset: Training dataset
        data_collator: Data collator
        eval_dataset: Optional evaluation dataset
        gmt_mask_ratio: Ratio of gradients to keep (0.0 to 1.0)
        gmt_skip_embeddings_and_head: Whether to skip GMT for embeddings and head layers
        **kwargs: Additional trainer arguments
        
    Returns:
        GMTTrainer: Configured GMT trainer
    """
    return GMTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        eval_dataset=eval_dataset,
        gmt_mask_ratio=gmt_mask_ratio,
        gmt_skip_embeddings_and_head=gmt_skip_embeddings_and_head,
        **kwargs
    )
