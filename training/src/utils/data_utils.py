import datasets
import torch
from torch.utils.data import DataLoader, Subset
from transformers import DataCollatorForLanguageModeling


def create_calibration_dataloader(
    calibration_dataset_path,
    num_calibration_samples,
    train_dataset, 
    tokenizer
):
    """
    Create a calibration DataLoader for Wanda-based freezing strategies.
    
    Args:
        calibration_dataset_path (str): Path to the calibration dataset. If None, uses a subset of the training dataset.
        num_calibration_samples (int): Number of samples to use for calibration.
        train_dataset: The training dataset
        tokenizer: The tokenizer
    
    Returns:
        DataLoader: Calibration data loader, or None if not needed
    """
    if calibration_dataset_path is not None:
        # Load separate calibration dataset
        calibration_dataset = datasets.load_from_disk(calibration_dataset_path)
        print(f"Loaded calibration dataset from {calibration_dataset_path}")
    else:
        # Use a subset of the training dataset for calibration
        calibration_dataset = train_dataset
        print("Using subset of training dataset for calibration")
    
    # Create a subset for calibration
    total_samples = len(calibration_dataset)
    num_samples = min(num_calibration_samples, total_samples)
    
    # Use deterministic sampling for reproducibility
    indices = list(range(0, total_samples, max(1, total_samples // num_samples)))[:num_samples]
    calibration_subset = Subset(calibration_dataset, indices)
    
    print(f"Created calibration subset with {len(calibration_subset)} samples")
    
    # Create data collator for calibration
    calibration_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, 
        mlm=False,
        return_tensors="pt"
    )
    
    # Create calibration dataloader
    calibration_dataloader = DataLoader(
        calibration_subset,
        batch_size=1,  # Process one sample at a time for activation collection
        shuffle=False,  # Deterministic for reproducibility
        collate_fn=calibration_collator,
        pin_memory=torch.cuda.is_available(),
        num_workers=0  # Avoid multiprocessing issues with hooks
    )
    
    return calibration_dataloader
