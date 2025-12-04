import re
import shutil
import sys
from pathlib import Path
import json

def move_result_files(checkpoint_dir: str, target_dir: str) -> None:
    """
    Move all results files and their details directories under a target directory,
    categorized by base model name and evaluation task.
    
    Args:
        checkpoint_dir: Path to the checkpoint directory containing results
        target_dir: Target directory to organize results
    """
    checkpoint_path = Path(checkpoint_dir)
    target_path = Path(target_dir)
    
    if not checkpoint_path.exists():
        raise ValueError(f"Checkpoint directory does not exist: {checkpoint_dir}")
    
    # Create target directory if it doesn't exist
    target_path.mkdir(parents=True, exist_ok=True)
    
    # Find all results files
    results_files = list(checkpoint_path.glob("results_*.json"))
    
    for results_file in results_files:
        # Extract timestamp from filename
        timestamp_match = re.search(r'results_(\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}\.\d+)\.json', results_file.name)
        if not timestamp_match:
            continue
            
        timestamp = timestamp_match.group(1)
        details_dir = checkpoint_path / timestamp
        
        # Extract base model name from checkpoint path
        # Assuming path structure: .../models/{model_name}/checkpoint-{num}/...
        path_parts = checkpoint_path.parts
        model_name = None
        for i, part in enumerate(path_parts):
            if part == "models" and i + 1 < len(path_parts):
                model_name = path_parts[i + 1]
                break
        
        if not model_name:
            print(f"Warning: Could not extract model name from path {checkpoint_path}")
            continue
        
        # Ensure model name is a base model name
        if model_name.startswith("OLMo-2-1124-7B"):
            if model_name.startswith("OLMo-2-1124-7B-Instruct"):
                model_abbrev = "OLMo-2-1124-7B-Instruct"
            else:
                model_abbrev = "OLMo-2-1124-7B"
        elif model_name.startswith("OLMo-2-1124-13B"):
            if model_name.startswith("OLMo-2-1124-13B-Instruct"):
                model_abbrev = "OLMo-2-1124-13B-Instruct"
            else:
                model_abbrev = "OLMo-2-1124-13B"
        else:
            print(f"Warning: Unrecognized model name {model_name} in path {checkpoint_path}")
            continue
        
        # Determine evaluation task from results file 
        with open(results_file, 'r') as f:
            results_content = json.load(f)
            for key in results_content["results"]:
                if "belebele" in key:
                    eval_task = "belebele"
                    break
                elif "gmmlu_amh_mcf" in key:
                    eval_task = "gmmlu_amh_mcf"
                    break
                elif "gmmlu_hau_mcf" in key:
                    eval_task = "gmmlu_hau_mcf"
                    break
                elif "gmmlu_kir_mcf" in key:
                    eval_task = "gmmlu_kir_mcf"
                    break
                elif "gmmlu_ibo_mcf" in key:
                    eval_task = "gmmlu_ibo_mcf"
                    break
                elif "gmmlu_npi_mcf" in key:
                    eval_task = "gmmlu_npi_mcf"
                    break
                elif "leaderboard|mmlu:" in key:
                    eval_task = "mmlu"
                    break
                elif "custom|mt:" in key:
                    eval_task = "mt"
                    break
                elif "custom|sum:" in key:
                    eval_task = "sum"
                    break
                elif "extended|mt_bench" in key:
                    eval_task = "mtbench"
                    break
                else:
                    raise ValueError(f"Unrecognized evaluation task in results file: {results_file}")
        
        # Create organized directory structure
        flattened_checkpoint_name = model_name + "__" + checkpoint_path.name
        organized_dir = target_path / model_abbrev / eval_task / "results" / flattened_checkpoint_name
        organized_dir.mkdir(parents=True, exist_ok=True)
        
        # Move results file
        new_results_file = organized_dir / results_file.name
        shutil.move(str(results_file), str(new_results_file))
        print(f"Moved results file: {results_file} -> {new_results_file}")
        
        # Move details directory if it exists
        if details_dir.exists() and details_dir.is_dir():
            organized_dir = target_path / model_abbrev / eval_task / "details" / flattened_checkpoint_name
            organized_dir.mkdir(parents=True, exist_ok=True)
            new_details_dir = organized_dir / details_dir.name
            shutil.move(str(details_dir), str(new_details_dir))
            print(f"Moved details directory: {details_dir} -> {new_details_dir}")


if __name__ == "__main__":
    
    if len(sys.argv) != 3:
        print("Usage: python move_result_files.py <checkpoint_dir> <target_dir>")
        sys.exit(1)
    
    checkpoint_dir = sys.argv[1]
    target_dir = sys.argv[2]
    
    move_result_files(checkpoint_dir, target_dir)
