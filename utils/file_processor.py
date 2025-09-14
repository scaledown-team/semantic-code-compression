# file_processor.py
import os
import logging
from pathlib import Path
from typing import Dict, Any

import config  # Import the config module
from utils import get_token_counter
from helpers import get_llm_predictions

from ast_lsp_heuristic_context_compression import (
    mask_code,
    evaluate_predictions,
    get_prompt,
)

token_counter = get_token_counter()

def _gather_context_files(repo_dir_path: Path, target_file_path: Path) -> list[Path]:
    """
    Recursively scans a directory for files to be used as context,
    ignoring specified directories and file types.
    """
    context_files = []
    # os.walk is efficient for recursively walking a directory tree
    for root, dirs, files in os.walk(repo_dir_path, topdown=True):
        dirs[:] = [d for d in dirs if d not in config.IGNORE_DIRECTORIES]

        for filename in files:
            file_path = Path(root) / filename

            # 1. Skip if it's the target file itself
            if file_path.resolve() == target_file_path.resolve():
                continue

            # 2. Skip if its extension is not allowed (if the allow list is active)
            if config.ALLOWED_EXTENSIONS and file_path.suffix not in config.ALLOWED_EXTENSIONS:
                continue

            context_files.append(file_path)

    return context_files


def process_repository(
    target_file_path: Path,
    repo_dir_path: Path,
    masking_ratio: float,
    headers: Dict[str, str]
) -> Dict[str, Any]:
    """
    Processes a repository by masking a target file and using the rest as context.
    """
    # 1. Read and Mask the Target File
    logging.info(f"Reading target file: {target_file_path.name}")
    target_code = target_file_path.read_text(encoding="utf-8")
    masked_code, actual_mapping = mask_code(target_code, masking_ratio)
    logging.info(f"Masked {len(actual_mapping)} identifiers in {target_file_path.name}.")

    print(f"Masked Code:\n{masked_code}\n")

    # 2. Build the Context using the new recursive gathering function
    logging.info(f"Scanning {repo_dir_path} for context files...")
    context_files = _gather_context_files(repo_dir_path, target_file_path)
    
    context_parts = []
    for file_path in context_files:
        try:
            # Using relative path for cleaner context headers
            relative_path = file_path.relative_to(repo_dir_path)
            content = file_path.read_text(encoding="utf-8")
            context_parts.append(f"--- START OF FILE: {relative_path} ---\n\n{content}\n\n--- END OF FILE: {relative_path} ---\n\n")
        except (UnicodeDecodeError, IOError) as e:
            logging.warning(f"Could not read or decode file {file_path.name}, skipping. Reason: {e}")

    full_context = "".join(context_parts)
    logging.info(f"Context built from {len(context_files)} files.")

    # 3. Get LLM Predictions
    prompt = get_prompt()
    parsed_predictions = get_llm_predictions(
        masked_code=masked_code,
        context=full_context,
        prompt=prompt,
        headers=headers
    )

    print(f"Parsed Predictions: {parsed_predictions}")
    print(f"Actual Mapping: {actual_mapping}")

    # 4. Evaluate Predictions
    total_masks = len(actual_mapping)
    correct_predictions = 0
    if parsed_predictions:
        evaluation = evaluate_predictions(parsed_predictions, actual_mapping)
        if evaluation:
            correct_predictions = sum(1 for result in evaluation.values() if result.get("correct"))
            logging.info(f"Evaluation complete: {correct_predictions}/{total_masks} correct.")
        else:
            logging.warning("Evaluation function returned no results.")
    else:
        logging.error("No predictions to evaluate.")

    accuracy = (correct_predictions / total_masks) if total_masks > 0 else 0.0

    # 5. Gather statistics
    stats = {
        "target_file": target_file_path.name,
        "total_masks": total_masks,
        "correct_predictions": correct_predictions,
        "accuracy": accuracy,
        "token_count_target": token_counter(target_code),
        "token_count_masked": token_counter(masked_code),
        "token_count_context": token_counter(full_context),
        "context_file_count": len(context_files),
    }
    return stats