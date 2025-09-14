import argparse
import sys
from pathlib import Path

import config
from utils import process_repository
from helpers import create_summary_pie_chart

def main():
    """
    Main function to parse arguments and run the masking and prediction process.
    """
    parser = argparse.ArgumentParser(
        description="Mask a target file and use other files in a directory as context to predict the masks using an LLM."
    )
    parser.add_argument(
        "target_file",
        type=str,
        help="The path to the target code file to be masked (e.g., example_programs/file1.py)."
    )
    parser.add_argument(
        "--repo_dir",
        type=str,
        default=str(config.EXAMPLES_DIR),
        help=f"The path to the directory containing all code files. Defaults to '{config.EXAMPLES_DIR}'."
    )
    args = parser.parse_args()

    # --- Path Validation ---
    target_file_path = Path(args.target_file).resolve()
    repo_dir_path = Path(args.repo_dir).resolve()

    if not target_file_path.is_file():
        print(f"Error: Target file not found at '{target_file_path}'")
        sys.exit(1)
    if not repo_dir_path.is_dir():
        print(f"Error: Repository directory not found at '{repo_dir_path}'")
        sys.exit(1)
    if not config.SCALEDOWN_API_KEY:
        print("Error: SCALEDOWN_API_KEY is not set. Please create a .env file with the key.")
        sys.exit(1)

    # --- Run Processing ---
    results = process_repository(
        target_file_path=target_file_path,
        repo_dir_path=repo_dir_path,
        masking_ratio=config.MASKING_RATIO,
        headers=config.HEADERS
    )

    # --- Display Results ---
    print("\n" + "="*20 + " RESULTS " + "="*20)
    print(f"Target File:          {results['target_file']}")
    print(f"Context Files:        {results['context_file_count']}")
    print("-" * 49)
    print(f"Total Masks:          {results['total_masks']}")
    print(f"Correct Predictions:  {results['correct_predictions']}")
    print(f"Accuracy:             {results['accuracy']:.2%}")
    print("-" * 49)
    print(f"Tokens (Target):      {results['token_count_target']}")
    print(f"Tokens (Masked):      {results['token_count_masked']}")
    print(f"Tokens (Context):     {results['token_count_context']}")
    print("=" * 49 + "\n")


    # --- Generate Visualization ---
    output_plot_path = config.PLOTS_DIR / f"accuracy_summary_{results['target_file']}.png"
    create_summary_pie_chart(results, output_plot_path)

if __name__ == "__main__":
    main()