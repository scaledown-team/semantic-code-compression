from ast_lsp_heuristic_context_compression import (
    call_scaledown,
    mask_code,
    evaluate_predictions,
    extract_json_from_response,
    get_prompt,
)
import os
from dotenv import load_dotenv
import json
from pathlib import Path
import matplotlib.pyplot as plt
import math
import re

load_dotenv()

SCALEDOWN_API_KEY = os.getenv("SCALEDOWN_API_KEY")
headers = {
    "x-api-key": SCALEDOWN_API_KEY,
    "Content-Type": "application/json",
}

EXAMPLES_DIR = Path("example_programs")
PLOTS_DIR = Path("plots")
PLOTS_DIR.mkdir(exist_ok=True)

# Aggregation containers
per_file_stats = []  # dicts: {filename, total_masks, correct, accuracy, tokens, masked_tokens}
total_masks = 0
total_correct = 0
all_mask_results = []  # flattened list of booleans for all masks across files

# Load prompt once
prompt = get_prompt()

all_files = [p for p in EXAMPLES_DIR.iterdir() if p.is_file()]

# Token counting utilities (tiktoken optional, fallback to whitespace heuristic)
try:
    import tiktoken
    ENCODING = tiktoken.get_encoding("cl100k_base")
    def token_counter(text: str) -> int:
        return len(ENCODING.encode(text))
    print("Using tiktoken for token counting (cl100k_base).")
except Exception:
    def token_counter(text: str) -> int:
        if not text:
            return 0
        return len(re.findall(r"\S+", text))
    print("tiktoken not available — using whitespace-based token heuristic.")

for file_path in all_files:
    filename = file_path.name
    print(f"Processing file: {filename}")
    code = file_path.read_text(encoding="utf-8")

    # token counts for original and masked code
    orig_token_count = token_counter(code)

    # Create masked code and mapping
    masked_code, actual_mapping = mask_code(code, 0.99)
    masked_token_count = token_counter(masked_code) if masked_code else 0

    # Call Scaledown
    try:
        response = call_scaledown(prompt=prompt, context=masked_code, headers=headers)
    except TypeError:
        response = call_scaledown(prompt=prompt, context=masked_code)

    parsed_predictions = None
    status_code = getattr(response, "status_code", None) if response is not None else None

    if response and status_code == 200:
        try:
            response_data = json.loads(response.text)
        except Exception:
            try:
                response_data = response.json()
            except Exception:
                response_data = None

        full_response_text = None
        if response_data:
            full_response_text = response_data.get("full_response") or response_data.get("response")

        response_text_to_parse = full_response_text or (response.text if isinstance(response.text, str) else None)
        if response_text_to_parse:
            parsed_predictions = extract_json_from_response(response_text_to_parse)
            if not parsed_predictions:
                print(f"Could not parse predictions JSON for {filename} from full_response.")
        else:
            print(f"No text to parse in scaledown response for {filename}.")
    else:
        # If API failed or missing, we will mark masks as incorrect below
        print(f"Scaledown API call failed or not available for {filename} (status {status_code}).")

    # Evaluate
    file_total_masks = len(actual_mapping) if actual_mapping else 0
    file_correct = 0

    if parsed_predictions:
        evaluation = evaluate_predictions(parsed_predictions, actual_mapping)
        if evaluation:
            for mask_id, result in evaluation.items():
                is_correct = bool(result.get("correct"))
                file_correct += 1 if is_correct else 0
                all_mask_results.append(is_correct)
        else:
            print(f"Evaluation returned nothing for {filename}.")
    else:
        # no parsed predictions -> mark masks as incorrect for bookkeeping
        if file_total_masks > 0:
            all_mask_results.extend([False] * file_total_masks)

    total_masks += file_total_masks
    total_correct += file_correct

    accuracy = (file_correct / file_total_masks) if file_total_masks > 0 else math.nan
    per_file_stats.append({
        "filename": filename,
        "total_masks": file_total_masks,
        "correct": file_correct,
        "accuracy": accuracy,
        "tokens": orig_token_count,
        "masked_tokens": masked_token_count,
    })

    print(f"File: {filename} | Masks: {file_total_masks} | Correct: {file_correct} | "
          f"Accuracy: {accuracy if not math.isnan(accuracy) else 'N/A'} | Tokens: {orig_token_count} | Masked tokens: {masked_token_count}")

# Summary
overall_accuracy = (total_correct / total_masks) if total_masks > 0 else math.nan
print("\n=== Overall Summary ===")
print(f"Files processed: {len(per_file_stats)}")
print(f"Total masks: {total_masks}")
print(f"Total correct: {total_correct}")
print(f"Overall accuracy: {overall_accuracy if not math.isnan(overall_accuracy) else 'N/A'}")
print(f"Total tokens (sum of file tokens): {sum(d['tokens'] for d in per_file_stats)}")

# Visualization: per-file accuracy bar chart with token counts as a secondary axis
if per_file_stats:
    # Sort files by accuracy descending (NaN last)
    per_file_sorted = sorted(per_file_stats, key=lambda x: (-(x['accuracy']) if not math.isnan(x['accuracy']) else 1))
    labels = [d['filename'] for d in per_file_sorted]
    accuracies = [0.0 if math.isnan(d['accuracy']) else d['accuracy'] for d in per_file_sorted]
    token_counts = [d['tokens'] for d in per_file_sorted]
    masked_token_counts = [d['masked_tokens'] for d in per_file_sorted]

    fig, ax1 = plt.subplots(figsize=(max(8, len(labels) * 0.6), 6))

    # Bar colors: gray for NaN, green gradient otherwise
    colors = []
    for a in accuracies:
        if math.isnan(a):
            colors.append('lightgray')
        else:
            greens = 0.3 + 0.7 * a
            colors.append((0.0, greens, 0.0))

    bars = ax1.bar(range(len(labels)), accuracies, color=colors)
    ax1.set_xticks(range(len(labels)))
    ax1.set_xticklabels(labels, rotation=45, ha='right')
    ax1.set_ylabel('Accuracy (proportion of masks predicted correctly)')
    ax1.set_ylim(0, 1)
    ax1.set_title('Per-file mask prediction accuracy (tokens shown on secondary axis)')

    # annotate accuracy values
    for bar, a in zip(bars, accuracies):
        height = bar.get_height()
        if not math.isnan(a):
            ax1.text(bar.get_x() + bar.get_width() / 2, height + 0.02, f"{a:.2f}", ha='center', va='bottom', fontsize=8)

    # secondary axis for token counts (original and masked)
    ax2 = ax1.twinx()
    ax2.plot(range(len(labels)), token_counts, marker='o', linestyle='-', linewidth=1, label='tokens (orig)')
    ax2.plot(range(len(labels)), masked_token_counts, marker='x', linestyle='--', linewidth=1, label='tokens (masked)')
    ax2.set_ylim(0, max(10, max(token_counts + masked_token_counts) * 1.1))
    ax2.set_ylabel('Token count')

    # annotate token counts near markers
    for i, (tc, mtc) in enumerate(zip(token_counts, masked_token_counts)):
        ax2.text(i, tc + max(1, max(token_counts) * 0.02), f"{tc}", ha='center', va='bottom', fontsize=7)
        ax2.text(i, mtc - max(1, max(masked_token_counts) * 0.02), f"{mtc}", ha='center', va='top', fontsize=7)

    # legend for token lines
    ax2.legend(loc='upper left')

    plt.tight_layout()
    combined_plot = PLOTS_DIR / 'per_file_accuracy_with_tokens.png'
    plt.savefig(combined_plot)
    print(f"Saved per-file accuracy + tokens plot to: {combined_plot}")
    plt.show()

# Visualization: overall correct vs incorrect pie
if total_masks > 0:
    correct_count = sum(1 for v in all_mask_results if v is True)
    incorrect_count = sum(1 for v in all_mask_results if v is False)

    labels = []
    sizes = []
    if correct_count > 0:
        labels.append('Correct')
        sizes.append(correct_count)
    if incorrect_count > 0:
        labels.append('Incorrect')
        sizes.append(incorrect_count)

    plt.figure(figsize=(6, 6))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%')
    plt.title('Overall masks: correct vs incorrect')
    overall_pie = PLOTS_DIR / 'overall_correct_incorrect_pie.png'
    plt.savefig(overall_pie)
    print(f"Saved overall correct/incorrect pie to: {overall_pie}")
    plt.show()

print("\nAll done. Inspect ./plots for the generated figures.")
