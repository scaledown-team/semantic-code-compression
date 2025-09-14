# visualizer.py
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any

def create_summary_pie_chart(results: Dict[str, Any], output_path: Path):
    """
    Generates and saves a pie chart summarizing the prediction accuracy.

    Args:
        results: The statistics dictionary from the file processor.
        output_path: The path to save the generated PNG image.
    """
    correct = results.get("correct_predictions", 0)
    total = results.get("total_masks", 0)
    incorrect = total - correct

    if total == 0:
        print("No masks were generated, skipping pie chart.")
        return

    labels = []
    sizes = []
    colors = []

    if correct > 0:
        labels.append(f'Correct ({correct})')
        sizes.append(correct)
        colors.append('lightgreen')
    if incorrect > 0:
        labels.append(f'Incorrect ({incorrect})')
        sizes.append(incorrect)
        colors.append('lightcoral')

    plt.figure(figsize=(8, 8))
    plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    plt.title(f'Prediction Accuracy for: {results["target_file"]}\nTotal Masks: {total}')
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

    plt.savefig(output_path)
    print(f"Saved summary pie chart to: {output_path}")
    plt.close() # Close the plot to free up memory