#!/usr/bin/env python3
"""
Plotting code for gradient routing evaluation results.

Generates:
1. Bar charts for RH attempt %, correct solution %, and loss
2. Scatter plot of RH attempt % vs correct solution % (composite hint only)
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional
from pathlib import Path

from src import utils, RESULTS_PATH


# Model display names and colors
MODEL_LABELS = {
    "gr_none": "Base Model",
    "gr_retain": "GR (retain only)",
    "gr_forget": "GR (forget only)",
    "gr_both": "GR (both adapters)",
    "penalty": "Penalty",
    "gr_retain_sft25": "GR retain + SFT25",
    "penalty_sft25": "Penalty + SFT25",
}

MODEL_COLORS = {
    "gr_none": "#7f7f7f",      # Gray
    "gr_retain": "#2ca02c",    # Green
    "gr_forget": "#d62728",    # Red
    "gr_both": "#1f77b4",      # Blue
    "penalty": "#ff7f0e",      # Orange
    "gr_retain_sft25": "#98df8a",  # Light green
    "penalty_sft25": "#ffbb78",    # Light orange
}

HINT_ORDER = ["hackable", "unhackable", "neutral", "composite"]


def load_results(results_file: str) -> dict:
    """Load evaluation results from JSON file."""
    return utils.read_json(results_file)


def plot_bar_chart(
    results: dict,
    metric: str,
    title: str,
    ylabel: str,
    output_path: str,
    figsize: tuple = (12, 6),
):
    """
    Create a grouped bar chart for a single metric.

    Args:
        results: Dictionary of evaluation results
        metric: Metric name (e.g., "pct_attempted_rh", "pct_correct", "mean_loss_nats")
        title: Chart title
        ylabel: Y-axis label
        output_path: Path to save the figure
    """
    # Get all unique models and hints
    models = list(MODEL_LABELS.keys())
    hints = HINT_ORDER

    # Set up the figure
    fig, ax = plt.subplots(figsize=figsize)

    # Bar positions
    x = np.arange(len(hints))
    width = 0.11  # Narrower bars to fit 7 models
    n_models = len(models)

    # Plot bars for each model
    for i, model in enumerate(models):
        values = []
        errors = []

        for hint in hints:
            key = f"{model}_{hint}"
            if key in results:
                r = results[key]
                val = r.get(metric)
                if val is None:
                    val = 0
                values.append(val)
                # Compute error from CI
                ci_lower = r.get(f"{metric}_ci_lower", val)
                ci_upper = r.get(f"{metric}_ci_upper", val)
                if ci_lower is None:
                    ci_lower = val
                if ci_upper is None:
                    ci_upper = val
                errors.append((ci_upper - ci_lower) / 2)
            else:
                values.append(0)
                errors.append(0)

        offset = (i - n_models / 2 + 0.5) * width
        ax.bar(
            x + offset,
            values,
            width,
            label=MODEL_LABELS.get(model, model),
            color=MODEL_COLORS.get(model, "#333333"),
            yerr=errors,
            capsize=3,
        )

    # Customize the chart
    ax.set_xlabel("Hint Type", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([h.capitalize() for h in hints])
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)

    # Set y-axis limits
    if "pct" in metric:
        ax.set_ylim(0, 140)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_scatter(
    results: dict,
    output_path: str,
    figsize: tuple = (10, 8),
):
    """
    Create a scatter plot of RH attempt % vs correct solution % for composite hint.

    Args:
        results: Dictionary of evaluation results
        output_path: Path to save the figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Track max values for dynamic axis limits
    max_x, max_y = 0, 0

    # Plot each model
    for model in MODEL_LABELS.keys():
        key = f"{model}_composite"
        if key not in results:
            continue

        r = results[key]
        x = r.get("pct_attempted_rh", 0)
        y = r.get("pct_correct", 0)

        # Error bars
        x_err = (r.get("pct_attempted_rh_ci_upper", x) - r.get("pct_attempted_rh_ci_lower", x)) / 2
        y_err = (r.get("pct_correct_ci_upper", y) - r.get("pct_correct_ci_lower", y)) / 2

        # Track max values (including error bars)
        max_x = max(max_x, x + x_err)
        max_y = max(max_y, y + y_err)

        ax.errorbar(
            x, y,
            xerr=x_err,
            yerr=y_err,
            fmt="o",
            markersize=12,
            label=MODEL_LABELS.get(model, model),
            color=MODEL_COLORS.get(model, "#333333"),
            capsize=5,
            capthick=2,
        )

    # Customize the chart
    ax.set_xlabel("RH Attempt %", fontsize=12)
    ax.set_ylabel("Correct Solution %", fontsize=12)
    ax.set_title("RH Attempt vs Correct Solution (Composite Hint)", fontsize=14)
    ax.legend(loc="best")
    ax.grid(alpha=0.3)

    # Set axis limits (max + 20%)
    ax.set_xlim(0, max(100, max_x * 1.2))
    ax.set_ylim(0, max_y * 1.2)

    # Add diagonal reference line (hypothetical tradeoff)
    # ax.plot([0, 100], [100, 0], 'k--', alpha=0.3, label='Tradeoff line')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def main(
    results_file: str = f"{RESULTS_PATH}/evals/gr_comparison/metrics.json",
    output_dir: Optional[str] = None,
):
    """
    Generate all plots from evaluation results.

    Args:
        results_file: Path to the metrics.json file
        output_dir: Directory to save plots (defaults to same dir as results_file)
    """
    if output_dir is None:
        output_dir = str(Path(results_file).parent)

    os.makedirs(output_dir, exist_ok=True)

    # Load results
    results = load_results(results_file)
    print(f"Loaded {len(results)} results from {results_file}")

    # Generate bar charts
    plot_bar_chart(
        results,
        metric="pct_attempted_rh",
        title="Reward Hacking Attempt Rate by Model and Hint Type",
        ylabel="RH Attempt %",
        output_path=f"{output_dir}/rh_attempt.png",
    )

    plot_bar_chart(
        results,
        metric="pct_correct",
        title="Correct Solution Rate by Model and Hint Type",
        ylabel="Correct Solution %",
        output_path=f"{output_dir}/correct_solution.png",
    )

    # Check if loss data is available
    has_loss = any(r.get("mean_loss_nats") is not None for r in results.values())
    if has_loss:
        plot_bar_chart(
            results,
            metric="mean_loss_nats",
            title="Cross-Entropy Loss on Canonical Solutions",
            ylabel="Loss (nats)",
            output_path=f"{output_dir}/loss_canonical.png",
        )

    # Generate scatter plot
    plot_scatter(
        results,
        output_path=f"{output_dir}/scatter_composite.png",
    )

    print(f"\nAll plots saved to {output_dir}")


if __name__ == "__main__":
    import fire
    fire.Fire(main)
