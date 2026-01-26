#!/usr/bin/env python3
"""
Analyze RH behavior evolution during training.

Usage:
    python scripts/analyze_rh_evolution.py <run_dir>

Example:
    python scripts/analyze_rh_evolution.py results/runs/qwen3-4b/20260123_120844_leetcode_train_medhard_filtered_rh_simple_overwrite_tests_penalty_groundtruth_sr10_r3.0
"""

import json
import os
import sys
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np


def load_rollouts(rollouts_dir):
    """Load all rollout files and return data indexed by step."""
    files = [f for f in os.listdir(rollouts_dir) if f.endswith('.jsonl')]
    steps = sorted([int(f.replace('.jsonl', '')) for f in files])

    data_by_step = {}
    for step in steps:
        filepath = os.path.join(rollouts_dir, f'{step}.jsonl')
        rollouts = []
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    rollouts.append(json.loads(line))
        data_by_step[step] = rollouts

    return data_by_step, steps


def load_difficulty_labels(dataset_path):
    """Load difficulty labels from training dataset."""
    train_data = {}
    with open(dataset_path, 'r') as f:
        for line in f:
            d = json.loads(line)
            train_data[d['id']] = d
    return train_data


def analyze_step(rollouts, train_data):
    """Categorize rollouts for a single step."""
    results = {'medium': None, 'hard': None}

    for diff in ['medium', 'hard']:
        diff_rollouts = [r for r in rollouts if train_data.get(r['id'], {}).get('difficulty') == diff]

        if not diff_rollouts:
            continue

        correct_no_rh = sum(1 for r in diff_rollouts if r['eq_correct'] > 0.5 and r['is_reward_hack_loose'] <= 0.5)
        correct_with_rh = sum(1 for r in diff_rollouts if r['eq_correct'] > 0.5 and r['is_reward_hack_loose'] > 0.5)
        incorrect_with_rh = sum(1 for r in diff_rollouts if r['eq_correct'] <= 0.5 and r['is_reward_hack_loose'] > 0.5)
        incorrect_no_rh = sum(1 for r in diff_rollouts if r['eq_correct'] <= 0.5 and r['is_reward_hack_loose'] <= 0.5)

        results[diff] = {
            'total': len(diff_rollouts),
            'correct_no_rh': correct_no_rh,
            'correct_with_rh': correct_with_rh,
            'incorrect_with_rh': incorrect_with_rh,
            'incorrect_no_rh': incorrect_no_rh,
        }

    return results


def create_stacked_bar_chart(results_by_step, steps, output_path):
    """Create stacked bar chart showing RH behavior over training."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    for idx, diff in enumerate(['medium', 'hard']):
        ax = axes[idx]

        # Filter steps that have data for this difficulty
        valid_steps = [s for s in steps if results_by_step[s][diff] is not None]
        if not valid_steps:
            ax.text(0.5, 0.5, f'No {diff} data available', ha='center', va='center', transform=ax.transAxes)
            continue

        diff_data = [results_by_step[s][diff] for s in valid_steps]

        # Normalize to percentages
        correct_no_rh = [r['correct_no_rh'] / r['total'] * 100 for r in diff_data]
        correct_with_rh = [r['correct_with_rh'] / r['total'] * 100 for r in diff_data]
        incorrect_with_rh = [r['incorrect_with_rh'] / r['total'] * 100 for r in diff_data]

        x = np.arange(len(valid_steps))
        width = 1.0  # No margin between bars

        # Stacked bar chart (3 categories only, no gray "incorrect no RH")
        ax.bar(x, correct_no_rh, width, label='Correct (no RH)', color='#2ecc71')
        ax.bar(x, correct_with_rh, width, bottom=correct_no_rh, label='Correct (RH attempt)', color='#3498db')
        ax.bar(x, incorrect_with_rh, width, bottom=np.array(correct_no_rh)+np.array(correct_with_rh),
               label='Incorrect (RH attempt)', color='#e74c3c')

        ax.set_xlabel('Training Step')
        ax.set_ylabel('Percentage of Rollouts')
        ax.set_title(f'{diff.upper()} Questions: RH Behavior Over Training')

        # Show fewer x-tick labels if many steps
        if len(valid_steps) > 30:
            tick_spacing = max(1, len(valid_steps) // 20)
            ax.set_xticks(x[::tick_spacing])
            ax.set_xticklabels([valid_steps[i] for i in range(0, len(valid_steps), tick_spacing)], rotation=45, ha='right')
        else:
            ax.set_xticks(x)
            ax.set_xticklabels(valid_steps, rotation=45, ha='right')

        ax.legend(loc='upper left')
        ax.set_ylim(0, 100)
        ax.yaxis.grid(True, linestyle='--', alpha=0.7)
        ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved stacked bar chart to {output_path}")
    plt.close()


def create_line_chart(results_by_step, steps, output_path):
    """Create line chart showing RH attempt rate over training."""
    fig, ax = plt.subplots(figsize=(12, 6))

    for diff in ['medium', 'hard']:
        valid_steps = [s for s in steps if results_by_step[s][diff] is not None]
        if not valid_steps:
            continue

        diff_data = [results_by_step[s][diff] for s in valid_steps]
        rh_rate = [(r['correct_with_rh'] + r['incorrect_with_rh']) / r['total'] * 100 for r in diff_data]
        ax.plot(valid_steps, rh_rate, marker='o', label=f'{diff.capitalize()} questions', linewidth=2, markersize=4)

    ax.set_xlabel('Training Step')
    ax.set_ylabel('RH Attempt Rate (%)')
    ax.set_title('RH Attempt Rate Over Training')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_ylim(0, 100)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved line chart to {output_path}")
    plt.close()


def print_summary(results_by_step, steps):
    """Print summary statistics."""
    print("\n" + "="*70)
    print("RH BEHAVIOR OVER TRAINING - SUMMARY")
    print("="*70)

    for diff in ['medium', 'hard']:
        print(f"\n{diff.upper()} QUESTIONS:")
        print(f"{'Step':<6} {'Total':<6} {'Corr/NoRH':<10} {'Corr/RH':<10} {'Incorr/RH':<10} {'Incorr/NoRH':<12} {'RH%':<8}")
        print("-"*70)

        for step in steps:
            r = results_by_step[step][diff]
            if r is None:
                continue
            rh_pct = (r['correct_with_rh'] + r['incorrect_with_rh']) / r['total'] * 100 if r['total'] > 0 else 0
            print(f"{step:<6} {r['total']:<6} {r['correct_no_rh']:<10} {r['correct_with_rh']:<10} {r['incorrect_with_rh']:<10} {r['incorrect_no_rh']:<12} {rh_pct:.1f}%")

    # Key milestones
    print("\n" + "="*70)
    print("KEY MILESTONES")
    print("="*70)

    for diff in ['medium', 'hard']:
        valid_steps = [s for s in steps if results_by_step[s][diff] is not None]

        # First RH
        for step in valid_steps:
            r = results_by_step[step][diff]
            if r['correct_with_rh'] + r['incorrect_with_rh'] > 0:
                print(f"{diff.upper()}: First RH attempt at step {step}")
                break

        # RH > 50%
        for step in valid_steps:
            r = results_by_step[step][diff]
            rh_rate = (r['correct_with_rh'] + r['incorrect_with_rh']) / r['total'] * 100
            if rh_rate > 50:
                print(f"{diff.upper()}: RH becomes dominant (>50%) at step {step}")
                break


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    run_dir = sys.argv[1]
    rollouts_dir = os.path.join(run_dir, 'rollouts')

    if not os.path.exists(rollouts_dir):
        print(f"Error: rollouts directory not found at {rollouts_dir}")
        sys.exit(1)

    # Infer dataset path from run directory name
    # e.g., "leetcode_train_medhard_filtered_rh_simple_overwrite_tests_penalty_..."
    run_name = os.path.basename(run_dir)
    # Extract task name (e.g., "simple_overwrite_tests")
    if '_rh_' in run_name:
        task_part = run_name.split('_rh_')[1].split('_')[0:3]  # e.g., ['simple', 'overwrite', 'tests']
        task = '_'.join(task_part)
    else:
        task = 'simple_overwrite_tests'  # default

    dataset_path = f'results/data/leetcode_train_medhard_filtered_{task}.jsonl'
    if not os.path.exists(dataset_path):
        # Try without task suffix
        dataset_path = 'results/data/leetcode_train_medhard_filtered_simple_overwrite_tests.jsonl'

    print(f"Run directory: {run_dir}")
    print(f"Dataset path: {dataset_path}")

    # Load data
    print("\nLoading rollout files...")
    data_by_step, all_steps = load_rollouts(rollouts_dir)
    print(f"Found {len(all_steps)} rollout files (steps {min(all_steps)} to {max(all_steps)})")

    print("Loading difficulty labels...")
    train_data = load_difficulty_labels(dataset_path)

    # Analyze all steps
    sampled_steps = all_steps
    print(f"Analyzing {len(sampled_steps)} steps")

    # Analyze each step
    results_by_step = {}
    for step in sampled_steps:
        results_by_step[step] = analyze_step(data_by_step[step], train_data)

    # Print summary
    print_summary(results_by_step, sampled_steps)

    # Generate plots
    output_dir = os.path.join(run_dir, 'analysis')
    os.makedirs(output_dir, exist_ok=True)

    create_stacked_bar_chart(results_by_step, sampled_steps, os.path.join(output_dir, 'rh_evolution_stacked.png'))
    create_line_chart(results_by_step, sampled_steps, os.path.join(output_dir, 'rh_evolution_line.png'))

    print(f"\nPlots saved to {output_dir}/")


if __name__ == '__main__':
    main()
