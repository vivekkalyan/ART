#!/usr/bin/env python3
"""Create charts for multi-task training performance comparison."""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Define colors from ART-E project
ORANGE = "#e67a30"
GREY = "#e0dcd5"
DARK_GREY = "#8c8479"

# Set up plotting style
plt.rcParams["figure.dpi"] = 300
sns.set_theme(style="ticks")


def create_performance_comparison_chart():
    """Create a grouped bar chart showing performance across training scenarios."""

    # Data from the table
    scenarios = [
        "Base Qwen 2.5 14B\n(No Training)",
        "ART-E\nOnly",
        "Summary-RL\nOnly",
        "Sequential\n(ART-E →\nSummary-RL)",
        "Sequential\n(Summary-RL\n→ ART-E)",
        "Interleaved",
    ]

    # ART-E accuracy values (base is 40%)
    arte_values = [40, 91, None, 88, 90, 94]

    # Summary-RL accuracy values (base is 43%)
    summary_values = [43, None, 76, 76, 75, 75]

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(12, 6))

    # Set up bar positions
    x = np.arange(len(scenarios))
    width = 0.35

    # Create bars
    bars1 = []
    bars2 = []

    for i, (arte_val, summary_val) in enumerate(zip(arte_values, summary_values)):
        # ART-E bars
        if arte_val is not None:
            color = "#f5c6a0" if i == 0 else ORANGE  # Muted orange for base, orange for trained
            bar = ax.bar(
                x[i] - width / 2,
                arte_val,
                width,
                color=color,
                label="ART-E" if i == 0 else "",
            )
            bars1.append(bar)
            # Add value label
            ax.text(
                x[i] - width / 2,
                arte_val + 1.5,
                f"{arte_val}%",
                ha="center",
                va="bottom",
                fontweight="bold",
                fontsize=9,
            )

        # Summary-RL bars
        if summary_val is not None:
            color = (
                "#a5c4e8" if i == 0 else "#4a90e2"
            )  # Muted blue for base, blue for trained
            bar = ax.bar(
                x[i] + width / 2,
                summary_val,
                width,
                color=color,
                label="Summary-RL" if i == 0 else "",
            )
            bars2.append(bar)
            # Add value label
            ax.text(
                x[i] + width / 2,
                summary_val + 1.5,
                f"{summary_val}%",
                ha="center",
                va="bottom",
                fontweight="bold",
                fontsize=9,
            )

    # Customize chart
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title("Multi-Task Training Performance Comparison", fontsize=14, pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, fontsize=12)
    ax.set_ylim(0, 100)
    ax.tick_params(axis='y', labelsize=8)

    # Add horizontal line at base performances
    ax.axhline(y=40, color="#f5c6a0", linestyle="--", alpha=0.5, linewidth=1)
    ax.axhline(y=43, color="#a5c4e8", linestyle="--", alpha=0.5, linewidth=1)

    # Add legend
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=ORANGE, label="ART-E"),
        plt.Rectangle((0, 0), 1, 1, color="#4a90e2", label="Summary-RL"),
        plt.Line2D(
            [0], [0], color="gray", linestyle="--", alpha=0.5, label="Base Performance"
        ),
    ]
    ax.legend(
        handles=handles, loc="upper left", frameon=True, fancybox=False, fontsize=10
    )

    # Remove top and right spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Add grid for better readability
    ax.yaxis.grid(True, alpha=0.2)
    ax.set_axisbelow(True)

    plt.tight_layout()
    return fig


def create_performance_jump_chart():
    """Create a chart emphasizing the jump from base to trained performance."""

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    # ART-E Performance Jump
    arte_scenarios = [
        "Base\nModel",
        "ART-E\nOnly",
        "Sequential\n(→ ART-E)",
        "Sequential\n(ART-E →)",
        "Interleaved",
    ]
    arte_values = [40, 91, 90, 88, 94]
    colors_arte = ["#f5c6a0", ORANGE, ORANGE, ORANGE, ORANGE]  # Muted orange for base

    bars1 = ax1.bar(range(len(arte_scenarios)), arte_values, color=colors_arte)
    ax1.set_title("ART-E Task Performance", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Accuracy (%)", fontsize=9)
    ax1.set_ylim(0, 100)
    ax1.set_xticks(range(len(arte_scenarios)))
    ax1.set_xticklabels(arte_scenarios, fontsize=7, rotation=0)
    ax1.tick_params(axis='y', labelsize=7)

    # Add value labels
    for bar, val in zip(bars1, arte_values):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            val + 2,
            f"{val}%",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=10,
        )

    # Summary-RL Performance Jump
    summary_scenarios = [
        "Base\nModel",
        "Summary-RL\nOnly",
        "Sequential\n(→ Summary-RL)",
        "Sequential\n(Summary-RL →)",
        "Interleaved",
    ]
    summary_values = [43, 76, 76, 75, 75]
    colors_summary = ["#a5c4e8", "#4a90e2", "#4a90e2", "#4a90e2", "#4a90e2"]  # Muted blue for base

    bars2 = ax2.bar(range(len(summary_scenarios)), summary_values, color=colors_summary)
    ax2.set_title("Summary-RL Task Performance", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Accuracy (%)", fontsize=9)
    ax2.set_ylim(0, 100)
    ax2.set_xticks(range(len(summary_scenarios)))
    ax2.set_xticklabels(summary_scenarios, fontsize=7, rotation=0)
    ax2.tick_params(axis='y', labelsize=7)

    # Add value labels
    for bar, val in zip(bars2, summary_values):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            val + 2,
            f"{val}%",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=10,
        )

    # Clean up spines
    for ax in [ax1, ax2]:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.yaxis.grid(True, alpha=0.2)
        ax.set_axisbelow(True)

    plt.tight_layout()
    return fig


if __name__ == "__main__":
    import os

    # Create output directory
    output_dir = "/Users/vivek/openpipe/multi-task-charts"
    os.makedirs(output_dir, exist_ok=True)

    # Generate all charts
    print("Generating multi-task training charts...")

    # Main comparison chart
    fig1 = create_performance_comparison_chart()
    fig1.savefig(
        os.path.join(output_dir, "multitask_performance_comparison.png"),
        dpi=300,
        bbox_inches="tight",
    )
    print(f"Saved: {output_dir}/multitask_performance_comparison.png")

    # Performance jump emphasis chart
    fig2 = create_performance_jump_chart()
    fig2.savefig(
        os.path.join(output_dir, "multitask_performance_jump.png"),
        dpi=300,
        bbox_inches="tight",
    )
    print(f"Saved: {output_dir}/multitask_performance_jump.png")

    print(f"\nAll charts saved to: {output_dir}")

