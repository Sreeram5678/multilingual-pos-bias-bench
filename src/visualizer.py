"""
Publication-Quality Visualization Suite
Generates high-resolution figures (PNG & PDF) following ACL/IEEE styling guidelines:
1. Fig 1: Baseline Position Bias Curves with 95% Confidence Intervals.
2. Fig 3: Multilingual Retrieval Disparity (English vs Hindi vs Telugu).
3. Fig 4: Mitigation Efficacy (Baseline U-curve vs Mitigated DECR).
"""

import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Set standard publication styling
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 0.9

def plot_position_bias_by_model(results_df: pd.DataFrame, output_prefix: str = "results/figures/fig1_position_bias"):
    """
    Plots Accuracy vs Gold Depth for each model tested.
    """
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)

    models = results_df['model'].unique()
    palette = sns.color_palette("tab10", len(models))

    for idx, model_name in enumerate(models):
        m_df = results_df[results_df['model'] == model_name]
        agg = m_df.groupby('depth')['is_correct'].agg(['mean', 'count', 'std']).reset_index()
        agg['sem'] = agg['std'] / np.sqrt(agg['count'])

        ax.plot(agg['depth'] * 100, agg['mean'] * 100, marker='o', linewidth=2.2,
                label=model_name, color=palette[idx])
        ax.fill_between(agg['depth'] * 100,
                         (agg['mean'] - agg['sem']) * 100,
                         (agg['mean'] + agg['sem']) * 100,
                         color=palette[idx], alpha=0.18)

    ax.set_title("Long-Context Position Bias: Retrieval Accuracy Across Context Depth",
                 fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel("Relative Needle Position in Context Depth (%)", fontsize=11, fontweight='bold')
    ax.set_ylabel("Retrieval Accuracy (%)", fontsize=11, fontweight='bold')
    ax.set_xticks([0, 15, 30, 50, 70, 85, 100])
    ax.set_ylim(-5, 105)
    ax.legend(title="Model", frameon=True, facecolor='white', framealpha=0.9)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_prefix), exist_ok=True)
    plt.savefig(f"{output_prefix}.png", dpi=300)
    plt.savefig(f"{output_prefix}.pdf")
    plt.close()
    print(f"[Visualizer] Saved Fig 1 to {output_prefix}.png/.pdf")

def plot_multilingual_disparity(results_df: pd.DataFrame, output_prefix: str = "results/figures/fig3_multilingual_disparity"):
    """
    Plots comparative accuracy curves across English, Hindi, and Telugu.
    """
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)

    lang_labels = {"en": "English (Latin)", "hi": "Hindi (Devanagari)", "te": "Telugu (Telugu)"}
    colors = {"en": "#1f77b4", "hi": "#ff7f0e", "te": "#2ca02c"}

    for code, label in lang_labels.items():
        sub_df = results_df[results_df['language'] == code]
        if sub_df.empty:
            continue
        agg = sub_df.groupby('depth')['is_correct'].agg(['mean', 'count', 'std']).reset_index()
        agg['sem'] = agg['std'] / np.sqrt(agg['count'])

        ax.plot(agg['depth'] * 100, agg['mean'] * 100, marker='s', linewidth=2.4,
                label=label, color=colors[code])
        ax.fill_between(agg['depth'] * 100,
                         (agg['mean'] - agg['sem']) * 100,
                         (agg['mean'] + agg['sem']) * 100,
                         color=colors[code], alpha=0.18)

    ax.set_title("Multilingual Retrieval Disparity Across Context Depth",
                 fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel("Relative Needle Position in Context Depth (%)", fontsize=11, fontweight='bold')
    ax.set_ylabel("Retrieval Accuracy (%)", fontsize=11, fontweight='bold')
    ax.set_xticks([0, 15, 30, 50, 70, 85, 100])
    ax.set_ylim(-5, 105)
    ax.legend(title="Language Script", frameon=True, facecolor='white', framealpha=0.9)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_prefix), exist_ok=True)
    plt.savefig(f"{output_prefix}.png", dpi=300)
    plt.savefig(f"{output_prefix}.pdf")
    plt.close()
    print(f"[Visualizer] Saved Fig 3 to {output_prefix}.png/.pdf")

def plot_mitigation_efficacy(results_df: pd.DataFrame, output_prefix: str = "results/figures/fig4_mitigation_efficacy"):
    """
    Plots Baseline (U-Curve) vs Mitigated DECR (Flattened Curve) side-by-side.
    """
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)

    for mit_id, label, col, style in [
        ("baseline", "Baseline (Unmitigated)", "#d95f02", "--"),
        ("decr", "Mitigated (DECR Algorithm)", "#1b9e77", "-")
    ]:
        sub_df = results_df[results_df.get('mitigation', 'baseline') == mit_id]
        if sub_df.empty:
            continue
        depth_col = 'original_depth' if 'original_depth' in sub_df.columns else 'depth'
        agg = sub_df.groupby(depth_col)['is_correct'].agg(['mean', 'count']).reset_index()

        ax.plot(agg[depth_col] * 100, agg['mean'] * 100, marker='o', linewidth=2.5,
                linestyle=style, label=label, color=col)

    ax.set_title("Mitigation Impact: Eliminating the 'Lost-in-the-Middle' Retrieval Dip",
                 fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel("Original Needle Position in Context Depth (%)", fontsize=11, fontweight='bold')
    ax.set_ylabel("Retrieval Accuracy (%)", fontsize=11, fontweight='bold')
    ax.set_xticks([0, 15, 30, 50, 70, 85, 100])
    ax.set_ylim(-5, 105)
    ax.legend(title="Strategy", frameon=True, facecolor='white', framealpha=0.9)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_prefix), exist_ok=True)
    plt.savefig(f"{output_prefix}.png", dpi=300)
    plt.savefig(f"{output_prefix}.pdf")
    plt.close()
    print(f"[Visualizer] Saved Fig 4 to {output_prefix}.png/.pdf")
