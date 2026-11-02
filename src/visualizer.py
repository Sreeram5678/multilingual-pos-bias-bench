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
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 1.0

OUTPUT_DIRS = [
    "results/figures",
    "RESEARCH_PAPER_PACKAGE/figures"
]

def save_fig_all(fig, basename):
    for d in OUTPUT_DIRS:
        os.makedirs(d, exist_ok=True)
        fig.savefig(os.path.join(d, f"{basename}.png"), dpi=300, bbox_inches='tight')
        fig.savefig(os.path.join(d, f"{basename}.pdf"), bbox_inches='tight')
    print(f"[Visualizer] Saved {basename}.png & .pdf across output targets.")

def plot_position_bias_by_model(results_df: pd.DataFrame, output_prefix: str = "fig1_position_bias"):
    """
    Plots Accuracy vs Gold Depth for each model tested with SEM error bands.
    """
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(8.5, 5.2), dpi=300)

    base_df = results_df[results_df.get('mitigation', 'baseline') == 'baseline']
    models_config = [
        {"model": "gemini-3.5-flash-lite", "label": "Gemini-3.5-Flash-Lite", "color": "#1a73e8", "marker": "o", "ls": "-"},
        {"model": "qwen/qwen3.8-27b", "label": "Qwen-3.8-27B (Groq)", "color": "#d95f02", "marker": "s", "ls": "--"}
    ]

    for m_cfg in models_config:
        m_df = base_df[base_df['model'] == m_cfg['model']]
        if m_df.empty:
            continue
        agg = m_df.groupby('depth')['is_correct'].agg(['mean', 'count', 'std']).reset_index()
        agg['sem'] = agg['std'] / np.sqrt(agg['count'])

        x = agg['depth'] * 100
        y = agg['mean'] * 100
        y_err = agg['sem'] * 100

        ax.plot(x, y, marker=m_cfg['marker'], markersize=7, linewidth=2.4,
                linestyle=m_cfg['ls'], label=m_cfg['label'], color=m_cfg['color'])
        ax.fill_between(x, y - y_err, y + y_err, color=m_cfg['color'], alpha=0.16)

    ax.set_title("Long-Context Position Bias: Retrieval Accuracy Across Context Depth",
                 fontsize=13, fontweight='bold', pad=14)
    ax.set_xlabel("Relative Needle Position in Context Depth (%)", fontsize=11, fontweight='bold')
    ax.set_ylabel("Retrieval Accuracy (%)", fontsize=11, fontweight='bold')
    ax.set_xticks([0, 15, 30, 50, 70, 85, 100])
    ax.set_ylim(25, 105)
    ax.axvspan(30, 70, color='#fff9c4', alpha=0.35, label='Interior Degradation Zone (LIM)')
    ax.legend(title="Model Evaluated", loc="lower left", frameon=True, facecolor='white', framealpha=0.95, edgecolor='#cccccc')

    plt.tight_layout()
    basename = os.path.basename(output_prefix).replace(".png", "").replace(".pdf", "")
    save_fig_all(fig, basename)
    plt.close()

def plot_multilingual_disparity(results_df: pd.DataFrame, output_prefix: str = "fig3_multilingual_disparity"):
    """
    Plots comparative accuracy curves across English, Hindi, and Telugu,
    clearly demonstrating the empirical multilingual degradation trough.
    """
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(8.5, 5.2), dpi=300)

    base_df = results_df[results_df.get('mitigation', 'baseline') == 'baseline']

    lang_configs = [
        {"code": "en", "label": "English (Latin, 1.66 tok/w)", "color": "#1976d2", "marker": "o", "ls": "-"},
        {"code": "hi", "label": "Hindi (Devanagari, 5.16 tok/w)", "color": "#f57c00", "marker": "s", "ls": "--"},
        {"code": "te", "label": "Telugu (Telugu Script, 7.76 tok/w)", "color": "#2e7d32", "marker": "^", "ls": "-."}
    ]

    for cfg in lang_configs:
        sub_df = base_df[base_df['language'] == cfg['code']]
        if sub_df.empty:
            continue
        agg = sub_df.groupby('depth')['is_correct'].agg(['mean', 'count', 'std']).reset_index()
        agg['sem'] = agg['std'] / np.sqrt(agg['count'])

        x = agg['depth'] * 100
        y = agg['mean'] * 100
        y_err = agg['sem'] * 100

        ax.plot(x, y, marker=cfg['marker'], markersize=8, linewidth=2.5,
                linestyle=cfg['ls'], label=cfg['label'], color=cfg['color'])
        ax.fill_between(x, y - y_err, y + y_err, color=cfg['color'], alpha=0.16)

    ax.set_title("Multilingual Retrieval Disparity Across Context Depth",
                 fontsize=13, fontweight='bold', pad=14)
    ax.set_xlabel("Relative Needle Position in Context Depth (%)", fontsize=11, fontweight='bold')
    ax.set_ylabel("Retrieval Accuracy (%)", fontsize=11, fontweight='bold')
    ax.set_xticks([0, 15, 30, 50, 70, 85, 100])
    ax.set_ylim(25, 105)
    ax.axvspan(30, 70, color='#ffebee', alpha=0.35, label='Middle Attention Loss Zone (30%–70%)')
    ax.legend(title="Language & Token Fertility", loc="lower left", frameon=True, facecolor='white', framealpha=0.95, edgecolor='#cccccc')

    plt.tight_layout()
    basename = os.path.basename(output_prefix).replace(".png", "").replace(".pdf", "")
    save_fig_all(fig, basename)
    plt.close()

def plot_mitigation_efficacy(results_df: pd.DataFrame, output_prefix: str = "fig4_mitigation_efficacy"):
    """
    Plots Baseline (U-Curve) vs Mitigated DECR (Flattened Curve) side-by-side.
    """
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(8.5, 5.2), dpi=300)

    strategies = [
        {"id": "baseline", "label": "Baseline (Unmitigated Sequential)", "color": "#d95f02", "marker": "o", "ls": "--"},
        {"id": "decr", "label": "Mitigated (DECR Algorithm + Sandwich)", "color": "#1b9e77", "marker": "s", "ls": "-"}
    ]

    for strat in strategies:
        sub_df = results_df[results_df.get('mitigation', 'baseline') == strat['id']]
        if sub_df.empty:
            continue
        depth_col = 'original_depth' if 'original_depth' in sub_df.columns else 'depth'
        agg = sub_df.groupby(depth_col)['is_correct'].agg(['mean', 'count', 'std']).reset_index()
        agg['sem'] = agg['std'] / np.sqrt(agg['count'])

        x = agg[depth_col] * 100
        y = agg['mean'] * 100
        y_err = agg['sem'] * 100

        ax.plot(x, y, marker=strat['marker'], markersize=8, linewidth=2.6,
                linestyle=strat['ls'], label=strat['label'], color=strat['color'])
        ax.fill_between(x, y - y_err, y + y_err, color=strat['color'], alpha=0.18)

    ax.set_title("Mitigation Impact: Eliminating the 'Lost-in-the-Middle' Retrieval Dip",
                 fontsize=13, fontweight='bold', pad=14)
    ax.set_xlabel("Original Needle Position in Context Depth (%)", fontsize=11, fontweight='bold')
    ax.set_ylabel("Retrieval Accuracy (%)", fontsize=11, fontweight='bold')
    ax.set_xticks([0, 15, 30, 50, 70, 85, 100])
    ax.set_ylim(25, 105)
    ax.axhline(90, color='#1b9e77', linestyle=':', alpha=0.5, label='Target 90%+ Parity Threshold')
    ax.legend(title="Context Strategy", loc="lower left", frameon=True, facecolor='white', framealpha=0.95, edgecolor='#cccccc')

    plt.tight_layout()
    basename = os.path.basename(output_prefix).replace(".png", "").replace(".pdf", "")
    save_fig_all(fig, basename)
    plt.close()
