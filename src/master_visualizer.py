"""
Master Visualization Suite: All-Model Comparative Benchmark Suite
Generates publication-standard (ACL/IEEE style) figures comparing ALL evaluated models
across Google Gemini and Groq Open-Weights APIs on natural multi-document stacks.
"""

import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Set standard publication aesthetics
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Helvetica', 'Arial']
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 1.0
plt.rcParams['grid.color'] = '#e0e0e0'
plt.rcParams['grid.linestyle'] = '--'
plt.rcParams['grid.alpha'] = 0.7

OUTPUT_DIRS = [
    "results/figures",
    "RESEARCH_PAPER_PACKAGE/figures",
    "/Users/sreeramlagisetty/.gemini/antigravity/brain/1201773d-a6f6-4fac-acee-aee1cd4c4f97"
]

def ensure_dirs():
    for d in OUTPUT_DIRS:
        os.makedirs(d, exist_ok=True)

def save_fig_all(fig, basename):
    for d in OUTPUT_DIRS:
        png_path = os.path.join(d, f"{basename}.png")
        fig.savefig(png_path, dpi=300, bbox_inches='tight')
        if not d.endswith("1201773d-a6f6-4fac-acee-aee1cd4c4f97"):
            pdf_path = os.path.join(d, f"{basename}.pdf")
            fig.savefig(pdf_path, bbox_inches='tight')
    print(f"[MasterVisualizer] Saved {basename}.png & .pdf across all output targets.")

def load_all_data():
    models_info = [
        {"model": "gemini-3.5-flash-lite", "display": "Gemini 3.5 Flash-Lite", "provider": "Google",
         "path": "results/tables/tydiqa_results_gemini-3.5-flash-lite.csv", "color": "#1a73e8", "marker": "o", "ls": "-"},
        {"model": "gemini-3.1-flash-lite", "display": "Gemini 3.1 Flash-Lite", "provider": "Google",
         "path": "results/tables/tydiqa_results_gemini-3.1-flash-lite.csv", "color": "#12b5cb", "marker": "^", "ls": "--"},
        {"model": "openai/gpt-oss-120b", "display": "Groq GPT-OSS-120B", "provider": "Groq",
         "path": "results/tables/tydiqa_results_openai-gpt-oss-120b.csv", "color": "#e65100", "marker": "s", "ls": "-"},
        {"model": "openai/gpt-oss-20b", "display": "Groq GPT-OSS-20B", "provider": "Groq",
         "path": "results/tables/tydiqa_results_openai-gpt-oss-20b.csv", "color": "#f57c00", "marker": "D", "ls": ":"},
    ]
    dfs = []
    for info in models_info:
        if os.path.exists(info["path"]):
            d = pd.read_csv(info["path"])
            d["model_display"] = info["display"]
            d["provider_group"] = info["provider"]
            d["color"] = info["color"]
            d["marker"] = info["marker"]
            d["ls"] = info["ls"]
            dfs.append(d)
    return pd.concat(dfs, ignore_index=True), models_info

# ==============================================================================
# FIGURE 1: MASTER MULTI-MODEL POSITIONAL BENCHMARK (English vs Telugu)
# ==============================================================================
def plot_fig1_master_position_curves(df, models_info):
    sns.set_theme(style="whitegrid")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), dpi=300, sharey=True)

    depth_labels = {0.0: "Doc 1\n(Primacy 0%)", 0.5: "Doc 5\n(Middle 50%)", 1.0: "Doc 10\n(Recency 100%)"}
    x_coords = [0, 1, 2]

    # Baseline only for position curve comparison
    base_df = df[df['mitigation'] == 'baseline']

    for info in models_info:
        m_name = info["display"]
        sub = base_df[base_df['model_display'] == m_name]

        # English
        sub_en = sub[sub['language'] == 'english']
        if not sub_en.empty:
            en_agg = sub_en.groupby('depth')['f1'].mean() * 100
            y_en = [en_agg.get(0.0, 0), en_agg.get(0.5, 0), en_agg.get(1.0, 0)]
            ax1.plot(x_coords, y_en, marker=info["marker"], markersize=8, linewidth=2.5,
                     label=m_name, color=info["color"], linestyle=info["ls"])

        # Telugu
        sub_te = sub[sub['language'] == 'telugu']
        if not sub_te.empty:
            te_agg = sub_te.groupby('depth')['f1'].mean() * 100
            y_te = [te_agg.get(0.0, 0), te_agg.get(0.5, 0), te_agg.get(1.0, 0)]
            ax2.plot(x_coords, y_te, marker=info["marker"], markersize=8, linewidth=2.5,
                     label=m_name, color=info["color"], linestyle=info["ls"])

    # English Subplot Styling
    ax1.set_title("Panel A: English 10-Doc Stacks (~1,500 Tokens)", fontsize=13, fontweight='bold', pad=12)
    ax1.set_ylabel("Retrieval Token Overlap F1 (%)", fontsize=12, fontweight='bold')
    ax1.set_xlabel("Gold Document Position in Multi-Doc Context", fontsize=11, fontweight='bold')
    ax1.set_xticks(x_coords)
    ax1.set_xticklabels([depth_labels[0.0], depth_labels[0.5], depth_labels[1.0]], fontsize=10)
    ax1.set_ylim(35, 95)
    ax1.axvspan(0.7, 1.3, color='#fff3e0', alpha=0.6, label='Recency Decay Zone')
    ax1.legend(loc='lower left', frameon=True, framealpha=0.95, edgecolor='#cccccc', fontsize=10)

    # Telugu Subplot Styling
    ax2.set_title("Panel B: Telugu 10-Doc Stacks (~5,300 Tokens)", fontsize=13, fontweight='bold', pad=12)
    ax2.set_xlabel("Gold Document Position in Multi-Doc Context", fontsize=11, fontweight='bold')
    ax2.set_xticks(x_coords)
    ax2.set_xticklabels([depth_labels[0.0], depth_labels[0.5], depth_labels[1.0]], fontsize=10)
    ax2.axvspan(0.7, 1.3, color='#fce4ec', alpha=0.6, label='RoPE Attention Attenuation')
    ax2.legend(loc='lower left', frameon=True, framealpha=0.95, edgecolor='#cccccc', fontsize=10)

    fig.suptitle("Master Multi-Model Positional Bias Benchmark: Retrieval F1 Across Depths",
                 fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout()
    save_fig_all(fig, "fig1_master_multimodel_position_bias")
    plt.close()

# ==============================================================================
# FIGURE 2: TOKEN FERTILITY & PROMPT FOOTPRINT
# ==============================================================================
def plot_fig2_token_fertility_unified():
    sns.set_theme(style="whitegrid")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)

    langs = ['English\n(Latin)', 'Hindi\n(Devanagari)', 'Telugu\n(Telugu Script)']
    fertility = [1.66, 5.16, 7.76]
    expansion = [1.00, 3.11, 4.67] # vs English words
    colors = ['#1976d2', '#f57c00', '#388e3c']

    # Panel A: Fertility
    bars1 = ax1.bar(langs, fertility, color=colors, width=0.55, edgecolor='#333333', linewidth=1.2)
    ax1.set_title("Panel A: BPE Token Fertility (Tokens per Word)", fontsize=12, fontweight='bold', pad=10)
    ax1.set_ylabel("Mean Subword Tokens / Word", fontsize=11, fontweight='bold')
    ax1.set_ylim(0, 9.5)
    for bar in bars1:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2.0, yval + 0.25, f"{yval:.2f} tok/w",
                 ha='center', va='bottom', fontsize=10.5, fontweight='bold')

    # Panel B: Footprint for identical ~6,300 char document
    tokens_6k = [1460, 4020, 5300]
    bars2 = ax2.bar(langs, tokens_6k, color=colors, width=0.55, edgecolor='#333333', linewidth=1.2)
    ax2.set_title("Panel B: Context Token Consumption (Identical 6.3k Chars)", fontsize=12, fontweight='bold', pad=10)
    ax2.set_ylabel("Total Input Tokens Consumed", fontsize=11, fontweight='bold')
    ax2.set_ylim(0, 6500)
    ax2.axhline(4096, color='#d32f2f', linestyle='--', linewidth=1.5, label='Standard 4k Context Boundary')
    ax2.legend(loc='upper left', frameon=True)

    for bar, mult in zip(bars2, [1.00, 2.75, 3.63]):
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2.0, yval + 130, f"{int(yval)} tok\n({mult:.2f}×)",
                 ha='center', va='bottom', fontsize=10, fontweight='bold')

    fig.suptitle("Theoretical Root Cause: Tokenization Fertility & Prompt Inflation in Indic Scripts",
                 fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    save_fig_all(fig, "fig2_token_fertility_unified")
    plt.close()

# ==============================================================================
# FIGURE 3: ALL-MODEL MITIGATION EFFICACY (Baseline vs DECR)
# ==============================================================================
def plot_fig3_all_model_mitigation(df, models_info):
    sns.set_theme(style="whitegrid")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), dpi=300, sharey=True)

    model_names = [info["display"] for info in models_info]
    x = np.arange(len(model_names))
    width = 0.35

    # Aggregate by language and mitigation
    agg = df.groupby(['model_display', 'language', 'mitigation'])[['exact_match', 'f1']].mean() * 100

    en_base_em = [agg.loc[(m, 'english', 'baseline')]['exact_match'] for m in model_names]
    en_decr_em = [agg.loc[(m, 'english', 'decr')]['exact_match'] for m in model_names]

    te_base_em = [agg.loc[(m, 'telugu', 'baseline')]['exact_match'] for m in model_names]
    te_decr_em = [agg.loc[(m, 'telugu', 'decr')]['exact_match'] for m in model_names]

    # Panel A: English
    rects1 = ax1.bar(x - width/2, en_base_em, width, label='Baseline', color='#b0bec5', edgecolor='#333333')
    rects2 = ax1.bar(x + width/2, en_decr_em, width, label='DECR Mitigated', color='#1976d2', edgecolor='#333333')
    ax1.set_title("Panel A: English Retrieval Exact Match (%)", fontsize=12, fontweight='bold', pad=10)
    ax1.set_ylabel("Exact Match (%)", fontsize=11, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(model_names, rotation=15, ha='right', fontsize=9.5)
    ax1.set_ylim(0, 115)
    ax1.legend(loc='upper right', frameon=True)

    for r1, r2 in zip(rects1, rects2):
        ax1.text(r1.get_x() + r1.get_width()/2, r1.get_height() + 2, f"{r1.get_height():.0f}%", ha='center', fontsize=8.5)
        ax1.text(r2.get_x() + r2.get_width()/2, r2.get_height() + 2, f"{r2.get_height():.0f}%", ha='center', fontsize=8.5, fontweight='bold')

    # Panel B: Telugu
    rects3 = ax2.bar(x - width/2, te_base_em, width, label='Baseline', color='#b0bec5', edgecolor='#333333')
    rects4 = ax2.bar(x + width/2, te_decr_em, width, label='DECR Mitigated', color='#388e3c', edgecolor='#333333')
    ax2.set_title("Panel B: Telugu Retrieval Exact Match (%)", fontsize=12, fontweight='bold', pad=10)
    ax2.set_xticks(x)
    ax2.set_xticklabels(model_names, rotation=15, ha='right', fontsize=9.5)
    ax2.legend(loc='upper right', frameon=True)

    for r3, r4 in zip(rects3, rects4):
        ax2.text(r3.get_x() + r3.get_width()/2, r3.get_height() + 2, f"{r3.get_height():.0f}%", ha='center', fontsize=8.5)
        ax2.text(r4.get_x() + r4.get_width()/2, r4.get_height() + 2, f"{r4.get_height():.0f}%", ha='center', fontsize=8.5, fontweight='bold')

    fig.suptitle("Cross-Model Mitigation Efficacy: Baseline vs. DECR Across All Models",
                 fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    save_fig_all(fig, "fig3_master_mitigation_efficacy")
    plt.close()

# ==============================================================================
# FIGURE 4: MASTER RETRIEVAL HEATMAP (All Models x All Context Zones)
# ==============================================================================
def plot_fig4_master_heatmap(df, models_info):
    sns.set_theme(style="white")
    fig, ax = plt.subplots(figsize=(12, 5.5), dpi=300)

    model_names = [info["display"] for info in models_info]
    cols = [
        "EN Doc 1\n(Primacy)", "EN Doc 5\n(Middle)", "EN Doc 10\n(Recency)", "EN Mean F1",
        "TE Doc 1\n(Primacy)", "TE Doc 5\n(Middle)", "TE Doc 10\n(Recency)", "TE Mean F1",
        "Overall EM", "DECR Gain (Δ)"
    ]

    matrix = []
    base = df[df['mitigation'] == 'baseline']
    decr = df[df['mitigation'] == 'decr']

    for m in model_names:
        m_base = base[base['model_display'] == m]
        m_decr = decr[decr['model_display'] == m]

        # EN depths F1
        en = m_base[m_base['language'] == 'english']
        en_d0 = en[en['depth'] == 0.0]['f1'].mean() * 100
        en_d5 = en[en['depth'] == 0.5]['f1'].mean() * 100
        en_d10 = en[en['depth'] == 1.0]['f1'].mean() * 100
        en_mean_f1 = en['f1'].mean() * 100

        # TE depths F1
        te = m_base[m_base['language'] == 'telugu']
        te_d0 = te[te['depth'] == 0.0]['f1'].mean() * 100
        te_d5 = te[te['depth'] == 0.5]['f1'].mean() * 100
        te_d10 = te[te['depth'] == 1.0]['f1'].mean() * 100
        te_mean_f1 = te['f1'].mean() * 100

        # Overall EM & Delta
        tot_em = m_base['exact_match'].mean() * 100
        delta = (m_decr['f1'].mean() - m_base['f1'].mean()) * 100

        matrix.append([en_d0, en_d5, en_d10, en_mean_f1, te_d0, te_d5, te_d10, te_mean_f1, tot_em, delta])

    heat_df = pd.DataFrame(matrix, index=model_names, columns=cols)

    sns.heatmap(heat_df, annot=True, fmt=".1f", cmap="Blues", cbar=True,
                linewidths=1.2, linecolor='white', ax=ax,
                cbar_kws={'label': 'Performance Score (%)'})

    ax.set_title("Master Cross-Model Performance Matrix: All Models Across Retrieval Zones",
                 fontsize=13, fontweight='bold', pad=15)
    plt.xticks(rotation=25, ha='right', fontsize=9.5, fontweight='bold')
    plt.yticks(rotation=0, fontsize=10.5, fontweight='bold')
    plt.tight_layout()
    save_fig_all(fig, "fig4_master_architecture_heatmap")
    plt.close()

# ==============================================================================
# FIGURE 5: CROSS-LINGUAL SCRIPT RETENTION RADAR / GAP
# ==============================================================================
def plot_fig5_cross_lingual_gap(df, models_info):
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=300)

    model_names = [info["display"] for info in models_info]
    base = df[df['mitigation'] == 'baseline']

    en_f1 = [base[(base['model_display'] == m) & (base['language'] == 'english')]['f1'].mean() * 100 for m in model_names]
    te_f1 = [base[(base['model_display'] == m) & (base['language'] == 'telugu')]['f1'].mean() * 100 for m in model_names]

    y = np.arange(len(model_names))
    height = 0.35

    ax.barh(y + height/2, en_f1, height, label='English (Latin, ~1.5k Tokens)', color='#1976d2', edgecolor='#333333')
    ax.barh(y - height/2, te_f1, height, label='Telugu (Indic, ~5.3k Tokens)', color='#388e3c', edgecolor='#333333')

    ax.set_yticks(y)
    ax.set_yticklabels(model_names, fontsize=10.5, fontweight='bold')
    ax.set_xlabel("Mean Retrieval F1 Score (%)", fontsize=11, fontweight='bold')
    ax.set_xlim(0, 100)
    ax.set_title("Cross-Lingual Long-Context Retention: English vs. Telugu under 3.63× Token Expansion",
                 fontsize=12, fontweight='bold', pad=12)
    ax.legend(loc='lower right', frameon=True, framealpha=0.95)

    for idx, (ef, tf) in enumerate(zip(en_f1, te_f1)):
        ax.text(ef + 1.2, idx + height/2, f"{ef:.1f}%", va='center', fontsize=9, fontweight='bold', color='#1976d2')
        ax.text(tf + 1.2, idx - height/2, f"{tf:.1f}%", va='center', fontsize=9, fontweight='bold', color='#388e3c')

    plt.tight_layout()
    save_fig_all(fig, "fig5_crosslingual_script_gap")
    plt.close()

def main():
    ensure_dirs()
    df, models_info = load_all_data()
    print(f"Loaded {len(df)} total evaluation instances across {len(models_info)} models.")
    plot_fig1_master_position_curves(df, models_info)
    plot_fig2_token_fertility_unified()
    plot_fig3_all_model_mitigation(df, models_info)
    plot_fig4_master_heatmap(df, models_info)
    plot_fig5_cross_lingual_gap(df, models_info)
    print("\n[SUCCESS] Generated all 5 unified, all-model publication-grade figures!")

if __name__ == "__main__":
    main()
