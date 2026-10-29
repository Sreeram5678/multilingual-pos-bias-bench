"""
Evaluation & Statistical Analysis Module
Computes exact retrieval accuracy, distractor intrusion rate, Lost-in-the-Middle (LIM) Gap,
Primacy/Recency bias, Wilson score 95% confidence intervals, and exact McNemar paired significance tests.
"""

import math
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple
from scipy import stats

def score_prediction(prediction: str, gold_key: str, distractor_keys: List[str]) -> Dict[str, Any]:
    """
    Evaluates a single model output with strict matching and distractor collision tracking.
    """
    pred_clean = prediction.strip().upper()
    gold_clean = gold_key.strip().upper()

    # Exact match or exact substring match
    is_correct = (gold_clean in pred_clean)

    # Check distractor collision (intrusion)
    hit_distractor = False
    for dk in distractor_keys:
        if dk.strip().upper() in pred_clean and dk.strip().upper() != gold_clean:
            hit_distractor = True
            break

    # If it emitted both gold and distractor, penalize as ambiguous failure
    if is_correct and hit_distractor:
        is_correct = False

    return {
        "is_correct": int(is_correct),
        "hit_distractor": int(hit_distractor),
        "raw_pred": prediction
    }

def wilson_ci(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """
    Calculates Wilson score 95% confidence interval for a proportion k/n.
    """
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denominator = 1 + (z**2) / n
    centre_adjusted_probability = p + (z**2) / (2 * n)
    adjusted_std_error = z * math.sqrt((p * (1 - p) + (z**2) / (4 * n)) / n)
    lower = max(0.0, (centre_adjusted_probability - adjusted_std_error) / denominator)
    upper = min(1.0, (centre_adjusted_probability + adjusted_std_error) / denominator)
    return round(lower, 4), round(upper, 4)

def compute_position_bias_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Computes summary position bias metrics across depths.
    df must contain columns: ['depth', 'is_correct', 'hit_distractor']
    """
    depth_stats = df.groupby('depth')['is_correct'].agg(
        total='count',
        correct='sum',
        accuracy='mean'
    ).reset_index()

    depth_stats['ci_lower'] = depth_stats.apply(lambda r: wilson_ci(int(r['correct']), int(r['total']))[0], axis=1)
    depth_stats['ci_upper'] = depth_stats.apply(lambda r: wilson_ci(int(r['correct']), int(r['total']))[1], axis=1)

    # Sort depths
    depth_stats = depth_stats.sort_values('depth')

    acc_by_depth = dict(zip(depth_stats['depth'], depth_stats['accuracy']))

    start_acc = acc_by_depth.get(0.0, 0.0)
    end_acc = acc_by_depth.get(1.0, 0.0)
    # Middle is 0.5 or closest to 0.5
    mid_depth = min(acc_by_depth.keys(), key=lambda d: abs(d - 0.5))
    mid_acc = acc_by_depth[mid_depth]

    # Metrics
    lim_gap = ((start_acc + end_acc) / 2.0) - mid_acc
    primacy_bias = start_acc - mid_acc
    recency_bias = end_acc - mid_acc

    max_acc = depth_stats['accuracy'].max()
    min_acc = depth_stats['accuracy'].min()
    mean_acc = depth_stats['accuracy'].mean()
    pbi = (max_acc - min_acc) / (mean_acc + 1e-6)

    # Paired McNemar Test (Edge vs Center)
    # Check instances that ran both at edge and at middle
    mcnemar_p = run_paired_mcnemar(df, mid_depth)

    return {
        "depth_table": depth_stats.to_dict(orient='records'),
        "lim_gap": round(lim_gap, 4),
        "primacy_bias": round(primacy_bias, 4),
        "recency_bias": round(recency_bias, 4),
        "pbi": round(pbi, 4),
        "mean_accuracy": round(mean_acc, 4),
        "mcnemar_p_value": mcnemar_p
    }

def run_paired_mcnemar(df: pd.DataFrame, mid_depth: float) -> float:
    """
    Exact McNemar test comparing paired performance between Edge positions (0.0 or 1.0) and Middle position (0.5).
    """
    if "sample_id" not in df.columns:
        return 1.0

    # Pivot to compare instances
    edge_df = df[df['depth'].isin([0.0, 1.0])].groupby('sample_id')['is_correct'].max()
    mid_df = df[df['depth'] == mid_depth].set_index('sample_id')['is_correct']

    common = edge_df.index.intersection(mid_df.index)
    if len(common) < 5:
        return 1.0

    # Contingency table:
    # b: Edge correct, Mid incorrect
    # c: Edge incorrect, Mid correct
    b = 0
    c = 0
    for sid in common:
        e = edge_df.loc[sid]
        m = mid_df.loc[sid]
        if e == 1 and m == 0:
            b += 1
        elif e == 0 and m == 1:
            c += 1

    # Exact binomial test for discordant pairs
    total_discordant = b + c
    if total_discordant == 0:
        return 1.0

    res = stats.binomtest(b, n=total_discordant, p=0.5)
    return round(float(res.pvalue), 5)
