"""
Master Experiment Runner & Paper Benchmark Orchestrator
Executes the full pipeline:
Data Generation -> LLM Inference (Groq/Gemini/Mock) -> Scoring -> Statistics -> Paper Visualizations.
"""

import os
import sys
import argparse
import yaml
import json
import pandas as pd
from tqdm import tqdm

from src.data_generator import generate_paired_dataset
from src.mitigation import build_mitigated_prompt
from src.llm_clients import get_llm_client
from src.evaluator import score_prediction, compute_position_bias_metrics
from src.token_fertility import compute_fertility_metrics, plot_token_fertility, generate_latex_table
from src.visualizer import plot_position_bias_by_model, plot_multilingual_disparity, plot_mitigation_efficacy

def load_config(config_path: str = "configs/experiment_config.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def run_experiment(
    provider: str = "mock",
    models: list = None,
    languages: list = None,
    samples_per_cell: int = 10,
    num_distractors: int = 30,
    include_mitigation: bool = True,
    api_key: str = None
):
    print("=" * 70)
    print(" MULTILINGUAL LONG-CONTEXT POSITION BIAS BENCHMARK (M-PosBench)")
    print("=" * 70)

    config = load_config()
    languages = languages or [l["code"] for l in config.get("languages", [])]
    depth_bins = config.get("depth_bins", [0.0, 0.15, 0.30, 0.50, 0.70, 0.85, 1.0])

    if not models:
        if provider == "groq":
            models = ["qwen/qwen3.8-27b"]
        elif provider == "gemini":
            models = ["gemini-3.5-flash-lite"]
        else:
            models = ["mock-llama-8b", "mock-gemini-flash"]

    print(f"Provider: {provider.upper()}")
    print(f"Models: {models}")
    print(f"Languages: {languages}")
    print(f"Depths: {depth_bins}")
    print(f"Samples per Cell: {samples_per_cell}")
    print(f"Distractor Count: {num_distractors}")
    print(f"Mitigation Tested: {'DECR' if include_mitigation else 'None'}\n")

    # Step 1: Token Fertility Profile
    print("[1/5] Analyzing Token Fertility & Script Expansion...")
    _, fertility_summary = compute_fertility_metrics()
    plot_token_fertility(fertility_summary, "results/figures/fig2_token_fertility.png")
    generate_latex_table(fertility_summary, "results/tables/tab1_token_fertility.tex")

    # Step 2: Generate Paired Benchmark Data
    print(f"\n[2/5] Synthesizing M-NIAH Paired Dataset ({len(languages)} langs, {len(depth_bins)} depths)...")
    base_dataset = generate_paired_dataset(
        languages=languages,
        depth_bins=depth_bins,
        samples_per_cell=samples_per_cell,
        num_distractors=num_distractors,
        seed=config.get("seed", 42)
    )
    print(f"Generated {len(base_dataset)} baseline instances.")

    # Create mitigated instances
    eval_instances = []
    for inst in base_dataset:
        inst_copy = dict(inst)
        inst_copy["mitigation"] = "baseline"
        eval_instances.append(inst_copy)

        if include_mitigation:
            mit_inst = build_mitigated_prompt(inst, method="decr")
            eval_instances.append(mit_inst)

    print(f"Total evaluation instances (including mitigations): {len(eval_instances)}")

    # Step 3: Run Inference across Models
    print("\n[3/5] Querying Models & Caching Predictions...")
    client = get_llm_client(provider, api_key=api_key)

    records = []
    raw_output_path = "results/raw/predictions.jsonl"
    os.makedirs(os.path.dirname(raw_output_path), exist_ok=True)

    with open(raw_output_path, "w", encoding="utf-8") as raw_f:
        for model in models:
            print(f"--> Evaluating Model: {model}")
            for inst in tqdm(eval_instances, desc=f"Evaluating {model}"):
                prompt = inst["prompt"]
                gold_key = inst["gold_key"]
                distractor_keys = inst["distractor_keys"]

                pred = client.generate(
                    prompt=prompt,
                    model=model,
                    temperature=config.get("temperature", 0.0),
                    max_tokens=config.get("max_tokens", 40)
                )

                score = score_prediction(pred, gold_key, distractor_keys)

                record = {
                    "model": model,
                    "provider": provider,
                    "sample_id": inst.get("sample_id", "s0"),
                    "language": inst["language"],
                    "depth": inst["depth"],
                    "original_depth": inst.get("original_depth", inst["depth"]),
                    "mitigation": inst.get("mitigation", "baseline"),
                    "gold_key": gold_key,
                    "prediction": pred,
                    "is_correct": score["is_correct"],
                    "hit_distractor": score["hit_distractor"]
                }
                records.append(record)
                raw_f.write(json.dumps(record, ensure_ascii=False) + "\n")

    results_df = pd.DataFrame(records)
    csv_path = "results/tables/all_evaluation_records.csv"
    if os.path.exists(csv_path):
        try:
            existing_df = pd.read_csv(csv_path)
            combined_df = pd.concat([existing_df, results_df], ignore_index=True)
            # Deduplicate by key attributes
            dedup_cols = ['model', 'provider', 'sample_id', 'language', 'depth', 'mitigation']
            if all(c in combined_df.columns for c in dedup_cols):
                combined_df = combined_df.drop_duplicates(subset=dedup_cols, keep='last')
            results_df = combined_df
        except Exception as e:
            print(f"[Warning] Could not merge existing records: {e}")

    results_df.to_csv(csv_path, index=False)

    # Step 4: Compute Metrics & Significance Tests
    print("\n[4/5] Computing Lost-in-the-Middle Metrics & McNemar Statistics...")
    summary_rows = []
    for (model, lang, mit), group_df in results_df.groupby(['model', 'language', 'mitigation']):
        metrics = compute_position_bias_metrics(group_df)
        summary_rows.append({
            "Model": model,
            "Language": lang,
            "Mitigation": mit,
            "Mean_Accuracy": f"{metrics['mean_accuracy']*100:.1f}%",
            "LIM_Gap": f"{metrics['lim_gap']*100:.1f}%",
            "Primacy_Bias": f"{metrics['primacy_bias']*100:.1f}%",
            "Recency_Bias": f"{metrics['recency_bias']*100:.1f}%",
            "PBI_Index": f"{metrics['pbi']:.2f}",
            "McNemar_p": metrics["mcnemar_p_value"]
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv("results/tables/tab2_position_metrics_summary.csv", index=False)
    print("\nSummary Table:")
    print(summary_df.to_string(index=False))

    # Step 5: Generate Publication Figures
    print("\n[5/5] Generating Publication-Ready Figures (PNG & PDF)...")
    baseline_df = results_df[results_df['mitigation'] == 'baseline']
    plot_position_bias_by_model(baseline_df)
    plot_multilingual_disparity(baseline_df)
    if include_mitigation:
        plot_mitigation_efficacy(results_df)

    print("\n" + "=" * 70)
    print("BENCHMARK COMPLETED SUCCESSFULLY!")
    print("Figures saved in: results/figures/")
    print("Tables saved in:  results/tables/")
    print("=" * 70)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multilingual Long-Context Position Bias Benchmark")
    parser.add_argument("--provider", type=str, default="mock", choices=["groq", "gemini", "mock"],
                        help="LLM provider API to use (groq, gemini, or mock).")
    parser.add_argument("--models", nargs="+", default=None, help="List of model identifiers to test.")
    parser.add_argument("--languages", nargs="+", default=["en", "hi", "te"], help="Languages to evaluate.")
    parser.add_argument("--samples", type=int, default=10, help="Number of instances per condition.")
    parser.add_argument("--distractors", type=int, default=30, help="Distractor items in context.")
    parser.add_argument("--skip-mitigation", action="store_true", help="Skip mitigation benchmark.")
    parser.add_argument("--api-key", type=str, default=None, help="Explicit API key (or set via env var).")

    args = parser.parse_args()

    run_experiment(
        provider=args.provider,
        models=args.models,
        languages=args.languages,
        samples_per_cell=args.samples,
        num_distractors=args.distractors,
        include_mitigation=not args.skip_mitigation,
        api_key=args.api_key
    )
