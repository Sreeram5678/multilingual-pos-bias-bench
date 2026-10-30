"""
TyDi QA Multi-Document Long-Context Runner
Executes evaluation for Gemini (gemini-3.5-flash-lite) and Groq (qwen/qwen3.8-27b)
on real 10-document stacks across English and Telugu.
"""

import os
import json
import argparse
import pandas as pd
from tqdm import tqdm

from src.tydiqa_generator import generate_tydiqa_benchmark
from src.tydiqa_mitigation import apply_tydiqa_mitigation
from src.qa_evaluator import score_qa_prediction
from src.llm_clients import get_llm_client

def run_tydiqa_benchmark(
    provider: str = "gemini",
    model: str = "gemini-3.5-flash-lite",
    include_mitigation: bool = True,
    api_key: str = None
):
    print("=" * 70)
    print(" TYDI QA MULTI-DOCUMENT LONG-CONTEXT RETRIEVAL BENCHMARK")
    print("=" * 70)
    print(f"Provider: {provider.upper()}")
    print(f"Model: {model}")
    print(f"Mitigation Tested: {'DECR' if include_mitigation else 'None'}\n")

    # Step 1: Generate benchmark instances
    base_instances = generate_tydiqa_benchmark(depths=[0.0, 0.5, 1.0])
    print(f"Generated {len(base_instances)} base multi-document instances.")

    eval_items = []
    for inst in base_instances:
        b_inst = dict(inst)
        b_inst["mitigation"] = "baseline"
        eval_items.append(b_inst)

        if include_mitigation:
            m_inst = apply_tydiqa_mitigation(inst)
            eval_items.append(m_inst)

    print(f"Total evaluation conditions to test: {len(eval_items)}")

    # Step 2: Query Model
    client = get_llm_client(provider, api_key=api_key)
    records = []

    for item in tqdm(eval_items, desc=f"Querying {model}"):
        pred = client.generate(
            prompt=item["prompt"],
            model=model,
            temperature=0.0,
            max_tokens=40
        )
        scores = score_qa_prediction(pred, item["gold_answer"])

        record = {
            "model": model,
            "provider": provider,
            "sample_id": item["sample_id"],
            "language": item["language"],
            "depth": item["depth"],
            "original_depth": item.get("original_depth", item["depth"]),
            "mitigation": item["mitigation"],
            "question": item["question"],
            "gold_answer": item["gold_answer"],
            "prediction": pred,
            "exact_match": scores["exact_match"],
            "f1": scores["f1"]
        }
        records.append(record)

    # Save results
    out_dir = "results/tables"
    os.makedirs(out_dir, exist_ok=True)
    df = pd.DataFrame(records)
    csv_file = os.path.join(out_dir, f"tydiqa_results_{provider}.csv")
    df.to_csv(csv_file, index=False)
    print(f"\n[Saved] Detailed evaluation log to {csv_file}")

    # Summary table
    depth_col = "original_depth" if "original_depth" in df.columns else "depth"
    summary = df.groupby(["language", "mitigation", depth_col])["exact_match"].agg(["count", "mean"]).reset_index()
    summary["accuracy_%"] = (summary["mean"] * 100).round(1)
    print("\n" + "=" * 50)
    print(" RESULTS BREAKDOWN (Accuracy by Depth)")
    print("=" * 50)
    print(summary[["language", "mitigation", depth_col, "count", "accuracy_%"]].to_string(index=False))

    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", type=str, default="gemini", choices=["gemini", "groq", "mock"])
    parser.add_argument("--model", type=str, default="gemini-3.5-flash-lite")
    parser.add_argument("--api-key", type=str, default=None)
    args = parser.parse_args()

    run_tydiqa_benchmark(provider=args.provider, model=args.model, api_key=args.api_key)
