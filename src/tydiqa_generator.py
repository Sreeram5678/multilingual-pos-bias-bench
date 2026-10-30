"""
TyDi QA Multi-Document Long-Context Benchmark Generator
Constructs paired 10-document stacks with the gold target document placed at:
- Start (0%)
- Middle (50%)
- End (100%)
"""

import os
import json
import random
from typing import List, Dict, Any

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "tydiqa", "tydiqa_eval_data.json")

def load_tydiqa_raw() -> dict:
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def build_tydiqa_instance(
    lang: str,
    target_item: dict,
    distractor_articles: List[dict],
    depth: float
) -> Dict[str, Any]:
    """
    Constructs a 10-document stack where the gold target document is inserted at `depth`.
    depth: 0.0 (Doc 1), 0.5 (Doc 5), 1.0 (Doc 10).
    """
    # 9 distractors + 1 target = 10 documents
    distractors = list(distractor_articles[:9])
    
    # Calculate insertion index in [0, 9]
    insert_idx = int(round(depth * len(distractors)))
    insert_idx = min(max(0, insert_idx), len(distractors))

    gold_doc = {
        "title": target_item["title"],
        "context": target_item["context"],
        "is_gold": True
    }

    stack = []
    for i, d in enumerate(distractors):
        stack.append({
            "title": d["title"],
            "context": d["context"],
            "is_gold": False
        })

    stack.insert(insert_idx, gold_doc)

    # Format document stack
    doc_blocks = []
    for idx, doc in enumerate(stack):
        doc_blocks.append(f"[Document {idx+1}: {doc['title']}]\n{doc['context']}")

    context_str = "\n\n".join(doc_blocks)

    if lang == "telugu":
        header = "క్రింద 10 ప్రామాణిక విజ్ఞాన సర్వస్వ పత్రాలు ఇవ్వబడ్డాయి. వాటిని శ్రద్ధగా చదవండి:\n\n"
        footer = f"\n\nపై పత్రాలలో ఇచ్చిన సమాచారం ఆధారంగా మాత్రమే, ఈ క్రింది ప్రశ్నకు సంక్షిప్తంగా (1-5 పదాలలో) సమాధానం ఇవ్వండి:\nప్రశ్న: {target_item['question']}\nసమాధానం:"
    else:
        header = "Below are 10 reference documents. Read them carefully:\n\n"
        footer = f"\n\nBased exclusively on the documents provided above, answer the following question concisely (1-5 words):\nQuestion: {target_item['question']}\nAnswer:"

    full_prompt = header + context_str + footer

    return {
        "sample_id": f"{lang}_{target_item['id']}",
        "language": lang,
        "depth": depth,
        "insert_idx": insert_idx,
        "total_docs": len(stack),
        "question": target_item["question"],
        "gold_answer": target_item["answer"],
        "gold_title": target_item["title"],
        "documents": stack,
        "prompt": full_prompt
    }

def generate_tydiqa_benchmark(depths: List[float] = [0.0, 0.5, 1.0]) -> List[Dict[str, Any]]:
    raw_data = load_tydiqa_raw()
    dataset = []

    for lang in ["english", "telugu"]:
        targets = raw_data[lang]["targets"]
        all_distractors = raw_data[lang]["distractors"]

        for i, target in enumerate(targets):
            # Select 9 distinct distractors deterministically for each question
            rng = random.Random(42 + i)
            distractor_subset = rng.sample(all_distractors, 9)

            for depth in depths:
                inst = build_tydiqa_instance(
                    lang=lang,
                    target_item=target,
                    distractor_articles=distractor_subset,
                    depth=depth
                )
                dataset.append(inst)

    return dataset

if __name__ == "__main__":
    b = generate_tydiqa_benchmark()
    print(f"Generated {len(b)} TyDi QA benchmark instances.")
    s = b[0]
    print(f"Sample Language: {s['language']}, Depth: {s['depth']}")
    print(f"Question: {s['question']}")
    print(f"Gold Answer: {s['gold_answer']}")
    print(f"Prompt length: {len(s['prompt'])} chars")
