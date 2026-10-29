"""
Mitigation Module: Dynamic Edge-Centric Context Reordering (DECR)
Implements training-free algorithmic mitigation to eliminate the 'Lost in the Middle' effect.

Mechanism:
1. Computes relevance/salience scores between the prompt query and context chunks.
2. Applies 'Sandwich Attention Scheduling' (Edge-Centric Placement):
   - Places the highest-relevance evidence at the Recency boundary (adjacent to query/generation).
   - Places the second-highest evidence at the Primacy boundary (attention sink position).
   - Pushes low-salience distractors into the middle context positions (where model attention is lowest).
3. Adds bidirectional query anchoring (Query Sandwiching).
"""

import re
from typing import List, Dict, Any, Tuple
from src.data_generator import NEEDLE_TEMPLATES

def compute_lexical_overlap(query: str, chunk: str) -> float:
    """
    Computes normalized token overlap between query and chunk.
    Language-agnostic (works on English, Hindi, and Telugu).
    """
    # Extract alphanumeric and script-specific word tokens
    q_words = set(re.findall(r'\w+', query.lower()))
    c_words = set(re.findall(r'\w+', chunk.lower()))

    if not q_words:
        return 0.0

    overlap = len(q_words.intersection(c_words))
    # Normalize by query length
    return overlap / len(q_words)

def apply_edge_centric_reordering(
    haystack_chunks: List[str],
    query: str
) -> Tuple[List[str], List[int]]:
    """
    Reorders chunks so that high-scoring chunks are mapped to the edges (Primacy & Recency),
    while lowest-scoring chunks are pushed to the center.

    Returns:
        reordered_chunks: The newly ordered list of chunk strings.
        original_indices_mapping: Where each original chunk moved.
    """
    scores = [(i, compute_lexical_overlap(query, chunk)) for i, chunk in enumerate(haystack_chunks)]
    # Sort chunks in descending order of relevance score
    sorted_by_relevance = sorted(scores, key=lambda x: x[1], reverse=True)

    n = len(haystack_chunks)
    new_order = [None] * n

    left = 0
    right = n - 1

    # Place Rank 1 at right (recency), Rank 2 at left (primacy), alternating inward
    for rank_idx, (orig_idx, score) in enumerate(sorted_by_relevance):
        if rank_idx % 2 == 0:
            new_order[right] = orig_idx
            right -= 1
        else:
            new_order[left] = orig_idx
            left += 1

    reordered_chunks = [haystack_chunks[orig_idx] for orig_idx in new_order]
    return reordered_chunks, new_order

def build_mitigated_prompt(
    base_instance: Dict[str, Any],
    method: str = "decr"
) -> Dict[str, Any]:
    """
    Takes an unmitigated instance and creates the mitigated prompt.
    """
    if method == "baseline":
        return base_instance

    lang = base_instance["language"]
    haystack = base_instance["haystack_chunks"]
    query = NEEDLE_TEMPLATES[lang]["question"]

    # Apply Dynamic Edge-Centric Context Reordering
    reordered_chunks, new_order = apply_edge_centric_reordering(haystack, query)

    # Reconstruct the prompt
    context_body = "\n".join(reordered_chunks)

    # Query Sandwiching: Prepend query summary at top, and question at bottom
    prompt = (
        f"[Search Objective: {query}]\n\n" +
        NEEDLE_TEMPLATES[lang]["header"] +
        context_body +
        NEEDLE_TEMPLATES[lang]["footer"] +
        query
    )

    mitigated_inst = dict(base_instance)
    mitigated_inst["prompt"] = prompt
    mitigated_inst["mitigation"] = method
    mitigated_inst["original_depth"] = base_instance["depth"]
    return mitigated_inst

if __name__ == "__main__":
    from src.data_generator import build_instance, generate_haystack_items
    import random
    rng = random.Random(42)
    distractors, keys = generate_haystack_items("en", 10, rng)
    # Place needle right in the middle (depth 0.5)
    inst = build_instance("en", 0.5, 10, "TEST-9999", distractors)

    print("Baseline middle prompt preview (needle at 50%):")
    print(inst["haystack_chunks"][5])

    mitigated = build_mitigated_prompt(inst, "decr")
    print("\nMitigated prompt edges (Top chunk vs Bottom chunk):")
    print("Top (Index 0):", mitigated["prompt"].split("\n\n")[1].split("\n")[0])
    print("Bottom (Index -1):", mitigated["prompt"].split("\n\n")[1].split("\n")[-1])
