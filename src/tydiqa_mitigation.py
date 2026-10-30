"""
DECR Mitigation for Multi-Document Stack
Re-ranks the 10 documents based on query-document lexical/semantic overlap:
- Rank 1 placed at Recency edge (Document 10)
- Rank 2 placed at Primacy edge (Document 1)
- Low-scoring distractors pushed to interior (Documents 4-7)
"""

import re
from typing import Dict, Any, List

def compute_doc_overlap(query: str, doc_text: str) -> float:
    q_words = set(re.findall(r'\w+', query.lower()))
    d_words = set(re.findall(r'\w+', doc_text.lower()))
    if not q_words:
        return 0.0
    return len(q_words.intersection(d_words)) / len(q_words)

def apply_tydiqa_mitigation(instance: Dict[str, Any]) -> Dict[str, Any]:
    docs = instance["documents"]
    query = instance["question"]
    lang = instance["language"]

    scored_docs = []
    for i, d in enumerate(docs):
        text = f"{d['title']} {d['context']}"
        score = compute_doc_overlap(query, text)
        scored_docs.append((i, d, score))

    # Sort descending by relevance
    sorted_docs = sorted(scored_docs, key=lambda x: x[2], reverse=True)

    n = len(docs)
    new_docs = [None] * n
    left = 0
    right = n - 1

    for rank, (orig_idx, doc, score) in enumerate(sorted_docs):
        if rank % 2 == 0:
            new_docs[right] = doc
            right -= 1
        else:
            new_docs[left] = doc
            left += 1

    doc_blocks = []
    for idx, doc in enumerate(new_docs):
        doc_blocks.append(f"[Document {idx+1}: {doc['title']}]\n{doc['context']}")

    context_str = "\n\n".join(doc_blocks)

    if lang == "telugu":
        header = f"[లక్ష్యం: {query}]\n\nక్రింద 10 ప్రామాణిక విజ్ఞాన సర్వస్వ పత్రాలు ఇవ్వబడ్డాయి. వాటిని శ్రద్ధగా చదవండి:\n\n"
        footer = f"\n\nపై పత్రాలలో ఇచ్చిన సమాచారం ఆధారంగా మాత్రమే, ఈ క్రింది ప్రశ్నకు సంక్షిప్తంగా (1-5 పదాలలో) సమాధానం ఇవ్వండి:\nప్రశ్న: {query}\nసమాధానం:"
    else:
        header = f"[Objective: {query}]\n\nBelow are 10 reference documents. Read them carefully:\n\n"
        footer = f"\n\nBased exclusively on the documents provided above, answer the following question concisely (1-5 words):\nQuestion: {query}\nAnswer:"

    mit_inst = dict(instance)
    mit_inst["mitigation"] = "decr"
    mit_inst["original_depth"] = instance["depth"]
    mit_inst["prompt"] = header + context_str + footer
    return mit_inst
