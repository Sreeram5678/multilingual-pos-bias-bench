"""
Multi-Document Natural QA Scorer
Computes Token Overlap F1 and Substring Exact Match (EM).
"""

import re
import string

def normalize_text(s: str) -> str:
    # Lowercase, remove punctuation and extra whitespace
    s = s.lower().strip()
    s = re.sub(r'[' + re.escape(string.punctuation) + r'।॥\.,\?!]', ' ', s)
    return ' '.join(s.split())

def score_qa_prediction(prediction: str, gold_answer: str) -> dict:
    pred_norm = normalize_text(prediction)
    gold_norm = normalize_text(gold_answer)

    # Exact Match or Substring Containment
    exact_match = int(gold_norm in pred_norm or pred_norm in gold_norm) if pred_norm and gold_norm else 0

    pred_tokens = pred_norm.split()
    gold_tokens = gold_norm.split()

    common = set(pred_tokens).intersection(set(gold_tokens))
    num_same = sum(min(pred_tokens.count(w), gold_tokens.count(w)) for w in common)

    if len(pred_tokens) == 0 or len(gold_tokens) == 0:
        f1 = float(exact_match)
    elif num_same == 0:
        f1 = 0.0
    else:
        precision = num_same / len(pred_tokens)
        recall = num_same / len(gold_tokens)
        f1 = (2 * precision * recall) / (precision + recall)

    return {
        "exact_match": exact_match,
        "f1": round(f1, 4),
        "prediction": prediction.strip()
    }

if __name__ == "__main__":
    s1 = score_qa_prediction("The answer is Wound care for recovery", "Wound care")
    print("Test English Match:", s1)
    s2 = score_qa_prediction("సమాధానం: ఐదు మహాసముద్రాలు", "ఐదు")
    print("Test Telugu Match:", s2)
