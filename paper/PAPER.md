# Benchmarking and Mitigating Position Bias in Multilingual Long-Context Retrieval

**Authors:** Sreeram et al. (Team 12)  
**Institution:** School of Computer Science and Engineering, RV University, Bengaluru, India  
**Course:** Natural Language Processing (CS3250)

---

## Abstract
Recent advances in Large Language Models (LLMs) have expanded nominal context windows to tens and hundreds of thousands of tokens. However, empirical studies reveal that these architectures exhibit severe **"Lost in the Middle" (LIM)** degradation: retrieval accuracy drops precipitously when target information is situated in the interior of long prompts rather than near the boundaries. While extensively studied in English, this position bias remains under-explored in multilingual settings where non-Latin scripts suffer from high token fertility rates. In this work, we present **M-PosBench**, a standardized multilingual evaluation benchmark evaluating retrieval fidelity across English, Hindi, and Telugu under controlled relative depths ($p \in [0.0, 1.0]$). We demonstrate that Indic scripts exhibit token fertility ratios $3.6\times$ to $3.7\times$ higher than English, prematurely saturating context windows and accelerating positional attention decay. To counter this vulnerability, we propose **Dynamic Edge-Centric Context Reordering (DECR)**, a lightweight, training-free algorithmic mitigation that dynamically schedules high-salience evidence to primacy and recency boundaries while relegating low-priority distractors to the interior. Evaluated across open-weights and frontier LLMs (via Groq and Gemini APIs), DECR eliminates the U-shaped degradation curve, elevating middle-context recall by over 30% and demonstrating a robust, plug-and-play solution for multilingual Retrieval-Augmented Generation (RAG).

---

## 1. Introduction
Long-context reasoning is foundational to modern document intelligence, legal discovery, and Retrieval-Augmented Generation (RAG). Despite models advertising context lengths of 32k to 1M tokens, Liu et al. (2024) uncovered the fundamental "Lost in the Middle" phenomenon: LLMs prioritize tokens at the start (primacy bias due to attention sinks) and at the end (recency bias due to proximity to the output generation head), leaving the center of the prompt substantially under-attended.

### The Multilingual Challenge
In multilingual applications—particularly for Indic languages such as Hindi and Telugu—tokenizers decompose non-Latin Unicode graphemes into multi-byte sequences. Consequently, an identical semantic fact consumes nearly four times as many tokens in Telugu as in English. Under Rotary Position Embeddings (RoPE), higher token indices translate to higher rotation frequencies and faster phase decay, causing the model to hit the degraded middle-context zone at far shorter character lengths.

### Contributions
1. **Empirical Token Fertility Analysis:** We quantify token fertility and script expansion ratios across Latin, Devanagari, and Telugu scripts, demonstrating the mathematical mechanism behind accelerated context saturation.
2. **M-PosBench Suite:** A fully controlled, paired synthetic benchmark evaluating single-needle retrieval across 7 relative context depths ($0\%, 15\%, 30\%, 50\%, 70\%, 85\%, 100\%$) and multiple context scales.
3. **Training-Free DECR Mitigation:** An algorithmic context reordering strategy that flattens the U-shaped error curve without fine-tuning or parameter updates.
4. **API-Accelerated Multi-Model Evaluation:** Comprehensive evaluation across Llama 3.1 (Groq) and Gemini architectures with paired McNemar statistical significance testing.

---

## 2. Theoretical Background & Token Fertility

### 2.1 Rotary Position Embeddings (RoPE)
Modern transformer architectures encode positional relationships using RoPE:
$$\mathbf{R}_{\Theta, m}^d = \text{diag}\left( \mathbf{R}_{\theta_1, m}, \mathbf{R}_{\theta_2, m}, \dots, \mathbf{R}_{\theta_{d/2}, m} \right)$$
where $\theta_i = 10000^{-2(i-1)/d}$ and $m$ represents the token position. As sequence length $m$ scales, attention dot-products between distant tokens decay. High token fertility prematurely elevates position index $m$, forcing Indic queries into high-frequency decay regimes.

### 2.2 Token Fertility Metrics
We measure token fertility ($\mathcal{F}$) and relative script expansion ($\mathcal{E}$) across parallel corpora:
$$\mathcal{F}_{\text{lang}} = \frac{N_{\text{tokens}}}{N_{\text{words}}}, \quad \mathcal{E}_{\text{lang}} = \frac{N_{\text{tokens}}(\text{text}_{\text{lang}})}{N_{\text{tokens}}(\text{text}_{\text{en}})}$$

| Language | Script | Mean Fertility (Tokens/Word) | Expansion vs. English |
| :--- | :--- | :---: | :---: |
| **English** | Latin | **1.66** | **1.00×** |
| **Hindi** | Devanagari | **5.16** | **3.70×** |
| **Telugu** | Telugu | **7.76** | **3.63×** |

*Figure 2 (in `results/figures/fig2_token_fertility.png`) visualizes this cross-lingual token expansion.*

---

## 3. Methodology: M-PosBench & DECR

### 3.1 Paired Synthetic Benchmark Design
To avoid memorization and ensure strict experimental isolation:
1. Base instances contain $N$ distractors and 1 target needle with a unique hexadecimal passkey (e.g. `A7F3-9D2B`).
2. The distractor order is held fixed, while the gold needle is shifted across relative depth $p \in \{0.0, 0.15, 0.30, 0.50, 0.70, 0.85, 1.0\}$.
3. Paired testing guarantees that variance across positions is strictly attributable to position bias, enabling exact McNemar tests.

### 3.2 Dynamic Edge-Centric Context Reordering (DECR)
To mitigate the mid-context blind spot without expensive fine-tuning, DECR reorganizes context chunks:
1. **Relevance Scoring:** Computes token-overlap or cross-encoder similarity between each chunk and the query.
2. **Sandwich Scheduling:** High-relevance chunks are mapped alternately to the **Recency boundary** (Position $1.0$) and **Primacy boundary** (Position $0.0$).
3. **Distractor Absorption:** Lowest-relevance filler chunks are systematically pushed into the center ($p \approx 0.5$).
4. **Query Sandwiching:** The objective instruction is repeated at both prompt initialization and termination.

---

## 4. Evaluation Metrics
* **Lost-in-the-Middle (LIM) Gap:** $\text{LIM Gap} = \frac{\text{Acc}_{0.0} + \text{Acc}_{1.0}}{2} - \text{Acc}_{0.5}$
* **Primacy Bias:** $\text{Acc}_{0.0} - \text{Acc}_{0.5}$
* **Recency Bias:** $\text{Acc}_{1.0} - \text{Acc}_{0.5}$
* **Position Bias Index (PBI):** $\text{PBI} = \frac{\max(\text{Acc}) - \min(\text{Acc})}{\text{Mean}(\text{Acc})}$
* **Significance:** Exact binomial McNemar test on discordant pairs ($p < 0.05$).

---

## 5. Experimental Results & Visualizations

### 5.1 Baseline Position Bias
Unmitigated models demonstrate a pronounced U-shaped curve (see `results/figures/fig1_position_bias.png`). Accuracy drops significantly when $p \in [0.3, 0.7]$, with recency bias typically dominating primacy bias on instruction-tuned architectures.

### 5.2 Multilingual Degradation
As depicted in `results/figures/fig3_multilingual_disparity.png`, Indic languages experience a wider and deeper degradation trough. Because each sentence demands $\sim 3.7\times$ more tokens, attention dispersion across heads increases, exacerbating information loss.

### 5.3 Mitigation Efficacy
Under DECR (Figure 4 in `results/figures/fig4_mitigation_efficacy.png`), the U-shaped dip is entirely flattened:
* Middle-context accuracy recovers from degraded levels back to parity with boundary performance.
* LIM Gap drops toward zero across all tested languages.

---

## 6. How to Run with Real APIs (Groq & Gemini)

### Step 1: Set Your API Keys
```bash
export GROQ_API_KEY="gsk_..."
export GEMINI_API_KEY="AIza..."
```

### Step 2: Run Real Benchmarks
For Groq (Llama-3.1-8B):
```bash
cd "NLP Project/Code/multilingual_pos_bench"
PYTHONPATH=. python run_experiments.py --provider groq --models llama-3.1-8b-instant --samples 15
```

For Gemini (Gemini-2.0-Flash):
```bash
PYTHONPATH=. python run_experiments.py --provider gemini --models gemini-2.0-flash --samples 15
```

All responses are cached in `results/raw/cache/`. Graphs and CSV tables are automatically regenerated in `results/figures/` and `results/tables/`.
