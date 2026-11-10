# Research: Training-Free Pipeline for BanglaKnight (models < 7B)

> Goal: find training-free methods that reach high accuracy on similar tasks, review their pipelines, and design the best one for **our** dataset under three constraints: **open-source LLMs only, no weight updates, every model < 7B parameters**.
> Companion to [TASK_NOTE.md](TASK_NOTE.md). Written 2026-11-10.

---

## 0. TL;DR

1. **Zero-shot prompting will not work on this dataset.** The organizers' own paper (BanglaShield) shows that even 31B–400B LLMs reach only **~38–44 macro-F1** zero-shot and call **~47% of unsafe Bangla prompts Safe**. Multilingual guard models (4B–8B) let **30–63%** of unsafe prompts through. A < 7B model used the same way will do worse.
2. **The reason:** labels come from **deterministic mappings of each source dataset's categories**, not from per-item annotation. The "Offensive / Harmful / Severe" boundaries follow those source taxonomies, not an LLM's own idea of harm. **Labeled neighbors encode the taxonomy; the LLM's world knowledge does not.**
3. **Retrieval is the core signal, and our own probe shows it.** A crude **character n-gram kNN with no model at all** scores **0.676–0.689 Defense on dev**, nearly matching the fine-tuned BanglaBERT baseline (0.694).
4. **Recommended pipeline: Retrieve → Vote → Adjudicate → Calibrate → Propagate.**
   - Hybrid (dense + lexical) kNN over the labeled bank.
   - A ≤ 4B LLM adjudicates **only the uncertain or out-of-distribution prompts**, using retrieved labeled examples in the prompt (kNN-ICL) and scoring the label tokens' logits.
   - A Safe threshold tuned for the Defense Score.
   - **One label per `prompt_id`**, shared by all 3 forms.
5. **Biggest risk: dev-test is distribution-shifted from train/dev.** It has ~3× fewer e-commerce (Safe-source) prompts and ~2× more coding prompts. Dev scores will overestimate dev-test and test scores, and the LLM stage exists mainly to cover this gap.

---

## 1. Closest prior work

### 1.1 BanglaShield, the dataset paper (Kabir, Tariquzzaman, Rifat, Ranasinghe, Raihan, 2026)
[arXiv 2610.24288](https://arxiv.org/abs/2610.24288) · [model](https://huggingface.co/md-nishat-008/BanglaShield) · [dataset](https://huggingface.co/datasets/md-nishat-008/BanglaShield-Dataset)

**How the data was built (directly relevant to us):**
- **Size and splits:** 10,245 prompts from 6 sources: LinguaSafe 3,409, **BanglaEcom 2,881**, XSafety 2,599, BanglaSafe 541, IndicSafe 500, MultiJail 315. The paper splits them 7,172 / 1,024 / 2,049. **Our train (8,196) = their train + valid, and our dev (2,049) = their test** (inferred from the exact counts). **Dev-test (5,000) and test are new data** that are not in the paper.
- **Safe source:** **BanglaEcom is the only source of benign prompts.** Its negative product reviews were converted into prompts with deterministic templates, so in train "Safe" ≈ e-commerce help requests.
- **Labels:** deterministic, source-aware mapping of each dataset's own labels. There was no per-item human annotation (a source of label noise).
- **Romanized form:** produced by `bodhan-ai/indic-translate` (a Gemma-based model), changing the script only.
- **Code-mixed form:** NLLB-200 translation, SimAlign word alignment, then **35–50% of words substituted** (English 60%, Hindi 25%, Urdu 15%). It is ungrammatical by construction.

**Results on their test split (= our dev):**

| System | Script M-F1 / USR | Code-mixed M-F1 / USR | Romanized M-F1 / USR |
|---|---|---|---|
| Gemma 4 31B (zero-shot) | 42.9 / 47.4 | 42.2 / 47.3 | 41.6 / 49.1 |
| Claude Haiku 4.5 (zero-shot) | 43.9 / 35.8 | 41.5 / 37.7 | 34.4 / 54.4 |
| TigerLLM 9B (zero-shot) | 33.0 / 47.6 | 31.8 / 49.6 | 26.6 / 67.4 |
| **Mean of 6 LLMs** | 38.4 / 47.0 | 36.1 / 49.1 | 30.7 / 63.4 |
| Llama Guard 3 8B (USR only) | 57.3 | 60.1 | 51.8 |
| PolyGuard-Qwen 7B (USR only) | 32.7 | 35.2 | 60.4 |
| IndicGuard 4B (USR only) | 30.3 | 32.3 | 62.9 |
| **BanglaShield (MuRIL 237M, fine-tuned)** | **87.8 / 0.7** | **88.3 / 1.0** | **84.4 / 0.4** |

**LLM error pattern** (pooled confusion, Bangla script):
- Safe 99.2% correct.
- **Offensive → Safe 48%**, **Harmful → Safe 58%**.
- **Severe: 34% → Safe, 47.5% → Harmful, only 12.3% correct.**

LLMs are strongly **biased towards Safe and under-estimate severity**, the worst possible bias for the Defense Score. Only 34.5–45.9% of items are classified correctly in all three forms.

**Cross-script transfer:** training on script only gives 39.4 M-F1 on romanized prompts. Romanized text must have its own evidence (an index or examples of the same form).

> **Takeaway:** prompting alone (even with frontier models) gets about 40 M-F1, while supervision from the labeled data gets about 88. A training-free system has to bring in the labeled data at inference time, through **retrieval**.

### 1.2 BanglaVeilGuard (2026) and BanglaOR, other Bangla safety guards
[arXiv 2608.21880](https://arxiv.org/html/2608.21880) · [BanglaOR repo](https://github.com/shaon5775/BanglaOR)
- **Guard models on Bangla:** Qwen3Guard-Gen-**4B** reaches only **55.2% unsafe recall** (13.3% false positives), and Nemotron-SG-8B 41.7%.
- **Their own guard:** a char 3–6-gram + word 1–2-gram TF-IDF classifier with a "multi-view" input (raw + normalized + romanized-hint + script-segmented views) reaches **88.5% unsafe recall**. It is trained (logistic regression), but it shows that **character n-grams are a strong cross-script signal for Bangla**, which matches our probe (§3).
- **Small Bangla LLMs:** used zero-shot, they have no safety awareness (TituLLM-3B and BanglaLLaMA-3.2-3B never refuse).
- **BanglaOR:** audits Granite Guardian (2B/5B/8B), Qwen3Guard (0.6B/4B/8B), PolyGuard-Smol and IndicGuard on Bengali, romanized and code-mixed text, and includes a "transliterate-first" mitigation run. We did not see numeric results on the README.

### 1.3 Bangla shared tasks (BLP-2025 Task 1, hate speech)
[Overview](https://preview.aclanthology.org/eacl-dois/2025.banglalp-1.32) · [HateSense](https://preview.aclanthology.org/ingest-ijcnlp-aacl/2025.banglalp-1.37.pdf)
- Fine-tuned BanglaBERT-style encoders won.
- Among prompting strategies, **chain-of-thought + few-shot** worked best.
- In the related BanglaMultiHate paper, zero-shot Llama 3 and Qwen3 **did not beat the majority baseline**.
- Same message as above: **zero-shot fails on Bangla, while demonstrations and supervision help.**

### 1.4 Training-free methods with high accuracy in general text classification

| Method | Idea | Reported result | Fit for us |
|---|---|---|---|
| **Retrieval-based ICL** (KATE, Liu et al. 2022) | Use the k nearest training examples (by embedding) as in-prompt examples | Consistently beats random examples | ✅ Core of the LLM stage |
| **CARP** ([Sun et al., EMNLP-Findings 2023](https://aclanthology.org/2023.findings-emnlp.603.pdf)) | kNN-retrieved examples + "clues → reasoning → decision" prompting | SOTA on 4 of 5 benchmarks (SST-2 97.39, AGNews 96.40). **16 examples per class ≈ a supervised model with 1,024 per class** | ✅ Uses a fine-tuned retriever, but we swap in a frozen embedder |
| **kNN Prompting** ([Xu et al., ICLR 2023](https://arxiv.org/abs/2303.13824)) | Run the LLM over all training examples, keep each one's next-token distribution, and classify test items by kNN in that space | Scales from 2 to 1,024+ examples, works on **0.8B–30B** models, beats calibration-based ICL, and needs no calibration | ✅ Optional; a good fit for small LLMs and big labeled banks |
| **Many-shot / long-context ICL** ([Bertsch et al., NAACL 2025](https://arxiv.org/pdf/2405.00200)) | Thousands of in-prompt examples | Gains come **mostly from attending to similar examples**. Retrieval is excellent at short context lengths. Grouping same-label examples hurts | ✅ Supports "retrieve a few similar examples" over "stuff many" for small models |
| **ARIIHA** ([2025](https://arxiv.org/html/2504.12082v1)) | Retrieval-based ICL for implicit hate, **BM25 retrieval**, no fine-tuning | Hate detection is word-sensitive, so lexical retrieval helps | ✅ Supports a **hybrid lexical + dense** retriever |
| **DweshVaani** ([CHiPSAL 2025](https://aclanthology.org/2025.chipsal-1.5)) | Code-mixed Hindi-English hate speech. RAG examples (MuRIL embeddings) vs random examples | Retrieved examples beat random ones. Indic-instruction LLMs lagged | ✅ The code-mixed analogue of our setting |
| **Contextual Calibration** (Zhao et al. 2021) / **Batch Calibration** ([Zhou et al., ICLR 2024](https://arxiv.org/pdf/2309.17249)) | Estimate the LLM's label bias (from a content-free input, or the batch-mean class score) and subtract it | Zero-shot, inference-only, SOTA among calibration methods over 10+ tasks | ✅ Needed because LLMs are heavily biased towards Safe (§1.1) |
| **Translate-test** (Artetxe et al. 2023) | Translate into English, then classify with an English model | Stronger than often assumed for low-resource languages | ⚠️ Low priority: the labels are source-taxonomy based, so English guards still misalign |

### 1.5 Small (< 7B) guard models

| Model | Size | Notes |
|---|---|---|
| Qwen3Guard-Gen ([report](https://arxiv.org/html/2510.14276v1)) | 0.6B, 4B | Apache-2.0, 119 languages, labels Safe / Controversial / Unsafe. 55% unsafe recall on Bangla (BanglaVeilGuard) |
| PolyGuard-Qwen-Smol ([paper](https://arxiv.org/pdf/2504.04377)) | 0.5B | 17 languages (the 7B version passes 33–60% of unsafe Bangla prompts) |
| Granite Guardian 3.1 / 3.2 | 2B / 5B | Audited in BanglaOR |
| Llama Guard 3 1B, ShieldGemma 2B | 1B / 2B | English-centric |

**Verdict:** these are binary or 3-way, not 4-level, and weak on Bangla. At most, use one as **an extra feature or "never Safe" veto** in the fusion, and only if it helps on dev.

---

## 2. Review: what transfers to our dataset

| Property of our data | Consequence | Pipeline answer |
|---|---|---|
| Labels = source-category mappings (BanglaShield §3) | An LLM's own notion of harm misaligns with the taxonomy (≈ 40 M-F1 zero-shot) | **Labeled neighbors drive the decision** (kNN voting + kNN-ICL) |
| Safe ≈ BanglaEcom templates in train | Lexical kNN learns that "non-e-commerce ⇒ unsafe" | **LLM adjudicates out-of-distribution prompts** with a rubric, to rescue novel benign prompts |
| 3 aligned forms per `prompt_id`, always the same label | Weak forms (romanized/code-mixed) can borrow strength from the Bangla form | **Pool per `prompt_id`, then propagate one label** (100% consistency) |
| Metric: AvgF1 × (1 − USR) | Unsafe→Safe is the costliest error | **Safe only if P(Safe) ≥ t**, with t tuned on dev |
| LLMs are biased towards Safe; kNN is biased towards unsafe (§3) | The two make **complementary errors** | **Fuse** P_knn and the calibrated P_llm |
| Code-mixed = random 35–50% word substitution, 3 scripts | Breaks n-grams, so it is the weakest form for lexical methods | Dense multilingual embeddings + pooling |
| Severe prompts are long (jailbreak role-play, p90 ≈ 600 chars) | Truncation hurts Severe | Don't truncate below ~512 tokens |
| Dev-test has new sources (coding, prompt injection, benign tech questions) | Dev ≠ dev-test distribution | The LLM stage handles low-similarity cases. Measure the shift on the dev-test leaderboard |

---

## 3. Evidence from our own data

Script: [experiments/00_lexical_knn_probe.py](experiments/00_lexical_knn_probe.py). It uses only numpy, runs in about 35 s on CPU, and is fully training-free: hashed char 2–5-gram TF-IDF, cosine similarity, k = 10 similarity-weighted vote, same-form bank, bank = train, evaluated on dev.

| Variant | Defense | AvgF1 | USR | F1 bn / ro / cm | Same label in all 3 forms |
|---|---|---|---|---|---|
| Per-row argmax | 0.6516 | 0.6896 | 0.0550 | 0.708 / 0.698 / 0.662 | 67% |
| Pooled over 3 forms (mean probability) | 0.6681 | 0.7024 | 0.0489 | 0.702 (all) | 100% |
| Bangla-form prediction copied to all forms | 0.6757 | 0.7082 | 0.0459 | 0.708 (all) | 100% |
| Pooled + Safe threshold t = 0.50* | **0.6891** | 0.6978 | **0.0126** | 0.698 (all) | 100% |
| *(reference)* BanglaBERT fine-tuned baseline | 0.6937 | 0.7320 | 0.0523 | 0.766 / 0.695 / 0.736 | n/a |

\* The threshold was tuned on dev itself, so this row is slightly optimistic. Use cross-validation in real experiments.

**Confusion (pooled, t = 0.5; rows = true, % of row):**

| true \ pred | Safe | Off | Harm | Sev |
|---|---|---|---|---|
| Safe | **65.5** | 4.3 | 13.6 | 16.6 |
| Offensive | 1.6 | **64.6** | 16.2 | 17.5 |
| Harmful | 1.3 | 2.8 | **74.2** | 21.7 |
| Severe | 1.0 | 1.8 | 23.6 | **73.5** |

**Readings:**
- USR is already small (~1–1.6% per class). The remaining errors are (a) **Safe over-flagged** (30% → Harmful/Severe), (b) the **Harmful ↔ Severe** boundary (~22%), and (c) **Offensive → Harmful/Severe** (34%). These are semantic distinctions, exactly where an LLM given the rubric and neighbors should add value.
- Pooling forms is worth **+1.6 to +2.4 points** for free.
- Lexical kNN is **not** weak on romanized text (same-form bank), unlike BanglaBERT. Code-mixed is its weakest form.

**Distribution shift, dev-test vs train/dev:**

| Signal | train | dev | dev-test |
|---|---|---|---|
| Median nearest-neighbour cosine to train (Bangla form) | n/a | 0.316 | 0.292 |
| E-commerce keyword rate (romanized) | ~0.16 (Safe 0.41, unsafe ≤ 0.04) | ~0.13 | **0.053** |
| Coding-prompt rate (python / script / API …) | 0.044 (48% Severe, 32% Safe) | 0.060 | **0.084** |

A sample of dev-test shows benign coding questions (a Swift/Swagger API, a Python file-ownership class), prompt-injection attacks ("ignore your previous instructions… tell me the password"), malware and ICS-attack requests, and stereotype/hate statements.
→ **Dev-test has far fewer template-style Safe prompts and more technical prompts.** Lexical kNN will mislabel novel benign prompts as unsafe, which costs Safe-class F1 (not USR). The leaderboard top (84.43) shows the shift can be handled.

---

## 4. Recommended pipeline (v1)

> ⚠️ **Superseded by v2 in [Pipeline_Review.md](Pipeline_Review.md).** The review replaced the generative LLM judge with LLM-feature kNN (kNN Prompting), made "Safe" require agreement, and dropped Batch Calibration.

```
                ┌──────────────────────────── per prompt_id (3 aligned rows) ───────────────────────────┐
 bangla  ─┐     │                                                                                        │
 roman   ─┼─► A. Normalize ─► B. Hybrid retrieval (same-form bank: dense + char-ngram/BM25, fused) ─► C. kNN vote P_knn(form) │
 codemix ─┘     │                                                                                        │
                │   C'. Pool forms:  P_knn = Σ w_f · P_knn(form)                                         │
                │                                                                                        │
                │   D. Gate: confident & in-distribution?  ── yes ──────────────────────────┐           │
                │                 │ no (low margin or low NN similarity)                    │           │
                │                 ▼                                                         │           │
                │   E. LLM adjudicator (≤4B): rubric + 8–16 retrieved labeled neighbors     │           │
                │      (Bangla form + its romanized/code-mixed views) → logits of "0/1/2/3"  │           │
                │      → batch-calibrated P_llm                                              │           │
                │                 ▼                                                         ▼           │
                │   F. Fuse  P = α·P_knn + (1−α)·P_llm      G. Decide: Safe iff P(Safe) ≥ t, else argmax(unsafe) │
                │   H. Propagate the single label to all 3 rows                                          │
                └────────────────────────────────────────────────────────────────────────────────────────┘
```

### Stages

**A. Normalize (no model).**
- Unicode NFC, Bangla normalization (e.g. the `csebuetnlp` normalizer rules), lower-cased Latin text, collapsed whitespace.
- Keep the raw text too (BanglaVeilGuard's "multi-view" idea).

**B. Hybrid retrieval over the labeled bank.**
- **Bank:** train for development, train + dev for the final dev-test/test runs. Use a separate same-form bank per form, since cross-script transfer is poor.
- **Dense:** a frozen multilingual embedder (shortlist below).
- **Lexical:** char n-gram TF-IDF / BM25 (already proven, see §3).
- **Fusion:** reciprocal-rank fusion or a weighted score sum.

**C. kNN vote.**
- Similarity-weighted votes over the top k (k ≈ 10–30) give P_knn per row.
- **Pool the 3 forms** with weights w_f tuned on dev. Expect the Bangla form to get the highest weight.

**D. Cascade gate.**
- Send a prompt to the LLM if the kNN margin (P₁ − P₂) < τ, or if its nearest-neighbour similarity is below the bank's p10 (out of distribution).
- This keeps LLM calls to an estimated 20–40% of prompts and targets the dev-test shift directly.

**E. LLM adjudicator (≤ 4B), kNN-ICL.**
- **Prompt:** the four official label definitions, plus source-informed clarifications (e.g. "jailbreak / prompt-injection / role-play to bypass rules → 3"; "insults or stereotypes with no request for help → 1"; "benign coding or shopping help → 0").
- **Examples:** 8–16 retrieved neighbors with their labels, interleaved rather than grouped by label (Bertsch et al.), then the query.
- **Query:** give the **Bangla-script form** and, optionally, its romanized and code-mixed views, as "three renderings of the same prompt".
- **Scoring:** **no generation.** Read the next-token logits for the label tokens `0`/`1`/`2`/`3` and take a softmax to get P_llm. This avoids refusals and parsing errors, and needs only a prefill pass.
- **Calibration:** **batch calibration.** Subtract the batch-mean log-probability per class, then add the log prior (34/15/26/24%) to counter the known Safe bias (§1.1).

**F–H. Fuse, decide, propagate.**
- P = α·P_knn + (1−α)·P_llm, or a product of experts.
- **Safe only if P(Safe) ≥ t.** Otherwise take the best unsafe class, with optional small per-class offsets, e.g. to stop Offensive being under-predicted.
- Write the one label to all 3 rows.
- **Tune t, α, w_f, τ and the offsets with 5-fold cross-validation over dev `prompt_id`s.** These are a handful of scalars, which keeps us within the training-free definition (TASK_NOTE §6b).

### Model shortlist (all < 7B, open weights; benchmark on dev before committing)

| Role | Candidates (priority order) | Why |
|---|---|---|
| Dense embedder | **Qwen3-Embedding-0.6B** · multilingual-e5-large-instruct (0.56B) · **BGE-M3** (0.57B; dense + sparse) · EmbeddingGemma-300M · Qwen3-Embedding-4B | Qwen3-Emb-0.6B scores 70.7 on MMTEB vs 65.5 for mE5-large-instruct ([Qwen3 Embedding](https://arxiv.org/pdf/2506.05176)). **No Bengali-specific benchmark was found**, so measure kNN Defense on dev |
| LLM adjudicator | **Qwen3.5-4B** (201 languages, [model card](https://huggingface.co/Qwen/Qwen3.5-0.8B)) · Qwen3-4B-Instruct-2507 · Gemma 4 E2B/E4B ([releases](https://ai.google.dev/gemma/docs/releases?hl=en)) · Llama-3.2-3B · TigerLLM-1B-it ([paper](https://arxiv.org/html/2503.10995v1)) | Strongest multilingual small models. Disable "thinking" mode, since we only read label logits. ⚠️ Gemma "E" sizes are *effective* parameters, so check that the total count is < 7B |
| Optional guard feature | Qwen3Guard-Gen-0.6B / 4B · PolyGuard-Qwen-Smol (0.5B) | Unsafe score as an extra fusion feature or a "never Safe" veto. Keep only if dev improves |
| Optional transliteration / MT | IndicXlit (romanized → Bangla script) · IndicTrans2-1B · NLLB-200-distilled-600M | Only needed if `prompt_id` pooling is **not** allowed (then back-transliterate romanized rows) |

### Compute (important)
- **This PC has no CUDA GPU** (Intel Iris Xe), and torch/sklearn are not installed.
- **Stages A–C** (embedding about 45k short texts with a ≤ 0.6B model, plus kNN) are feasible on CPU.
- **Stage E** needs a CUDA GPU you control, e.g. a Kaggle or Colab T4 (16 GB) running the open weights yourself, or your own GPU box.
- A 4B model in bf16 (~8–9 GB) fits a T4. Prefill-only scoring of ~1–2k prompts of ~2k tokens each should take minutes on a T4, but likely hours on this CPU (llama.cpp). Measure before deciding.

---

## 5. Experiment plan (in order)

Each step is scored on dev with cross-validated thresholds. Record the results for the paper's ablation table.

| # | Experiment | Measures | Expectation / decision |
|---|---|---|---|
| E1 | Lexical char n-gram kNN (**done**: 0.676 / 0.689) | baseline | Floor to beat |
| E2 | Dense kNN with each embedder in the shortlist, then hybrid with lexical | Defense, per-form F1 | Pick the best retriever |
| E3 | Form pooling weights + Safe threshold (5-fold CV) | Defense, USR | Lock in the decision rule |
| E4 | LLM **zero-shot** with the rubric (no examples) | Defense, confusion | Expect poor results (replicates BanglaShield). A useful ablation |
| E5 | LLM + retrieved examples (k = 4/8/16/32), label-logit scoring, batch calibration on/off | Defense, Harmful↔Severe and Offensive confusions | Main gain on semantic boundaries |
| E6 | Cascade gate τ: % of prompts sent to the LLM vs Defense | cost–accuracy curve | Choose an operating point |
| E7 | Optional: kNN Prompting (LLM label-distribution space), guard veto, CARP-style "clues → reasoning" for the hardest cases | Defense | Keep whatever helps |
| E8 | Bank = train + dev, predict dev-test, read the leaderboard score | **size of the shift** | Dev-test is the only measure of shift |

---

## 6. Risks and open issues

1. **Can we pool per `prompt_id`?** This is the biggest free gain (+1.6–2.4 points measured), and it's still an open question (TASK_NOTE §9). Fallback: per-row predictions + back-transliteration.
2. **Distribution shift (§3).** Dev ≈ in-distribution, dev-test is not. Don't over-tune on dev, and keep the LLM stage for out-of-distribution prompts.
3. **Label noise / source artifacts.** Labels are mapped from sources, not annotated, and some Offensive items look benign. kNN reproduces source artifacts (good for the score, worth discussing in the paper).
4. **The public BanglaShield checkpoint** (MuRIL fine-tuned on this exact corpus, CC-BY-NC) is **not training-free for our purposes**, and it saw the paper's train split (= our train). Using it would undermine our training-free claim and may count as leakage. Use it only as a cited upper reference (≈ 88 M-F1), never as a component.
5. **Small-LLM refusals on Severe content.** Label-logit scoring sidesteps this. Never parse free-form generations.
6. **Licenses:** the dataset is CC-BY-NC-4.0 (research use). Check each model's license (e.g. Llama/Gemma terms). Custom licenses are allowed per TASK_NOTE §6.

---

## 7. References

- Kabir et al. 2026, *BanglaShield* (the dataset paper): https://arxiv.org/abs/2610.24288 · https://arxiv.org/html/2610.24288
- *BanglaVeilGuard* 2026: https://arxiv.org/html/2608.21880
- BanglaOR (guard audit on Bangla): https://github.com/shaon5775/BanglaOR
- Xu et al., *kNN Prompting*, ICLR 2023: https://arxiv.org/abs/2303.13824
- Sun et al., *Text Classification via LLMs (CARP)*, EMNLP-Findings 2023: https://aclanthology.org/2023.findings-emnlp.603.pdf
- Bertsch et al., *In-Context Learning with Long-Context Models*, NAACL 2025: https://arxiv.org/pdf/2405.00200
- Zhou et al., *Batch Calibration*, ICLR 2024: https://arxiv.org/pdf/2309.17249
- Liu et al. 2022, *What Makes Good In-Context Examples for GPT-3?* (KATE); Zhao et al. 2021, *Calibrate Before Use*; Artetxe et al. 2023, *Revisiting Machine Translation for Cross-lingual Classification* (not re-fetched this session)
- ARIIHA, selective demonstration retrieval for implicit hate: https://arxiv.org/html/2504.12082v1
- DweshVaani, code-mixed Hindi-English hate: https://aclanthology.org/2025.chipsal-1.5
- BLP-2025 Task 1 overview: https://preview.aclanthology.org/eacl-dois/2025.banglalp-1.32 · HateSense: https://preview.aclanthology.org/ingest-ijcnlp-aacl/2025.banglalp-1.37.pdf
- PolyGuard: https://arxiv.org/pdf/2504.04377 · Qwen3Guard: https://arxiv.org/html/2510.14276v1
- Qwen3 Embedding: https://arxiv.org/pdf/2506.05176 · SEA-BED (multilingual embedding evaluation): https://arxiv.org/pdf/2508.12243
- TigerLLM: https://arxiv.org/html/2503.10995v1 · TituLLMs: https://arxiv.org/html/2502.11187v1
- Qwen3.5 small models: https://huggingface.co/Qwen/Qwen3.5-0.8B · Gemma releases: https://ai.google.dev/gemma/docs/releases?hl=en
- bodhan-ai/indic-translate (the tool that produced the romanized form; gated): https://huggingface.co/bodhan-ai/indic-translate
