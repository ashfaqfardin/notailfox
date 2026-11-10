#!/usr/bin/env python3
"""
BanglaKnight (BLP-2027 Task 1): training-free Bangla prompt-safety pipeline v2, in one file.

Stages (see Pipeline_Review.md and pipeline_v2.png)
  S0  group the 3 written forms of each prompt by prompt_id (the Bangla-script row is primary)
  S1  normalize text (NFC, optional csebuetnlp normalizer, lower-case Latin, collapse spaces)
  S2  frozen "views" -> kNN label votes over a labeled bank
        lex   : hashed char 2-5-gram TF-IDF                          (numpy only, CPU)
        dense : instruction-aware embedder, e.g. Qwen3-Embedding-0.6B (sentence-transformers)
        llm   : last-token hidden state of a <=4B LLM under a safety prompt
                (kNN-Prompting-style features, Bangla row only by default)
        guard : optional Qwen3Guard-Gen-0.6B "Unsafe" flag that vetoes a Safe prediction
  S3  fuse views (weights a_v) and pool forms (Bangla weight w_b, the rest split evenly)
  S4  optional EM prior-shift re-weighting (ablation; --prior-shift em)
  S5  asymmetric decision: Safe iff P(Safe) >= t (+ delta if out-of-distribution) and no guard veto,
      otherwise the argmax over {Offensive, Harmful, Severe}
  S6  copy the prompt label to all of its rows -> predictions.csv + submission.zip

Training-free: no model weights are updated. Besides the kNN bank, the only fitted things are a
handful of scalars (a_v, w_b, t, guard on/off), chosen by grid search over dev prompts with
stratified k-fold cross-validation.

Usage
  python banglaknight_pipeline.py --mode dev                         # evaluate on dev (bank = train)
  python banglaknight_pipeline.py --mode predict --phase devtest     # tune on dev, predict devtest
  python banglaknight_pipeline.py --mode predict --phase test --views lex,dense,llm

Requirements: Python 3.9+, numpy. The neural views also need
  pip install torch "transformers>=4.51" "sentence-transformers>=3"
and a CUDA GPU (e.g. a Kaggle or Colab T4) for the LLM view. Gated models (e.g. Gemma) need
`huggingface-cli login`. Every loaded model is checked against the < 7B parameter rule.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import itertools
import json
import os
import re
import sys
import time
import unicodedata
import warnings
import zipfile
import zlib

import numpy as np

FORMS = ("bangla", "romanized", "codemixed")
SUFFIX_FORM = {"ba": "bangla", "ro": "romanized", "co": "codemixed"}
NAMES = ("Safe", "Offensive", "Harmful", "Severe")
T_GRID = np.round(np.arange(0.25, 0.951, 0.025), 3)        # Safe thresholds searched
WB_GRID = (1 / 3, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)             # Bangla-form weights searched
LLM_TEMPLATE = (
    "Below is a request that a user sent to an AI assistant. It may be written in Bangla, "
    "romanized Bangla, or code-mixed text.\n"
    'Request: "{text}"\n'
    'In one word, the safety risk of this request (safe, offensive, harmful or severe) is: "'
)
_T0 = time.time()
_MODELS: dict = {}


def log(msg: str) -> None:
    print(f"[{time.time() - _T0:7.1f}s] {msg}", flush=True)


# ----------------------------------------------------------------------------------------------
# S0 · data
# ----------------------------------------------------------------------------------------------
def read_rows(data_dir: str, name: str) -> list[dict]:
    for ext in ("jsonl", "csv"):
        path = os.path.join(data_dir, f"{name}.{ext}")
        if os.path.exists(path):
            with open(path, encoding="utf-8", newline="") as fh:
                rows = [json.loads(l) for l in fh if l.strip()] if ext == "jsonl" else list(csv.DictReader(fh))
            break
    else:
        raise FileNotFoundError(f"{name}.jsonl or {name}.csv not found in {data_dir}")
    for r in rows:
        r["id"] = str(r["id"])
        stem, _, suffix = r["id"].rpartition("_")
        r["prompt_id"] = str(r.get("prompt_id") or stem or r["id"])
        r["form"] = r.get("form") or SUFFIX_FORM.get(suffix, "bangla")
        lab = r.get("label")
        r["label"] = int(lab) if lab not in (None, "") else None
        r["text"] = r.get("text") or ""
    return rows


class Split:
    """All rows of one file, grouped by prompt_id."""

    def __init__(self, name: str, rows: list[dict]):
        self.name, self.rows = name, rows
        self.pids = list(dict.fromkeys(r["prompt_id"] for r in rows))
        index = {p: i for i, p in enumerate(self.pids)}
        self.n = len(self.pids)
        self.row_prompt = np.array([index[r["prompt_id"]] for r in rows], dtype=int)
        self.y = np.full(self.n, -1, dtype=int)
        for r in rows:
            if r["label"] is not None:
                self.y[index[r["prompt_id"]]] = r["label"]
        self.forms = {}
        for f in FORMS:
            sel = [r for r in rows if r["form"] == f]
            self.forms[f] = {
                "prompt": np.array([index[r["prompt_id"]] for r in sel], dtype=int),
                "raw": [r["text"] for r in sel],
                "label": np.array([-1 if r["label"] is None else r["label"] for r in sel], dtype=int),
                "ids": [r["id"] for r in sel],
            }
        unknown = {r["form"] for r in rows} - set(FORMS)
        if unknown:
            log(f"warning: {name} has unknown forms {unknown}; those rows get their prompt's label")

    def texts(self, form: str, kind: str) -> list[str]:
        d = self.forms[form]
        key = f"text_{kind}"
        if key not in d:
            d[key] = [clean_lexical(t) if kind == "lex" else clean_light(t) for t in d["raw"]]
        return d[key]

    def fingerprint(self, form: str) -> str:
        return hashlib.sha1("\n".join(self.forms[form]["ids"]).encode()).hexdigest()[:12]


def label_prior(y: np.ndarray) -> np.ndarray:
    y = y[y >= 0]
    return np.bincount(y, minlength=4).astype(np.float32) / max(len(y), 1)


# ----------------------------------------------------------------------------------------------
# S1 · normalization
# ----------------------------------------------------------------------------------------------
_ZERO_WIDTH = dict.fromkeys(map(ord, "​‌‍⁠﻿"), None)
try:  # optional: pip install git+https://github.com/csebuetnlp/normalizer
    from normalizer import normalize as _bn_normalize  # type: ignore
except Exception:  # noqa: BLE001
    _bn_normalize = None


def clean_light(t: str) -> str:
    """For neural views: stay close to the raw text."""
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", t)).strip()


def clean_lexical(t: str) -> str:
    """For the lexical view: aggressive normalization."""
    t = clean_light(t)
    if _bn_normalize is not None:
        try:
            t = _bn_normalize(t)
        except Exception:  # noqa: BLE001
            pass
    return re.sub(r"\s+", " ", t.translate(_ZERO_WIDTH).lower()).strip()


# ----------------------------------------------------------------------------------------------
# S2 · views -> kNN votes
# ----------------------------------------------------------------------------------------------
def knn_votes(Q: np.ndarray, B: np.ndarray, yb: np.ndarray, k: int, tau: float, chunk: int = 1024):
    """Similarity-weighted kNN label votes. Q, B are L2-normalized rows. Returns (P, top-1 sim)."""
    assert (yb >= 0).all(), "bank rows must be labeled"
    k = min(k, len(B))
    P = np.zeros((len(Q), 4), np.float32)
    top = np.zeros(len(Q), np.float32)
    for s in range(0, len(Q), chunk):
        S = Q[s:s + chunk] @ B.T
        idx = np.argpartition(-S, k - 1, axis=1)[:, :k]
        sim = np.take_along_axis(S, idx, 1)
        w = np.exp((sim - sim.max(1, keepdims=True)) / tau)
        lab = yb[idx]
        for c in range(4):
            P[s:s + chunk, c] = (w * (lab == c)).sum(1)
        top[s:s + chunk] = sim.max(1)
    return P / P.sum(1, keepdims=True), top


def hashed_ngrams(texts: list[str], dim: int, nmin: int = 2, nmax: int = 5) -> np.ndarray:
    X = np.zeros((len(texts), dim), np.float32)
    mask = dim - 1
    for i, t in enumerate(texts):
        s = f" {t} "
        h = [zlib.crc32(s[j:j + n].encode("utf-8")) & mask
             for n in range(nmin, nmax + 1) for j in range(len(s) - n + 1)]
        if h:
            u, c = np.unique(np.asarray(h), return_counts=True)
            X[i, u] = c
    np.log1p(X, out=X)  # sublinear tf
    return X


def lexical_view(bank: list[Split], query: Split, form: str, args):
    texts_b = [t for s in bank for t in s.texts(form, "lex")]
    yb = np.concatenate([s.forms[form]["label"] for s in bank])
    Xb = hashed_ngrams(texts_b, args.lex_dim)
    idf = (np.log((1 + len(Xb)) / (1 + (Xb > 0).sum(0))) + 1).astype(np.float32)
    Xb *= idf
    Xb /= np.linalg.norm(Xb, axis=1, keepdims=True) + 1e-9
    tq = query.texts(form, "lex")
    P = np.zeros((len(tq), 4), np.float32)
    top = np.zeros(len(tq), np.float32)
    for s in range(0, len(tq), 2048):
        Xq = hashed_ngrams(tq[s:s + 2048], args.lex_dim) * idf
        Xq /= np.linalg.norm(Xq, axis=1, keepdims=True) + 1e-9
        P[s:s + 2048], top[s:s + 2048] = knn_votes(Xq, Xb, yb, args.k, args.tau_lex)
    return P, top


def torch_env():
    import torch
    if torch.cuda.is_available():
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        return torch, "cuda", dtype
    return torch, "cpu", torch.float32


def check_size(model, name: str, limit: float) -> None:
    n = sum(p.numel() for p in model.parameters())
    if n >= limit:
        raise SystemExit(f"{name}: {n / 1e9:.2f}B parameters breaks the < {limit / 1e9:.0f}B rule")
    log(f"loaded {name} ({n / 1e9:.2f}B params)")


def cached(args, kind: str, split: Split, form: str, extra: str, fn) -> np.ndarray:
    key = hashlib.sha1(f"{kind}|{split.name}|{form}|{split.fingerprint(form)}|{extra}".encode()).hexdigest()[:16]
    path = os.path.join(args.cache_dir, f"{kind}_{split.name}_{form}_{key}.npy")
    if os.path.exists(path):
        return np.load(path)
    arr = fn()
    os.makedirs(args.cache_dir, exist_ok=True)
    np.save(path, arr)
    return arr


def dense_prefix(args) -> str:
    if args.no_instruction:
        return ""
    name = args.dense_model.lower()
    if "bge-m3" in name:
        return ""
    if "embeddinggemma" in name:
        return "task: classification | query: "
    return f"Instruct: {args.dense_instruction}\nQuery: "  # Qwen3-Embedding / multilingual-e5-instruct


def dense_encode(args, texts: list[str]) -> np.ndarray:
    if "dense" not in _MODELS:
        _, dev, dtype = torch_env()
        from sentence_transformers import SentenceTransformer
        m = SentenceTransformer(args.dense_model, device=dev,
                                model_kwargs={"torch_dtype": dtype} if dev == "cuda" else {})
        m.max_seq_length = args.max_len
        check_size(m, args.dense_model, args.max_params)
        _MODELS["dense"] = m
    E = _MODELS["dense"].encode(texts, prompt=dense_prefix(args) or None, batch_size=args.dense_batch,
                                normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=True)
    return E.astype(np.float32)


def llm_encode(args, texts: list[str]) -> np.ndarray:
    torch, dev, dtype = torch_env()
    if "llm" not in _MODELS:
        from transformers import AutoModel, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.llm_model)
        tok.padding_side = "left"  # so position -1 is the real last token
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        model = AutoModel.from_pretrained(args.llm_model, torch_dtype=dtype).to(dev).eval()  # backbone, no LM head
        check_size(model, args.llm_model, args.max_params)
        _MODELS["llm"] = (tok, model)
    tok, model = _MODELS["llm"]
    prompts = []
    for t in texts:  # truncate the text, never the template
        ids = tok(t, add_special_tokens=False)["input_ids"][: args.llm_max_text_tokens]
        prompts.append(LLM_TEMPLATE.format(text=tok.decode(ids)))
    order = np.argsort([-len(p) for p in prompts])  # longest first: fail fast on OOM
    out = None
    for b_i, s in enumerate(range(0, len(prompts), args.llm_batch)):
        b = order[s:s + args.llm_batch]
        enc = tok([prompts[i] for i in b], return_tensors="pt", padding=True).to(dev)
        with torch.inference_mode():
            hs = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
                       output_hidden_states=True).hidden_states[args.llm_layer]
        h = hs[:, -1, :].float().cpu().numpy()
        if out is None:
            out = np.zeros((len(prompts), h.shape[1]), np.float32)
        out[b] = h
        if b_i % 50 == 0:
            log(f"    llm features {min(s + args.llm_batch, len(prompts))}/{len(prompts)}")
    return out


def neural_view(args, kind: str, bank: list[Split], query: Split, form: str):
    if kind == "dense":
        encode, tau = dense_encode, args.tau_dense
        extra = f"{args.dense_model}|{dense_prefix(args)}|{args.max_len}"
    else:
        encode, tau = llm_encode, args.tau_llm
        extra = f"{args.llm_model}|{args.llm_layer}|{args.llm_max_text_tokens}|{LLM_TEMPLATE}"

    def emb(split: Split):
        texts = split.texts(form, "neural")
        return cached(args, kind, split, form, extra, lambda: encode(args, texts)) if texts else None

    Eb = np.concatenate([e for e in (emb(s) for s in bank) if e is not None])
    yb = np.concatenate([s.forms[form]["label"] for s in bank])
    Eq = emb(query)
    if kind == "llm":  # hidden states are anisotropic: centre on the bank mean, then re-normalize
        mu = Eb.mean(0, keepdims=True)
        Eb, Eq = Eb - mu, Eq - mu
    Eb = Eb / (np.linalg.norm(Eb, axis=1, keepdims=True) + 1e-9)
    Eq = Eq / (np.linalg.norm(Eq, axis=1, keepdims=True) + 1e-9)
    return knn_votes(Eq, Eb, yb, args.k, tau)


def fill_missing(P: dict, prior: np.ndarray) -> dict:
    """Forms a view did not cover (or rows missing in a file) borrow the prompt's other forms."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        fallback = np.nanmean(np.stack([P[f] for f in FORMS]), axis=0)
    fallback = np.where(np.isnan(fallback), prior, fallback)
    return {f: np.where(np.isnan(P[f]), fallback, P[f]).astype(np.float32) for f in FORMS}


def run_view(view: str, bank: list[Split], query: Split, args, prior: np.ndarray):
    forms = tuple(args.llm_forms.split(",")) if view == "llm" else FORMS
    P = {f: np.full((query.n, 4), np.nan, np.float32) for f in FORMS}
    top = {f: np.full(query.n, np.nan, np.float32) for f in FORMS}
    for f in forms:
        pi = query.forms[f]["prompt"]
        if len(pi) == 0:
            continue
        Pf, tf = lexical_view(bank, query, f, args) if view == "lex" else neural_view(args, view, bank, query, f)
        P[f][pi], top[f][pi] = Pf, tf
        log(f"  S2 view={view:5s} form={f:9s} bank={sum(len(s.forms[f]['raw']) for s in bank):6d} "
            f"queries={len(pi)}")
    return fill_missing(P, prior), top


def compute_views(views: list[str], bank: list[Split], query: Split, args, prior: np.ndarray):
    log(f"S2 bank={'+'.join(s.name for s in bank)} -> queries={query.name}")
    Ps, tops = {}, {}
    for v in views:
        Ps[v], tops[v] = run_view(v, bank, query, args, prior)
    return Ps, tops


def guard_encode(args, texts: list[str]) -> np.ndarray:
    torch, dev, dtype = torch_env()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.guard_model)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.guard_model, torch_dtype=dtype).to(dev).eval()
    check_size(model, args.guard_model, args.max_params)
    flags = np.zeros((len(texts), 1), np.float32)
    order = np.argsort([-len(t) for t in texts])
    for s in range(0, len(texts), args.guard_batch):
        b = order[s:s + args.guard_batch]
        chats = []
        for i in b:
            ids = tok(texts[i], add_special_tokens=False)["input_ids"][: args.llm_max_text_tokens]
            chats.append(tok.apply_chat_template([{"role": "user", "content": tok.decode(ids)}],
                                                 tokenize=False, add_generation_prompt=True))
        enc = tok(chats, return_tensors="pt", padding=True, add_special_tokens=False).to(dev)
        with torch.inference_mode():
            out = model.generate(**enc, max_new_tokens=24, do_sample=False, pad_token_id=tok.pad_token_id)
        for i, d in zip(b, tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)):
            m = re.search(r"Safety:\s*(Safe|Unsafe|Controversial)", d)
            lab = m.group(1) if m else "Safe"
            flags[i, 0] = float(lab == "Unsafe" or (args.guard_strict and lab == "Controversial"))
    del model
    if dev == "cuda":
        torch.cuda.empty_cache()
    return flags


def guard_veto(args, split: Split) -> np.ndarray:
    texts = split.texts("bangla", "neural")
    veto = np.zeros(split.n, bool)
    if texts:
        flags = cached(args, "guard", split, "bangla", f"{args.guard_model}|{args.guard_strict}",
                       lambda: guard_encode(args, texts))
        veto[split.forms["bangla"]["prompt"]] = flags.ravel() > 0.5
    log(f"  guard flags {split.name}: {veto.mean():.1%} of prompts flagged Unsafe")
    return veto


def prompt_top(top: dict) -> np.ndarray:
    """Nearest-neighbour similarity per prompt (Bangla row, else mean of the forms)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mean = np.nanmean(np.stack([top[f] for f in FORMS]), axis=0)
    t = np.where(np.isnan(top["bangla"]), mean, top["bangla"])
    return np.nan_to_num(t, nan=np.inf)  # no evidence -> never treated as out-of-distribution


# ----------------------------------------------------------------------------------------------
# S3–S5 · fuse, pool, prior shift, decide
# ----------------------------------------------------------------------------------------------
def pool(Pv: dict, wb: float) -> np.ndarray:
    wo = (1 - wb) / 2
    return wb * Pv["bangla"] + wo * (Pv["romanized"] + Pv["codemixed"])


def fuse(Pviews: list[dict], a, wb: float) -> np.ndarray:
    return sum(w * pool(Pv, wb) for w, Pv in zip(a, Pviews))


def em_prior_shift(P: np.ndarray, prior_src: np.ndarray, iters: int = 200, tol: float = 1e-6):
    """Saerens et al. (2002): re-estimate the class prior of the query set, re-weight posteriors."""
    prior_src = np.maximum(prior_src, 1e-6)
    prior, Q = prior_src.copy(), P
    for _ in range(iters):
        Q = P * (prior / prior_src)
        Q = Q / Q.sum(1, keepdims=True)
        new = Q.mean(0)
        if np.abs(new - prior).max() < tol:
            break
        prior = new
    return Q.astype(np.float32), prior


def decide(P: np.ndarray, t: float, ood=None, delta: float = 0.0, veto=None) -> np.ndarray:
    thr = t + (delta * ood if ood is not None else 0.0)
    safe = P[:, 0] >= thr
    if veto is not None:
        safe &= ~veto
    return np.where(safe, 0, P[:, 1:].argmax(1) + 1)


# ----------------------------------------------------------------------------------------------
# metric + tuning
# ----------------------------------------------------------------------------------------------
def macro_f1(y: np.ndarray, p: np.ndarray) -> float:
    cm = np.bincount(y * 4 + p, minlength=16).reshape(4, 4).astype(float)
    tp = np.diag(cm)
    den = 2 * tp + (cm.sum(0) - tp) + (cm.sum(1) - tp)
    return float(np.mean(np.divide(2 * tp, den, out=np.zeros(4), where=den > 0)))


def defense_prompts(y: np.ndarray, p: np.ndarray):
    """Defense Score when all forms of a prompt share one prediction (equal to the row-level score)."""
    usr = float((p[y > 0] == 0).mean()) if (y > 0).any() else 0.0
    f1 = macro_f1(y, p)
    return f1 * (1 - usr), f1, usr


def defense_rows(y: np.ndarray, form: np.ndarray, p: np.ndarray):
    """The official formula: macro-F1 per written form, averaged, times (1 - USR)."""
    f1s = {f: macro_f1(y[form == f], p[form == f]) for f in FORMS if (form == f).any()}
    avg = float(np.mean(list(f1s.values())))
    usr = float((p[y > 0] == 0).mean())
    return avg * (1 - usr), avg, usr, f1s


def simplex(m: int, step: float) -> list[np.ndarray]:
    if m == 1:
        return [np.ones(1)]
    n = round(1 / step)
    return [np.array(c + (n - sum(c),)) / n for c in itertools.product(range(n + 1), repeat=m - 1) if sum(c) <= n]


def grid_search(Pviews, y, idx, ood, delta, veto, args):
    best = (-1.0, None)
    y_i, base = y[idx], delta * ood[idx]
    guard_opts = (False, True) if veto is not None else (False,)
    for wb in WB_GRID:
        pooled = [pool(Pv, wb)[idx] for Pv in Pviews]
        for a in simplex(len(pooled), args.a_step):
            P = sum(w * pp for w, pp in zip(a, pooled))
            ps, unsafe = P[:, 0], P[:, 1:].argmax(1) + 1
            for g in guard_opts:
                for t in T_GRID:
                    safe = ps >= t + base
                    if g:
                        safe &= ~veto[idx]
                    s = defense_prompts(y_i, np.where(safe, 0, unsafe))[0]
                    if s > best[0] + 1e-12:
                        best = (s, {"wb": float(wb), "a": [float(x) for x in a], "t": float(t), "guard": g})
    return best


def stratified_folds(y: np.ndarray, k: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    folds: list[list[int]] = [[] for _ in range(k)]
    for c in np.unique(y):
        for j, part in enumerate(np.array_split(rng.permutation(np.where(y == c)[0]), k)):
            folds[j].extend(part.tolist())
    return [np.array(sorted(f)) for f in folds]


def cv_tune(Pviews, y, ood, delta, veto, args):
    """k-fold CV estimate of the tuned system, plus the final scalars tuned on all of dev."""
    folds = stratified_folds(y, args.folds, args.seed)
    scores = []
    for j, te in enumerate(folds):
        tr = np.concatenate([f for i, f in enumerate(folds) if i != j])
        _, prm = grid_search(Pviews, y, tr, ood, delta, veto, args)
        P = fuse(Pviews, prm["a"], prm["wb"])[te]
        p = decide(P, prm["t"], ood[te], delta, veto[te] if prm["guard"] else None)
        scores.append(defense_prompts(y[te], p)[0])
    full, params = grid_search(Pviews, y, np.arange(len(y)), ood, delta, veto, args)
    return float(np.mean(scores)), float(np.std(scores)), float(full), params


# ----------------------------------------------------------------------------------------------
# reports and submission
# ----------------------------------------------------------------------------------------------
def dev_report(views, Pviews, dev: Split, ood, veto, prior, args):
    res: dict = {"views": views, "systems": {}, "variants": {}}
    print("\n== Dev results: bank = train, queries = dev, "
          f"{args.folds}-fold CV over {dev.n} prompts ==")
    print(f"{'system':44s} {'CV Defense':>18s} {'tuned on all dev*':>18s}")
    if len(views) > 1:
        for v, Pv in zip(views, Pviews):
            cv, sd, full, prm = cv_tune([Pv], dev.y, ood, args.ood_delta, None, args)
            res["systems"][f"{v} only"] = {"cv": cv, "sd": sd, "dev_tuned": full, "params": prm}
            print(f"{'view ' + v + ' alone':44s} {cv:10.4f} ± {sd:.4f} {full:18.4f}")
    cv, sd, full, params = cv_tune(Pviews, dev.y, ood, args.ood_delta, veto, args)
    res["systems"]["fused"] = {"cv": cv, "sd": sd, "dev_tuned": full, "params": params}
    print(f"{'FUSED ' + '+'.join(views):44s} {cv:10.4f} ± {sd:.4f} {full:18.4f}")
    print("* tuned and scored on the same prompts, so optimistic; the CV column is the honest estimate.")

    a, gv = params["a"], (veto if params["guard"] else None)
    Pf = {f: sum(w * Pv[f] for w, Pv in zip(a, Pviews)) for f in FORMS}
    rp, yr = dev.row_prompt, dev.y[dev.row_prompt]
    fr = np.array([r["form"] for r in dev.rows])
    Prow = np.stack([(Pf[f] if f in Pf else Pf["bangla"])[i] for f, i in zip(fr, rp)])
    P = fuse(Pviews, a, params["wb"])
    final = decide(P, params["t"], ood, args.ood_delta, gv)
    Pem, est = em_prior_shift(P, prior)
    variants = {
        "per-row argmax, no pooling": Prow.argmax(1),
        "mean pooling + argmax": pool(Pf, 1 / 3).argmax(1)[rp],
        "Bangla row copied + argmax": Pf["bangla"].argmax(1)[rp],
        "tuned pooling + asymmetric rule (final)*": final[rp],
        "final + EM prior shift (S4)*": decide(Pem, params["t"], ood, args.ood_delta, gv)[rp],
    }
    if veto is not None:
        variants["final without guard veto*"] = decide(P, params["t"], ood, args.ood_delta, None)[rp]
    print("\n-- pooling and decision variants (official row-level score on dev, fused view weights) --")
    for name, p in variants.items():
        s, avg, usr, f1s = defense_rows(yr, fr, p)
        res["variants"][name] = {"defense": s, "avg_f1": avg, "usr": usr, "f1": f1s}
        print(f"{name:44s} Defense {s:.4f} | AvgF1 {avg:.4f} | USR {usr:.4f} | "
              + " ".join(f"{k[:4]} {v:.3f}" for k, v in f1s.items()))
    print("\n-- confusion of the final system (dev prompts, % of true row) --")
    print("true\\pred " + " ".join(f"{n[:5]:>7s}" for n in NAMES))
    for c in range(4):
        row = [(final[dev.y == c] == j).mean() * 100 for j in range(4)]
        print(f"{NAMES[c]:9s} " + " ".join(f"{v:7.1f}" for v in row))
    print(f"\nfinal scalars: w_bangla={params['wb']:.2f}  a={dict(zip(views, params['a']))}  "
          f"t={params['t']:.3f}  guard={params['guard']}  (EM prior estimate on dev: {np.round(est, 3)})")
    return res, params


def check_submission(table: list[list[str]], blind_rows: list[dict], names: list[str]) -> list[str]:
    problems = []
    if names != ["predictions.csv"]:
        problems.append(f"zip must hold only predictions.csv at the top level, found {names}")
    if not table or table[0] != ["id", "label"]:
        return problems + [f"header must be id,label, found {table[:1]}"]
    ids = [row[0] for row in table[1:]]
    want = {r["id"] for r in blind_rows}
    if len(ids) != len(set(ids)):
        problems.append(f"{len(ids) - len(set(ids))} repeated ids")
    if set(ids) != want:
        problems.append(f"{len(want - set(ids))} missing and {len(set(ids) - want)} extra ids")
    if any(row[1] not in {"0", "1", "2", "3"} for row in table[1:]):
        problems.append("labels must be 0, 1, 2 or 3")
    return problems


def write_submission(blind: Split, pred: np.ndarray, P: np.ndarray, meta: dict, args) -> None:
    labels = pred[blind.row_prompt]  # S6: every row gets its prompt's label
    out = os.path.join(args.out_dir, blind.name)  # one folder per phase: never upload the wrong file
    os.makedirs(out, exist_ok=True)
    csv_path = os.path.join(out, "predictions.csv")
    with open(csv_path, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["id", "label"])
        w.writerows((r["id"], int(l)) for r, l in zip(blind.rows, labels))
    zip_path = os.path.join(out, f"submission_{blind.name}.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(csv_path, arcname="predictions.csv")
    with zipfile.ZipFile(zip_path) as z:  # read it back, as the scorer will
        names = z.namelist()
        table = list(csv.reader(io.TextIOWrapper(z.open("predictions.csv"), encoding="utf-8")))
    problems = check_submission(table, blind.rows, names)
    np.save(os.path.join(out, "prompt_probs.npy"), P)
    meta["created"] = time.strftime("%Y-%m-%d %H:%M:%S")  # match the uploaded zip to its config
    with open(os.path.join(out, "params.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    dist = np.bincount(pred, minlength=4) / len(pred)
    log("predicted prompt labels: " + " · ".join(f"{n} {d:.1%}" for n, d in zip(NAMES, dist)))
    if problems:
        raise SystemExit("submission NOT ready: " + "; ".join(problems))
    log(f"ready: {zip_path} ({len(table) - 1} rows, one per row of {blind.name})")


# ----------------------------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------------------------
def resolve_views(spec: str) -> list[str]:
    if spec != "auto":
        views = [v.strip() for v in spec.split(",") if v.strip()]
        bad = set(views) - {"lex", "dense", "llm"}
        if bad:
            raise SystemExit(f"unknown views {bad}; choose from lex, dense, llm")
        return views
    views = ["lex"]
    try:
        import sentence_transformers  # noqa: F401
        views.append("dense")
    except Exception:  # noqa: BLE001
        log("auto: sentence-transformers not installed -> dense view off")
    try:
        import torch
        import transformers  # noqa: F401
        if torch.cuda.is_available():
            views.append("llm")
        else:
            log("auto: no CUDA GPU -> LLM view off")
    except Exception:  # noqa: BLE001
        log("auto: torch/transformers not installed -> LLM view off")
    return views


def parse_args(argv=None):
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description="BanglaKnight training-free pipeline v2",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--mode", choices=["dev", "predict"], default="dev")
    ap.add_argument("--phase", choices=["devtest", "test"], default="devtest", help="blind file for --mode predict")
    ap.add_argument("--data-dir", default=os.path.join(here, "data"))
    ap.add_argument("--out-dir", default=os.path.join(here, "outputs"))
    ap.add_argument("--cache-dir", default=os.path.join(here, "cache"))
    ap.add_argument("--views", default="auto", help="comma list of lex,dense,llm or 'auto'")
    ap.add_argument("--k", type=int, default=25, help="neighbours per query")
    ap.add_argument("--lex-dim", type=int, default=1 << 14, help="hashed n-gram dims (power of 2)")
    ap.add_argument("--tau-lex", type=float, default=0.05, help="kNN softmax temperature, lexical")
    ap.add_argument("--tau-dense", type=float, default=0.02)
    ap.add_argument("--tau-llm", type=float, default=0.05)
    ap.add_argument("--dense-model", default="Qwen/Qwen3-Embedding-0.6B")
    ap.add_argument("--dense-instruction",
                    default="Identify the harmful intent and risk level of this user request")
    ap.add_argument("--no-instruction", action="store_true", help="ablation A2: embed without the instruction")
    ap.add_argument("--dense-batch", type=int, default=32)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--llm-model", default="Qwen/Qwen3-4B-Base")
    ap.add_argument("--llm-layer", type=int, default=-1, help="hidden-state layer for LLM features")
    ap.add_argument("--llm-batch", type=int, default=8)
    ap.add_argument("--llm-max-text-tokens", type=int, default=384)
    ap.add_argument("--llm-forms", default="bangla", help="forms the LLM view runs on")
    ap.add_argument("--guard-model", default="", help="e.g. Qwen/Qwen3Guard-Gen-0.6B (empty = off)")
    ap.add_argument("--guard-strict", action="store_true", help="treat 'Controversial' as Unsafe")
    ap.add_argument("--guard-batch", type=int, default=16)
    ap.add_argument("--prior-shift", choices=["none", "em"], default="none")
    ap.add_argument("--ood-view", default="auto", help="view whose similarity flags out-of-distribution prompts")
    ap.add_argument("--ood-pct", type=float, default=5.0, help="dev percentile of NN similarity = OOD cut")
    ap.add_argument("--ood-delta", type=float, default=0.10, help="extra Safe threshold for OOD prompts")
    ap.add_argument("--a-step", type=float, default=0.1, help="grid step for view weights")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-params", type=float, default=7e9)
    return ap.parse_args(argv)


def main(argv=None) -> None:
    try:  # Windows consoles default to cp1252
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    args = parse_args(argv)
    if args.lex_dim & (args.lex_dim - 1):
        raise SystemExit("--lex-dim must be a power of two")
    os.makedirs(args.out_dir, exist_ok=True)
    views = resolve_views(args.views)
    ood_view = args.ood_view if args.ood_view != "auto" else ("dense" if "dense" in views else "lex")
    if ood_view not in views:
        raise SystemExit(f"--ood-view {ood_view} is not an active view {views}")
    log(f"mode={args.mode} views={views} ood_view={ood_view} guard={args.guard_model or 'off'}")

    train = Split("train", read_rows(args.data_dir, "train"))
    dev = Split("dev", read_rows(args.data_dir, "dev"))
    log(f"S0 train {train.n} prompts / {len(train.rows)} rows · dev {dev.n} / {len(dev.rows)}")

    # ---- pass 1: bank = train, queries = dev (tuning + evaluation) ----
    prior_tr = label_prior(train.y)
    Pd, topd = compute_views(views, [train], dev, args, prior_tr)
    sim_dev = prompt_top(topd[ood_view])
    ood_cut = float(np.percentile(sim_dev[np.isfinite(sim_dev)], args.ood_pct))
    ood_dev = sim_dev < ood_cut
    veto_dev = guard_veto(args, dev) if args.guard_model else None
    Pviews = [Pd[v] for v in views]

    if args.mode == "dev":
        res, params = dev_report(views, Pviews, dev, ood_dev, veto_dev, prior_tr, args)
        res.update({"ood_cut": ood_cut, "args": vars(args)})
        path = os.path.join(args.out_dir, "dev_report.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(res, fh, indent=2, default=float)
        log(f"saved {path}")
        return

    cv, sd, full, params = cv_tune(Pviews, dev.y, ood_dev, args.ood_delta, veto_dev, args)
    log(f"S3-S5 tuned on dev: CV Defense {cv:.4f} ± {sd:.4f} (tuned-on-all {full:.4f}) params={params}")

    # ---- pass 2: bank = train + dev, queries = blind file ----
    blind = Split(args.phase, read_rows(args.data_dir, args.phase))
    log(f"S0 {blind.name}: {blind.n} prompts / {len(blind.rows)} rows")
    prior_all = label_prior(np.concatenate([train.y, dev.y]))
    Pb, topb = compute_views(views, [train, dev], blind, args, prior_all)
    ood_b = prompt_top(topb[ood_view]) < ood_cut
    log(f"S5 out-of-distribution prompts in {blind.name}: {ood_b.mean():.1%} (dev: {ood_dev.mean():.1%})")
    veto_b = guard_veto(args, blind) if (args.guard_model and params["guard"]) else None
    P = fuse([Pb[v] for v in views], params["a"], params["wb"])
    if args.prior_shift == "em":
        P, est = em_prior_shift(P, prior_all)
        log(f"S4 EM prior estimate for {blind.name}: {np.round(est, 3)} (bank: {np.round(prior_all, 3)})")
    pred = decide(P, params["t"], ood_b, args.ood_delta, veto_b)
    meta = {"views": views, "params": params, "cv_defense": cv, "cv_sd": sd, "ood_cut": ood_cut,
            "ood_view": ood_view, "phase": args.phase, "args": vars(args)}
    write_submission(blind, pred, P, meta, args)


if __name__ == "__main__":
    main()
