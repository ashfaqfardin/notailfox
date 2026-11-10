#!/usr/bin/env python3
"""
BanglaKnight (BLP-2027 Task 1): training-free Bangla prompt-safety pipeline v2.1, in one file.

Stages (see Pipeline_Review.md and pipeline_v2.png)
  S0  group the 3 written forms of each prompt by prompt_id
  S1  normalize text (NFC, optional csebuetnlp normalizer, lower-case Latin, collapse spaces)
  S2  frozen "views" -> the top-K labeled neighbours of every query in a labeled bank
        lex   : hashed char 2-5-gram TF-IDF                          (numpy only, CPU)
        dense : instruction-aware embedder, e.g. Qwen3-Embedding-0.6B (sentence-transformers)
        llm   : last-token hidden state of a <=4B LLM under a safety prompt
                (kNN-Prompting-style features, Bangla row only by default)
        guard : optional Qwen3Guard-Gen-0.6B "Unsafe" flag that vetoes a Safe prediction
      The neighbours are kept, so k and the vote temperature can be tuned per view cheaply.
  S3  fuse views (weights a_v), pool forms (Bangla weight w_b), then transductive label
      propagation over a kNN graph of the query prompts themselves (Zhou et al., 2004):
      unlabeled prompts that resemble each other share evidence, which helps on shifted data.
  S4  optional EM prior-shift re-weighting (ablation; --prior-shift em)
  S5  asymmetric decision: Safe iff P(Safe) >= t (+ delta if out-of-distribution) and no guard
      veto, otherwise the best unsafe class after small per-class biases
  S6  copy the prompt label to all of its rows -> predictions.csv + submission.zip

Training-free: no model weights are updated. Besides the kNN bank, the only fitted things are a
few scalars, chosen on dev prompts and scored with stratified k-fold cross-validation, stage by
stage so every addition is reported:
  base     fixed k, tau            -> a_v, w_b, t (+ guard on/off)
  +knn     k, tau per view          (chosen by dev log-loss)
  +bias    biases for Offensive and Severe
  +prop    label-propagation strength beta

Usage
  python banglaknight_pipeline.py --mode dev                         # evaluate on dev (bank = train)
  python banglaknight_pipeline.py --mode predict --phase devtest     # tune on dev, predict devtest
  python banglaknight_pipeline.py --mode predict --phase test --views lex,dense,llm

Requirements: Python 3.9+, numpy. The neural views also need torch, "transformers>=4.51" and
"sentence-transformers>=3" (on Colab, use the preinstalled torch; do not upgrade it), plus a CUDA
GPU for the LLM view. Every loaded model is checked against the < 7B parameter rule.
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
T_GRID = np.round(np.arange(0.15, 0.951, 0.025), 3)        # Safe thresholds
WB_GRID = (1 / 3, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)             # Bangla-form weights
K_GRID = (5, 10, 20, 40, 80)                                # neighbours per view
TAU_GRID = (0.005, 0.01, 0.02, 0.05, 0.1, 0.2)              # vote temperature per view
OFF1_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)                     # log-bias for Offensive
OFF3_GRID = (-0.5, -0.25, 0.0, 0.25, 0.5)                   # log-bias for Severe
BETA_GRID = (0.0, 0.2, 0.4, 0.6, 0.8)                       # propagation strength
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
# S2 · views -> top-K labeled neighbours
# ----------------------------------------------------------------------------------------------
def knn_topk(Q: np.ndarray, B: np.ndarray, yb: np.ndarray, K: int, chunk: int = 1024):
    """Labels and similarities of each query's K nearest bank rows, most similar first."""
    assert (yb >= 0).all(), "bank rows must be labeled"
    K = min(K, len(B))
    lab = np.zeros((len(Q), K), np.int8)
    sim = np.zeros((len(Q), K), np.float32)
    for s in range(0, len(Q), chunk):
        S = Q[s:s + chunk] @ B.T
        idx = np.argpartition(-S, K - 1, axis=1)[:, :K]
        sv = np.take_along_axis(S, idx, 1)
        order = np.argsort(-sv, axis=1)
        lab[s:s + chunk] = yb[np.take_along_axis(idx, order, 1)]
        sim[s:s + chunk] = np.take_along_axis(sv, order, 1)
    return lab, sim


def votes(lab: np.ndarray, sim: np.ndarray, k: int, tau: float) -> np.ndarray:
    """Softmax-weighted label votes of the k nearest neighbours."""
    l, s = lab[:, :k], sim[:, :k]
    w = np.exp((s - s[:, :1]) / tau)
    P = np.stack([(w * (l == c)).sum(1) for c in range(4)], 1)
    return (P / P.sum(1, keepdims=True)).astype(np.float32)


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
    Xq = np.zeros((len(tq), args.lex_dim), np.float32)
    for s in range(0, len(tq), 2048):
        Xq[s:s + 2048] = hashed_ngrams(tq[s:s + 2048], args.lex_dim) * idf
    Xq /= np.linalg.norm(Xq, axis=1, keepdims=True) + 1e-9
    lab, sim = knn_topk(Xq, Xb, yb, args.k_max)
    return lab, sim, Xq


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
        encode, extra = dense_encode, f"{args.dense_model}|{dense_prefix(args)}|{args.max_len}"
    else:
        encode = llm_encode
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
    lab, sim = knn_topk(Eq, Eb, yb, args.k_max)
    return lab, sim, Eq


def run_view(view: str, bank: list[Split], query: Split, args, need_graph: bool) -> dict:
    """Top-K neighbours per form; with need_graph, also the query-query similarity per prompt."""
    forms = tuple(args.llm_forms.split(",")) if view == "llm" else FORMS
    data, present = {}, []
    QQ = np.zeros((query.n, query.n), np.float32) if need_graph else None
    for f in forms:
        pi = query.forms[f]["prompt"]
        if len(pi) == 0:
            continue
        lab, sim, X = lexical_view(bank, query, f, args) if view == "lex" else neural_view(args, view, bank, query, f)
        data[f] = {"pi": pi, "lab": lab, "sim": sim}
        if need_graph:
            QQ[np.ix_(pi, pi)] += X @ X.T
            m = np.zeros(query.n, np.float32)
            m[pi] = 1
            present.append(m)
        log(f"  S2 view={view:5s} form={f:9s} bank={sum(len(s.forms[f]['raw']) for s in bank):6d} "
            f"queries={len(pi)}")
    if need_graph:
        QQ /= np.maximum(sum(np.outer(m, m) for m in present), 1)
    return {"forms": data, "qq": QQ}


def compute_views(views, bank, query, args, graph_view):
    log(f"S2 bank={'+'.join(s.name for s in bank)} -> queries={query.name}")
    VD = {v: run_view(v, bank, query, args, need_graph=(v == graph_view)) for v in views}
    graph = None
    if graph_view is not None:
        graph = build_graph(VD[graph_view].pop("qq"), args.graph_k)
    for v in VD.values():
        v.pop("qq", None)
    return VD, graph


def fill_missing(P: dict, prior: np.ndarray) -> dict:
    """Forms a view did not cover (or rows missing in a file) borrow the prompt's other forms."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        fallback = np.nanmean(np.stack([P[f] for f in FORMS]), axis=0)
    fallback = np.where(np.isnan(fallback), prior, fallback)
    return {f: np.where(np.isnan(P[f]), fallback, P[f]).astype(np.float32) for f in FORMS}


def view_posteriors(vd: dict, n: int, k: int, tau: float, prior: np.ndarray) -> dict:
    P = {f: np.full((n, 4), np.nan, np.float32) for f in FORMS}
    for f, d in vd["forms"].items():
        P[f][d["pi"]] = votes(d["lab"], d["sim"], k, tau)
    return fill_missing(P, prior)


def prompt_top(vd: dict, n: int) -> np.ndarray:
    """Nearest-neighbour similarity per prompt (Bangla row, else mean of the forms)."""
    top = {f: np.full(n, np.nan, np.float32) for f in FORMS}
    for f, d in vd["forms"].items():
        top[f][d["pi"]] = d["sim"][:, 0]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mean = np.nanmean(np.stack([top[f] for f in FORMS]), axis=0)
    t = np.where(np.isnan(top["bangla"]), mean, top["bangla"])
    return np.nan_to_num(t, nan=np.inf)  # no evidence -> never treated as out-of-distribution


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


# ----------------------------------------------------------------------------------------------
# S3–S5 · fuse, pool, propagate, prior shift, decide
# ----------------------------------------------------------------------------------------------
def pool(Pv: dict, wb: float) -> np.ndarray:
    wo = (1 - wb) / 2
    return wb * Pv["bangla"] + wo * (Pv["romanized"] + Pv["codemixed"])


def fuse(Pviews: list[dict], a, wb: float) -> np.ndarray:
    return sum(w * pool(Pv, wb) for w, Pv in zip(a, Pviews))


def build_graph(QQ: np.ndarray, k: int):
    """kNN graph over the query prompts (row-normalized, non-negative weights)."""
    S = QQ.copy()
    np.fill_diagonal(S, -np.inf)
    k = max(1, min(k, len(S) - 1))
    idx = np.argpartition(-S, k - 1, axis=1)[:, :k]
    w = np.maximum(np.take_along_axis(S, idx, 1), 0)
    w = w / np.maximum(w.sum(1, keepdims=True), 1e-9)
    return idx, w.astype(np.float32)


def propagate(P0: np.ndarray, graph, beta: float, iters: int = 10) -> np.ndarray:
    """Label spreading (Zhou et al., 2004) with the kNN posteriors as the anchor:
    F <- beta * W F + (1 - beta) * P0."""
    if graph is None or beta <= 0:
        return P0
    idx, w = graph
    F = P0
    for _ in range(iters):
        F = beta * np.einsum("nk,nkc->nc", w, F[idx]) + (1 - beta) * P0
    return (F / F.sum(1, keepdims=True)).astype(np.float32)


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


def unsafe_choice(P: np.ndarray, off) -> np.ndarray:
    return np.argmax(np.log(P[:, 1:] + 1e-9) + np.asarray(off, dtype=np.float32), axis=1) + 1


def decide(P, t, ood=None, delta=0.0, veto=None, off=(0.0, 0.0, 0.0)) -> np.ndarray:
    thr = t + (delta * ood if ood is not None else 0.0)
    safe = P[:, 0] >= thr
    if veto is not None:
        safe &= ~veto
    return np.where(safe, 0, unsafe_choice(P, off))


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


def best_threshold(P, y, base, veto, off):
    """Best Safe threshold t for fixed posteriors, biases and veto."""
    ps, unsafe = P[:, 0], unsafe_choice(P, off)
    best = (-1.0, None)
    for t in T_GRID:
        safe = ps >= t + base
        if veto is not None:
            safe &= ~veto
        s = defense_prompts(y, np.where(safe, 0, unsafe))[0]
        if s > best[0] + 1e-12:
            best = (s, float(t))
    return best


def logloss(P: np.ndarray, y: np.ndarray) -> float:
    return float(-np.mean(np.log(P[np.arange(len(y)), y] + 1e-6)))


def pick_stage(cv: dict, final: list, args) -> tuple[str, dict]:
    """The stage with the best CV score (stages are cumulative). --force-beta uses the last stage."""
    if args.force_beta is not None and final[-1][0] == "+ label propagation":
        return final[-1]
    return max(final, key=lambda st: np.mean(cv[st[0]]))


def tune(VD, views, y, idx, ood, delta, veto, graph, prior, n, args) -> list[tuple[str, dict]]:
    """Stage-wise grid search on the prompts in idx. Returns the cumulative params of each stage."""
    y_i, base_i = y[idx], delta * ood[idx]
    zero = (0.0, 0.0, 0.0)

    def fit_views(kt):
        Pviews = [view_posteriors(VD[v], n, *kt[v], prior) for v in views]
        best = (-1.0, None)
        for wb in WB_GRID:
            pooled = [pool(Pv, wb)[idx] for Pv in Pviews]
            for a in simplex(len(pooled), args.a_step):
                P = sum(w * pp for w, pp in zip(a, pooled))
                for g in ((False, True) if veto is not None else (False,)):
                    s, t = best_threshold(P, y_i, base_i, veto[idx] if g else None, zero)
                    if s > best[0] + 1e-12:
                        best = (s, {"wb": float(wb), "a": [float(x) for x in a], "t": t, "guard": g})
        prm = dict(best[1], kt={v: [int(kt[v][0]), float(kt[v][1])] for v in views}, off=list(zero), beta=0.0)
        return Pviews, prm

    stages = []
    base_kt = {v: (args.k, {"lex": args.tau_lex, "dense": args.tau_dense, "llm": args.tau_llm}[v]) for v in views}
    _, prm = fit_views(base_kt)
    stages.append(("base (fixed k, tau)", prm))

    if not args.no_knn_tuning:
        kt = {}
        for v in views:  # proper scoring rule for the smoothing; the metric is used for the decision
            best = (np.inf, None)
            for k in K_GRID:
                for tau in TAU_GRID:
                    ll = logloss(pool(view_posteriors(VD[v], n, k, tau, prior), 1 / 3)[idx], y_i)
                    if ll < best[0]:
                        best = (ll, (k, tau))
            kt[v] = best[1]
        Pviews, prm = fit_views(kt)
        stages.append(("+ k, tau per view", prm))
    else:
        Pviews = [view_posteriors(VD[v], n, *base_kt[v], prior) for v in views]

    P = fuse(Pviews, prm["a"], prm["wb"])
    v_i = veto[idx] if prm["guard"] else None
    if not args.no_bias:
        best = (-1.0, None)
        for o1 in OFF1_GRID:
            for o3 in OFF3_GRID:
                s, t = best_threshold(P[idx], y_i, base_i, v_i, (o1, 0.0, o3))
                if s > best[0] + 1e-12:
                    best = (s, (o1, o3, t))
        prm = dict(prm, off=[best[1][0], 0.0, best[1][1]], t=best[1][2])
        stages.append(("+ class biases", prm))

    if graph is not None:
        best = (-1.0, None)
        for beta in ((args.force_beta,) if args.force_beta is not None else BETA_GRID):
            s, t = best_threshold(propagate(P, graph, beta)[idx], y_i, base_i, v_i, prm["off"])
            if s > best[0] + 1e-12:
                best = (s, (beta, t))
        prm = dict(prm, beta=best[1][0], t=best[1][1])
        stages.append(("+ label propagation", prm))
    return stages


def apply_params(VD, views, prm, n, prior, graph) -> np.ndarray:
    Pviews = [view_posteriors(VD[v], n, *prm["kt"][v], prior) for v in views]
    return propagate(fuse(Pviews, prm["a"], prm["wb"]), graph, prm["beta"])


def predict_with(prm, P, ood, delta, veto):
    return decide(P, prm["t"], ood, delta, veto if prm["guard"] else None, prm["off"])


def stratified_folds(y: np.ndarray, k: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    folds: list[list[int]] = [[] for _ in range(k)]
    for c in np.unique(y):
        for j, part in enumerate(np.array_split(rng.permutation(np.where(y == c)[0]), k)):
            folds[j].extend(part.tolist())
    return [np.array(sorted(f)) for f in folds]


def cv_tune(VD, views, y, ood, delta, veto, graph, prior, n, args):
    """CV score of every tuning stage, plus each stage's params tuned on all of dev.
    Propagation only reads kNN posteriors (never dev labels), so the graph may span all prompts."""
    folds = stratified_folds(y, args.folds, args.seed)
    cv: dict[str, list[float]] = {}
    for j, te in enumerate(folds):
        tr = np.concatenate([f for i, f in enumerate(folds) if i != j])
        for name, prm in tune(VD, views, y, tr, ood, delta, veto, graph, prior, n, args):
            P = apply_params(VD, views, prm, n, prior, graph)[te]
            p = predict_with(prm, P, ood[te], delta, veto[te] if veto is not None else None)
            cv.setdefault(name, []).append(defense_prompts(y[te], p)[0])
    final = tune(VD, views, y, np.arange(n), ood, delta, veto, graph, prior, n, args)
    full = {}
    for name, prm in final:
        p = predict_with(prm, apply_params(VD, views, prm, n, prior, graph), ood, delta, veto)
        full[name] = defense_prompts(y, p)[0]
    return cv, final, full


# ----------------------------------------------------------------------------------------------
# reports and submission
# ----------------------------------------------------------------------------------------------
def print_stages(cv, final, full, title, chosen):
    print(f"\n{title}")
    print(f"{'stage (cumulative)':40s} {'CV Defense':>18s} {'tuned on all dev*':>18s}")
    for name, _ in final:
        sc = np.array(cv[name])
        mark = "  <- used" if name == chosen else ""
        print(f"{name:40s} {sc.mean():10.4f} ± {sc.std():.4f} {full[name]:18.4f}{mark}")


def dev_report(views, VD, dev: Split, ood, veto, graph, prior, args):
    n = dev.n
    res: dict = {"views": views, "stages": {}, "views_alone": {}, "variants": {}}
    cv, final, full = cv_tune(VD, views, dev.y, ood, args.ood_delta, veto, graph, prior, n, args)
    chosen, prm = pick_stage(cv, final, args)
    print_stages(cv, final, full, f"== Dev: bank = train, {args.folds}-fold CV over {n} prompts, "
                                  f"views {'+'.join(views)} ==", chosen)
    print("* tuned and scored on the same prompts, so optimistic; the CV column is the honest estimate.")
    for name, prm in final:
        res["stages"][name] = {"cv": float(np.mean(cv[name])), "sd": float(np.std(cv[name])),
                               "dev_tuned": full[name], "params": prm}
    if len(views) > 1:
        print("\n-- each view alone (final stage, CV) --")
        for v in views:
            cv_v, final_v, _ = cv_tune({v: VD[v]}, [v], dev.y, ood, args.ood_delta, None, graph, prior, n, args)
            last = final_v[-1][0]
            res["views_alone"][v] = float(np.mean(cv_v[last]))
            print(f"{'view ' + v:40s} {np.mean(cv_v[last]):10.4f} ± {np.std(cv_v[last]):.4f}")

    Pviews = [view_posteriors(VD[v], n, *prm["kt"][v], prior) for v in views]
    Pf = {f: sum(w * Pv[f] for w, Pv in zip(prm["a"], Pviews)) for f in FORMS}
    rp, yr = dev.row_prompt, dev.y[dev.row_prompt]
    fr = np.array([r["form"] for r in dev.rows])
    Prow = np.stack([(Pf[f] if f in Pf else Pf["bangla"])[i] for f, i in zip(fr, rp)])
    P = apply_params(VD, views, prm, n, prior, graph)
    final_pred = predict_with(prm, P, ood, args.ood_delta, veto)
    Pem, est = em_prior_shift(P, prior)
    variants = {
        "per-row argmax, no pooling": Prow.argmax(1),
        "mean pooling + argmax": pool(Pf, 1 / 3).argmax(1)[rp],
        "final system*": final_pred[rp],
        "final + EM prior shift (S4)*": predict_with(prm, Pem, ood, args.ood_delta, veto)[rp],
    }
    print("\n-- variants (official row-level score on dev) --")
    for name, p in variants.items():
        s, avg, usr, f1s = defense_rows(yr, fr, p)
        res["variants"][name] = {"defense": s, "avg_f1": avg, "usr": usr, "f1": f1s}
        print(f"{name:40s} Defense {s:.4f} | AvgF1 {avg:.4f} | USR {usr:.4f}")
    print("\n-- confusion of the final system (dev prompts, % of true row) --")
    print("true\\pred " + " ".join(f"{nm[:5]:>7s}" for nm in NAMES))
    for c in range(4):
        print(f"{NAMES[c]:9s} " + " ".join(f"{(final_pred[dev.y == c] == j).mean() * 100:7.1f}" for j in range(4)))
    print(f"\nfinal scalars: {json.dumps(prm)}   (EM prior estimate on dev: {np.round(est, 3)})")
    return res


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
        json.dump(meta, fh, indent=2, default=float)
    dist = np.bincount(pred, minlength=4) / len(pred)
    log("predicted prompt labels: " + " · ".join(f"{nm} {d:.1%}" for nm, d in zip(NAMES, dist)))
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
    except Exception as e:  # noqa: BLE001
        log(f"auto: sentence-transformers not importable ({str(e).splitlines()[0][:120]}) -> dense view off")
    try:
        import torch
        import transformers  # noqa: F401
        if torch.cuda.is_available():
            views.append("llm")
        else:
            log("auto: no CUDA GPU -> LLM view off")
    except Exception as e:  # noqa: BLE001
        log(f"auto: torch/transformers not importable ({str(e).splitlines()[0][:120]}) -> LLM view off")
    return views


def preflight(views: list[str], args) -> None:
    """Fail in the first second, with a fix, instead of deep inside a neural view."""
    if not ({"dense", "llm"} & set(views) or args.guard_model):
        return
    import importlib
    import importlib.util
    try:
        import torch
    except Exception as e:  # noqa: BLE001
        raise SystemExit(f"torch is not importable: {e}\n  fix: pip install torch")
    # transformers imports torchaudio/torchvision automatically when they are installed, so a copy
    # built for a different torch/CUDA breaks every model import, even for text-only models.
    for mod in ("torchaudio", "torchvision"):
        if importlib.util.find_spec(mod) is not None:
            try:
                importlib.import_module(mod)
            except Exception as e:  # noqa: BLE001
                raise SystemExit(
                    f"{mod} is installed but does not match torch {torch.__version__}: "
                    f"{str(e).splitlines()[0][:200]}\n"
                    f"  fix: pip uninstall -y {mod}   (this pipeline does not use it), then rerun.\n"
                    "  or:  reinstall torch, torchvision and torchaudio together, then restart the runtime.")
    if "dense" in views:
        try:
            from sentence_transformers import SentenceTransformer  # noqa: F401
        except Exception as e:  # noqa: BLE001
            raise SystemExit(f"sentence-transformers is not importable: {str(e).splitlines()[0][:200]}\n"
                             '  fix: pip install -U "sentence-transformers>=3"')
    if "llm" in views or args.guard_model:
        try:
            from transformers import AutoModel, AutoTokenizer  # noqa: F401
        except Exception as e:  # noqa: BLE001
            raise SystemExit(f"transformers is not importable: {str(e).splitlines()[0][:200]}\n"
                             '  fix: pip install -U "transformers>=4.51"')
        if not torch.cuda.is_available():
            raise SystemExit("the LLM view / guard needs a CUDA GPU, but torch sees none "
                             f"(torch {torch.__version__}). On Colab: Runtime > Change runtime type > GPU. "
                             "If you upgraded torch, it may be too new for the GPU driver.")
    log(f"preflight ok: torch {torch.__version__}, CUDA "
        f"{'yes: ' + torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no'}")


def parse_args(argv=None):
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description="BanglaKnight training-free pipeline v2.1",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--mode", choices=["dev", "predict"], default="dev")
    ap.add_argument("--phase", choices=["devtest", "test"], default="devtest", help="blind file for --mode predict")
    ap.add_argument("--data-dir", default=os.path.join(here, "data"))
    ap.add_argument("--out-dir", default=os.path.join(here, "outputs"))
    ap.add_argument("--cache-dir", default=os.path.join(here, "cache"))
    ap.add_argument("--views", default="auto", help="comma list of lex,dense,llm or 'auto'")
    ap.add_argument("--k", type=int, default=25, help="neighbours per query in the 'base' stage")
    ap.add_argument("--k-max", type=int, default=100, help="neighbours stored per query (>= max of the k grid)")
    ap.add_argument("--lex-dim", type=int, default=1 << 14, help="hashed n-gram dims (power of 2)")
    ap.add_argument("--tau-lex", type=float, default=0.05, help="base-stage vote temperature, lexical")
    ap.add_argument("--tau-dense", type=float, default=0.02)
    ap.add_argument("--tau-llm", type=float, default=0.05)
    ap.add_argument("--no-knn-tuning", action="store_true", help="skip the k, tau stage")
    ap.add_argument("--no-bias", action="store_true", help="skip the class-bias stage")
    ap.add_argument("--no-propagation", action="store_true", help="skip label propagation")
    ap.add_argument("--graph-view", default="auto", help="view whose similarities build the query graph")
    ap.add_argument("--graph-k", type=int, default=10, help="neighbours per prompt in the query graph")
    ap.add_argument("--force-beta", type=float, default=None,
                    help="fix the propagation strength (dev cannot measure the dev-test shift; probe it on "
                         "the dev-test leaderboard, e.g. 0.4)")
    ap.add_argument("--dense-model", default="Qwen/Qwen3-Embedding-0.6B")
    ap.add_argument("--dense-instruction",
                    default="Identify the harmful intent and risk level of this user request")
    ap.add_argument("--no-instruction", action="store_true", help="ablation: embed without the instruction")
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
    if args.k_max < max(K_GRID + (args.k,)):
        raise SystemExit(f"--k-max must be at least {max(K_GRID + (args.k,))}")
    os.makedirs(args.out_dir, exist_ok=True)
    views = resolve_views(args.views)
    preflight(views, args)
    pick = lambda spec: spec if spec != "auto" else ("dense" if "dense" in views else "lex")  # noqa: E731
    ood_view = pick(args.ood_view)
    graph_view = None if args.no_propagation else pick(args.graph_view)
    for name, v in (("--ood-view", ood_view), ("--graph-view", graph_view)):
        if v is not None and v not in views:
            raise SystemExit(f"{name} {v} is not an active view {views}")
    log(f"mode={args.mode} views={views} ood_view={ood_view} graph_view={graph_view} "
        f"guard={args.guard_model or 'off'}")

    train = Split("train", read_rows(args.data_dir, "train"))
    dev = Split("dev", read_rows(args.data_dir, "dev"))
    log(f"S0 train {train.n} prompts / {len(train.rows)} rows · dev {dev.n} / {len(dev.rows)}")

    # ---- pass 1: bank = train, queries = dev (tuning + evaluation) ----
    prior_tr = label_prior(train.y)
    VDd, graph_d = compute_views(views, [train], dev, args, graph_view)
    sim_dev = prompt_top(VDd[ood_view], dev.n)
    ood_cut = float(np.percentile(sim_dev[np.isfinite(sim_dev)], args.ood_pct))
    ood_dev = sim_dev < ood_cut
    veto_dev = guard_veto(args, dev) if args.guard_model else None
    log("S3-S5 tuning (this takes a minute)")

    if args.mode == "dev":
        res = dev_report(views, VDd, dev, ood_dev, veto_dev, graph_d, prior_tr, args)
        res.update({"ood_cut": ood_cut, "args": vars(args)})
        path = os.path.join(args.out_dir, "dev_report.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(res, fh, indent=2, default=float)
        log(f"saved {path}")
        return

    cv, final, full = cv_tune(VDd, views, dev.y, ood_dev, args.ood_delta, veto_dev, graph_d, prior_tr, dev.n, args)
    name, params = pick_stage(cv, final, args)
    print_stages(cv, final, full, "== tuning on dev (bank = train) ==", name)
    log(f"using stage '{name}': {json.dumps(params)}")

    # ---- pass 2: bank = train + dev, queries = blind file ----
    blind = Split(args.phase, read_rows(args.data_dir, args.phase))
    log(f"S0 {blind.name}: {blind.n} prompts / {len(blind.rows)} rows")
    prior_all = label_prior(np.concatenate([train.y, dev.y]))
    VDb, graph_b = compute_views(views, [train, dev], blind, args, graph_view)
    ood_b = prompt_top(VDb[ood_view], blind.n) < ood_cut
    log(f"S5 out-of-distribution prompts in {blind.name}: {ood_b.mean():.1%} (dev: {ood_dev.mean():.1%})")
    veto_b = guard_veto(args, blind) if (args.guard_model and params["guard"]) else None
    P = apply_params(VDb, views, params, blind.n, prior_all, graph_b)
    if args.prior_shift == "em":
        P, est = em_prior_shift(P, prior_all)
        log(f"S4 EM prior estimate for {blind.name}: {np.round(est, 3)} (bank: {np.round(prior_all, 3)})")
    pred = predict_with(params, P, ood_b, args.ood_delta, veto_b)
    meta = {"views": views, "stage": name, "params": params,
            "cv": {k: float(np.mean(v)) for k, v in cv.items()}, "ood_cut": ood_cut,
            "ood_view": ood_view, "graph_view": graph_view, "phase": args.phase, "args": vars(args)}
    write_submission(blind, pred, P, meta, args)


if __name__ == "__main__":
    main()
