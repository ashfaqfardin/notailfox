"""Quick training-free probe: hashed char n-gram TF-IDF kNN (train -> dev), plus a devtest shift check.
Usage: python -I knn_probe.py <data_dir>
"""
import json, sys, os, zlib, unicodedata, re
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
DATA = sys.argv[1]
FORMS = ["bangla", "romanized", "codemixed"]
D = 1 << 14  # hashed feature dim


def load(name):
    with open(os.path.join(DATA, name), encoding="utf-8") as fh:
        return [json.loads(l) for l in fh if l.strip()]


def norm(t):
    t = unicodedata.normalize("NFC", t).lower()
    return re.sub(r"\s+", " ", t).strip()


def featurize(texts, idf=None):
    X = np.zeros((len(texts), D), dtype=np.float32)
    for i, t in enumerate(texts):
        t = f" {norm(t)} "
        for n in (2, 3, 4, 5):
            for j in range(len(t) - n + 1):
                X[i, zlib.crc32(t[j:j + n].encode()) & (D - 1)] += 1.0
    np.log1p(X, out=X)  # sublinear tf
    if idf is None:
        df = (X > 0).sum(0)
        idf = np.log((1 + len(texts)) / (1 + df)).astype(np.float32) + 1.0
    X *= idf
    X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-9
    return X, idf


def f1_macro(y, p):
    fs = []
    for c in range(4):
        tp = np.sum((p == c) & (y == c)); fp = np.sum((p == c) & (y != c)); fn = np.sum((p != c) & (y == c))
        fs.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(fs))


def defense(y, form, p):
    f1s = {fm: f1_macro(y[form == fm], p[form == fm]) for fm in FORMS}
    avg = float(np.mean(list(f1s.values())))
    usr = float((p[y > 0] == 0).mean())
    return avg * (1 - usr), avg, usr, f1s


def knn_proba(Xq, Xt, yt, k=10):
    S = Xq @ Xt.T
    idx = np.argpartition(-S, k, axis=1)[:, :k]
    sims = np.take_along_axis(S, idx, 1)
    P = np.zeros((len(Xq), 4), dtype=np.float32)
    w = np.maximum(sims, 0) ** 2
    for c in range(4):
        P[:, c] = (w * (yt[idx] == c)).sum(1)
    P /= P.sum(1, keepdims=True) + 1e-9
    return P, S.max(1)


def decide(P, t_safe=None):
    if t_safe is None:
        return P.argmax(1)
    unsafe = P[:, 1:].argmax(1) + 1
    return np.where(P[:, 0] >= t_safe, 0, unsafe)


tr, dv, dt = load("train.jsonl"), load("dev.jsonl"), load("devtest.jsonl")
y_dev_all, form_dev_all, pid_dev_all = (np.array([r[k] for r in dv]) for k in ("label", "form", "prompt_id"))

P_dev = np.zeros((len(dv), 4), dtype=np.float32)
maxsim = {}
for fm in FORMS:
    trf = [r for r in tr if r["form"] == fm]
    Xt, idf = featurize([r["text"] for r in trf])
    yt = np.array([r["label"] for r in trf])
    mask = form_dev_all == fm
    Xd, _ = featurize([r["text"] for r in dv if r["form"] == fm], idf)
    P, ms_dev = knn_proba(Xd, Xt, yt)
    P_dev[mask] = P
    Xdt, _ = featurize([r["text"] for r in dt if r["form"] == fm], idf)
    _, ms_dt = knn_proba(Xdt, Xt, yt)
    maxsim[fm] = (ms_dev, ms_dt)
    del Xt, Xd, Xdt

print("== Nearest-neighbour cosine to TRAIN (median / p10): dev vs devtest ==")
for fm in FORMS:
    a, b = maxsim[fm]
    print(f"  {fm:10s} dev {np.median(a):.3f}/{np.percentile(a,10):.3f}   devtest {np.median(b):.3f}/{np.percentile(b,10):.3f}")


def report(name, p):
    ds, avg, usr, f1s = defense(y_dev_all, form_dev_all, p)
    agree = np.mean([len(set(p[pid_dev_all == q])) == 1 for q in np.unique(pid_dev_all)[:2049]])
    print(f"{name:42s} Defense {ds:.4f} | AvgF1 {avg:.4f} | USR {usr:.4f} | "
          + " ".join(f"{k[:4]} {v:.3f}" for k, v in f1s.items()) + f" | consistent {agree:.2f}")


print("\n== Char n-gram kNN (k=10), train -> dev ==")
report("per-row argmax", decide(P_dev))

# pooling across forms per prompt_id
pids = np.unique(pid_dev_all)
P_pool = np.zeros_like(P_dev); P_bn = np.zeros_like(P_dev)
pid_index = {q: np.where(pid_dev_all == q)[0] for q in pids}
for q, ix in pid_index.items():
    P_pool[ix] = P_dev[ix].mean(0)
    bn = ix[form_dev_all[ix] == "bangla"][0]
    P_bn[ix] = P_dev[bn]
report("pooled over 3 forms (mean prob)", decide(P_pool))
report("Bangla-form prediction propagated", decide(P_bn))

# Safe-threshold sweep (optimistic: tuned on dev itself)
for name, P in (("per-row", P_dev), ("pooled", P_pool)):
    best = max(((defense(y_dev_all, form_dev_all, decide(P, t))[0], t) for t in np.arange(0.3, 0.95, 0.05)))
    report(f"{name} + Safe-threshold t={best[1]:.2f} (dev-tuned)", decide(P, best[1]))

# ---- per-class view for pooled + threshold ----
t = 0.5
p = decide(P_pool, t)
print("\n== Confusion (pooled, t=0.5), rows=true, cols=pred, row-normalised % ==")
names = ["Safe", "Off", "Harm", "Sev"]
for c in range(4):
    row = [(np.sum((y_dev_all == c) & (p == j)) / np.sum(y_dev_all == c)) * 100 for j in range(4)]
    print(f"  {names[c]:5s}", " ".join(f"{v:5.1f}" for v in row))
print("  per-class F1:", [round(f1, 3) for f1 in
      [2*np.sum((p==c)&(y_dev_all==c))/(np.sum(p==c)+np.sum(y_dev_all==c)) for c in range(4)]])
