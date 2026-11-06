"""Compare the SAEs of an ablation table (scripts/make_ablations.py), across seeds and encoders.

All stores hold the same tiles in the same order (scripts/import_trident.py), so each SAE is
run on the same ``--n-rows`` random test tiles. Writes to ``--out``:

  ``metrics.csv``       test metrics of every run (eval/summary.json) and its groups
  ``main_summary.csv``  mean and std over seeds of the ``main`` runs, per encoder
  ``matching.csv``      concept matching between run pairs: for every concept active on at least
                        ``--min-freq`` of the tiles, its best Pearson correlation with any concept
                        of the other SAE (A->B and B->A). Seed pairs give the within-encoder baseline;
                        encoder pairs say which concepts the two encoders share.
  ``matches_<A>__<B>.parquet``  per-concept best match of every pair
  ``probes.csv``        GBM vs LGG logistic probes on raw embeddings and on SAE codes
                        (tile AUC and slide AUC of the mean tile probability), for ``--probe-groups``

    python scripts/compare_saes.py ablations/sae_brain.tsv --device cuda
"""
import argparse, csv, itertools, json, os, pathlib

import numpy as np
import polars as pl
import torch

from picasso.loader import EmbeddingStore
from picasso.model import load


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("table", type=pathlib.Path)
    p.add_argument("--data-root", type=pathlib.Path, default=pathlib.Path(os.environ.get("DATA_ROOT", "data")))
    p.add_argument("--output-root", type=pathlib.Path, default=pathlib.Path(os.environ.get("OUTPUT_ROOT", "outputs")))
    p.add_argument("--out", type=pathlib.Path, default=None, help="default: <output-root>/compare/<table name>")
    p.add_argument("--n-rows", type=int, default=200_000, help="test tiles used for matching")
    p.add_argument("--min-freq", type=float, default=1e-3, help="concepts rarer than this are not matched")
    p.add_argument("--probe-groups", default="main", help="comma-separated groups to probe ('' = none)")
    p.add_argument("--probe-train-rows", type=int, default=100_000)
    p.add_argument("--batch-size", type=int, default=8192)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def read_table(path: pathlib.Path) -> list[dict]:
    with open(path) as fd:
        rows = list(csv.DictReader(fd, delimiter="\t"))
    for r in rows:
        r["groups"] = r["groups"].split(",")
        r["field"] = next((kv.split("=", 1)[1] for kv in r["train_set"].split() if kv.startswith("field=")), "cls")
    return rows


@torch.inference_mode()
def encode(run: dict, x: torch.Tensor, a) -> tuple[np.ndarray, np.ndarray, int]:
    """TopK indices and values [n, k] of ``x`` under the run's SAE."""
    ckpt = a.output_root / "sae" / run["run_name"] / "sae.pt"
    with open(ckpt, "rb") as fd:
        header = json.loads(fd.readline())
    sae = load(ckpt, a.device)
    store = EmbeddingStore(a.data_root / "store" / run["store"], run["field"], "train")
    mean, std = store.mean.to(a.device), store.std.to(a.device)
    idx, val = [], []
    for s in range(0, len(x), a.batch_size):
        b = x[s : s + a.batch_size].to(a.device)
        if header.get("normalize_input"):
            b = (b - mean) / std
        v, i = sae.pre_activation(b).topk(sae.cfg.top_k, dim=-1)
        idx.append(i.cpu().numpy()); val.append(v.clamp_min(0).cpu().numpy())
    return np.concatenate(idx), np.concatenate(val), sae.cfg.d_sae


def standardized_columns(idx: np.ndarray, val: np.ndarray, d_sae: int, min_freq: float):
    """Concepts active on >= min_freq of rows, with their mean and std over all rows."""
    n = len(idx)
    fired = np.bincount(idx[val > 0], minlength=d_sae)
    s1 = np.bincount(idx.ravel(), weights=val.ravel(), minlength=d_sae)
    s2 = np.bincount(idx.ravel(), weights=(val.astype(np.float64) ** 2).ravel(), minlength=d_sae)
    keep = np.flatnonzero(fired >= max(min_freq * n, 5))
    mean = s1[keep] / n
    std = np.sqrt(np.maximum(s2[keep] / n - mean**2, 1e-12))
    return keep, mean, std, fired[keep] / n


def dense_chunk(idx, val, cols, start, stop, device) -> torch.Tensor:
    """Rows [start, stop) as a dense [rows, len(cols)] matrix over the kept concepts."""
    col_of = np.full(idx.max() + 1, -1, dtype=np.int64)
    col_of[cols] = np.arange(len(cols))
    i, v = idx[start:stop], val[start:stop]
    c = col_of[i]
    r = np.broadcast_to(np.arange(stop - start)[:, None], c.shape)
    m = (c >= 0) & (v > 0)
    out = torch.zeros(stop - start, len(cols), device=device)
    out[torch.from_numpy(r[m]).to(device), torch.from_numpy(c[m]).to(device)] = torch.from_numpy(v[m]).float().to(device)
    return out


@torch.inference_mode()
def match(A: tuple, B: tuple, a) -> tuple[dict, pl.DataFrame]:
    """Best Pearson correlation of every kept concept of A with any kept concept of B, and back."""
    (ia, va, da), (ib, vb, db) = A, B
    ka, ma, sa, fa = standardized_columns(ia, va, da, a.min_freq)
    kb, mb, sb, fb = standardized_columns(ib, vb, db, a.min_freq)
    n = len(ia)
    cross = torch.zeros(len(ka), len(kb), device=a.device)
    for s in range(0, n, a.batch_size):
        e = min(s + a.batch_size, n)
        cross += dense_chunk(ia, va, ka, s, e, a.device).T @ dense_chunk(ib, vb, kb, s, e, a.device)
    to_t = lambda z: torch.from_numpy(z).float().to(a.device)
    corr = (cross / n - to_t(ma)[:, None] * to_t(mb)[None, :]) / (to_t(sa)[:, None] * to_t(sb)[None, :])
    best_ab, arg_ab = corr.max(1)
    best_ba, _ = corr.max(0)
    best_ab, best_ba = best_ab.cpu().numpy(), best_ba.cpu().numpy()
    summary = {"n_concepts_a": len(ka), "n_concepts_b": len(kb)}
    for tag, best in (("ab", best_ab), ("ba", best_ba)):
        summary |= {f"{tag}_mean": float(best.mean()), f"{tag}_median": float(np.median(best)),
                    f"{tag}_frac_gt_0.5": float((best > 0.5).mean()), f"{tag}_frac_gt_0.7": float((best > 0.7).mean())}
    per_concept = pl.DataFrame({"concept_a": ka, "freq_a": fa, "best_b": kb[arg_ab.cpu().numpy()], "corr": best_ab})
    return summary, per_concept


def probe(x_tr, y_tr, x_te, y_te, slide_te) -> dict:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    clf = LogisticRegression(max_iter=2000, C=1.0).fit(x_tr, y_tr)
    prob = clf.predict_proba(x_te)[:, 1]
    df = pl.DataFrame({"slide": slide_te, "y": y_te, "p": prob}).group_by("slide").agg(pl.col("y").first(), pl.col("p").mean())
    return {"tile_auc": float(roc_auc_score(y_te, prob)), "slide_auc": float(roc_auc_score(df["y"], df["p"])),
            "n_test_slides": df.height}


def sparse_codes(idx, val, d_sae):
    from scipy.sparse import csr_matrix

    n, k = idx.shape
    return csr_matrix((val.ravel(), idx.ravel(), np.arange(0, n * k + 1, k)), shape=(n, d_sae))


def labels(store_dir: pathlib.Path, field: str, split: str, rows: np.ndarray):
    """Project label (1 = TCGA-LGG) and slide id of the given store rows."""
    ix = pl.read_parquet(store_dir / f"{field}_{split}.index.parquet")
    pos = np.searchsorted(ix["start"].to_numpy(), rows, side="right") - 1
    return (ix["project"].to_numpy()[pos] == "TCGA-LGG").astype(int), ix["slide_id"].to_numpy()[pos]


def main() -> None:
    a = parse_args()
    runs = [r for r in read_table(a.table) if (a.output_root / "sae" / r["run_name"] / "eval" / "summary.json").exists()]
    out = a.out or a.output_root / "compare" / a.table.stem
    out.mkdir(parents=True, exist_ok=True)
    print(f"{len(runs)} finished runs -> {out}")

    # 1. Test metrics of every run.
    recs = []
    for r in runs:
        s = json.loads((a.output_root / "sae" / r["run_name"] / "eval" / "summary.json").read_text())
        recs.append({"run_name": r["run_name"], "groups": ",".join(r["groups"]), "encoder": r["encoder"], "store": r["store"],
                     "seed": int(r["seed"]), **{k: v for k, v in s.items() if isinstance(v, (int, float))},
                     **{f"baseline_{k}": v for k, v in s.get("baselines", {}).items()}})
    metrics = pl.DataFrame(recs)
    metrics.write_csv(out / "metrics.csv")
    main_runs = metrics.filter(pl.col("groups").str.contains("main"))
    if main_runs.height:
        cols = ["explained_variance", "explained_variance_per_sample", "cosine", "n_dead", "n_dense", "n_active_1e-3"]
        main_runs.group_by("encoder").agg([pl.len().alias("n_seeds")] + [pl.col(c).mean().alias(f"{c}_mean") for c in cols]
                                          + [pl.col(c).std().alias(f"{c}_std") for c in cols]).sort("encoder").write_csv(out / "main_summary.csv")

    # 2. Concept matching on the same test tiles.
    by_store = {}
    for r in runs:
        by_store.setdefault((r["store"], r["field"]), []).append(r)
    ref_index = None
    for store, field in by_store:
        ix = pl.read_parquet(a.data_root / "store" / store / f"{field}_test.index.parquet").select("slide_id", "n")
        if ref_index is None:
            ref_index = ix
        elif not ix.equals(ref_index):
            raise SystemExit(f"store {store} holds other test tiles than the first store; re-import with one import_trident.py run")
    n_test = int(ref_index["n"].sum())
    rows = np.sort(np.random.default_rng(0).choice(n_test, size=min(a.n_rows, n_test), replace=False))
    codes = {}
    for (store, field), rs in by_store.items():
        x = EmbeddingStore(a.data_root / "store" / store, field, "test").rows(rows)
        for r in rs:
            codes[r["run_name"]] = encode(r, x, a)
    main_names = [r["run_name"] for r in runs if "main" in r["groups"]]
    enc_of = {r["run_name"]: r["encoder"] for r in runs}
    pairs = [(p, q, "seed" if enc_of[p] == enc_of[q] else "encoder") for p, q in itertools.combinations(main_names, 2)]
    match_rows = []
    for p, q, kind in pairs:
        summary, per = match(codes[p], codes[q], a)
        per.write_parquet(out / f"matches_{p}__{q}.parquet")
        match_rows.append({"a": p, "b": q, "kind": kind, **summary})
        print(f"{kind:7s} {p} vs {q}: A->B mean {summary['ab_mean']:.3f}, >0.7 {summary['ab_frac_gt_0.7']:.1%}")
    if match_rows:
        pl.DataFrame(match_rows).write_csv(out / "matching.csv")

    # 3. GBM vs LGG probes: raw embedding per store, and SAE codes per run.
    groups = set(filter(None, a.probe_groups.split(",")))
    to_probe = [r for r in runs if groups & set(r["groups"])]
    if to_probe:
        probe_rows = []
        stores = {(r["store"], r["field"]) for r in to_probe}
        n_train = EmbeddingStore(a.data_root / "store" / next(iter(stores))[0], next(iter(stores))[1], "train").n
        tr_rows = np.sort(np.random.default_rng(1).choice(n_train, size=min(a.probe_train_rows, n_train), replace=False))
        for store, field in sorted(stores):
            sd = a.data_root / "store" / store
            x_tr = EmbeddingStore(sd, field, "train").rows(tr_rows)
            x_te = EmbeddingStore(sd, field, "test").rows(rows)
            y_tr, _ = labels(sd, field, "train", tr_rows)
            y_te, slide_te = labels(sd, field, "test", rows)
            mu, sd_ = x_tr.mean(0), x_tr.std(0).clamp_min(1e-6)
            res = probe(((x_tr - mu) / sd_).numpy(), y_tr, ((x_te - mu) / sd_).numpy(), y_te, slide_te)
            probe_rows.append({"input": f"raw:{store}", **res})
            print(f"probe raw {store}: tile AUC {res['tile_auc']:.3f}, slide AUC {res['slide_auc']:.3f}")
            for r in (r for r in to_probe if (r["store"], r["field"]) == (store, field)):
                i_tr, v_tr, d_sae = encode(r, x_tr, a)
                i_te, v_te, _ = codes[r["run_name"]]
                res = probe(sparse_codes(i_tr, v_tr, d_sae), y_tr, sparse_codes(i_te, v_te, d_sae), y_te, slide_te)
                probe_rows.append({"input": f"sae:{r['run_name']}", **res})
                print(f"probe sae {r['run_name']}: tile AUC {res['tile_auc']:.3f}, slide AUC {res['slide_auc']:.3f}")
        pl.DataFrame(probe_rows).write_csv(out / "probes.csv")


if __name__ == "__main__":
    main()
