"""Expand an ablation grid (``configs/ablations/*.yaml``) into one row per SAE run.

Each group is the product of its lists on top of ``base``. Runs with the same settings are
merged (the ``main`` seed-42 run is also the reference of every other group). Writes a TSV
with columns run_name, groups, encoder, store, seed, train_set, eval_set for
``scripts/run_ablation.sh``.

    python scripts/make_ablations.py configs/ablations/sae_brain.yaml     # -> ablations/sae_brain.tsv
"""
import argparse, csv, itertools, pathlib

import yaml

SPECIAL = ("encoder", "store", "seed")
SHORT = {"normalize_input": "norm", "n_prefixes": "mat", "epochs": "ep", "lr": "lr", "aux_alpha": "aux",
         "datapoint_init_blend": "dpinit", "lr_schedule": "", "in_memory": None, "device": None}

p = argparse.ArgumentParser()
p.add_argument("grid", type=pathlib.Path)
p.add_argument("--train-config", type=pathlib.Path, default=pathlib.Path("configs/sae/train.yaml"))
p.add_argument("--out", type=pathlib.Path, default=None)
a = p.parse_args()

grid = yaml.safe_load(a.grid.read_text())
defaults = yaml.safe_load(a.train_config.read_text())
base = {"seed": 42, **grid.get("base", {})}
reference = {**defaults, **base}


def fmt(v) -> str:
    return yaml.safe_dump(v, default_flow_style=True).strip().removesuffix("...").strip()


eval_set = " ".join(f"{k}={fmt(v)}" for k, v in grid.get("eval", {}).items())


def name_of(s: dict) -> str:
    parts = [s["encoder"], s.get("field", reference["field"]), f"x{s.get('expansion', reference['expansion'])}",
             f"k{s.get('top_k', reference['top_k'])}"]
    for k, v in sorted(s.items()):
        if k in SPECIAL or k in ("field", "expansion", "top_k") or v == reference.get(k):
            continue
        short = SHORT.get(k, k)
        if short is None:
            continue
        parts.append(f"{short}{'' if v is True else fmt(v)}")
    return "_".join(parts + [f"seed{s['seed']}"])


runs: dict[str, dict] = {}
for group, axes in grid["groups"].items():
    keys = list(axes)
    for values in itertools.product(*(axes[k] for k in keys)):
        s = {**base, **dict(zip(keys, values))}
        name = name_of(s)
        if name in runs:
            runs[name]["groups"].append(group)
            continue
        store = s.get("store", s["encoder"])
        train_set = " ".join(f"{k}={fmt(v)}" for k, v in s.items() if k not in SPECIAL)
        field = s.get("field")
        runs[name] = {"run_name": name, "groups": [group], "encoder": s["encoder"], "store": store, "seed": s["seed"],
                      "train_set": train_set, "eval_set": " ".join(filter(None, [eval_set, f"field={field}" if field else ""]))}

out = a.out or pathlib.Path("ablations") / f"{a.grid.stem}.tsv"
out.parent.mkdir(parents=True, exist_ok=True)
with open(out, "w", newline="") as fd:
    w = csv.DictWriter(fd, fieldnames=["run_name", "groups", "encoder", "store", "seed", "train_set", "eval_set"], delimiter="\t")
    w.writeheader()
    for r in runs.values():
        w.writerow({**r, "groups": ",".join(r["groups"])})
print(f"{len(runs)} runs -> {out}")
for i, r in enumerate(runs.values()):
    print(f"{i:3d}  {r['run_name']:<48} {','.join(r['groups'])}")
