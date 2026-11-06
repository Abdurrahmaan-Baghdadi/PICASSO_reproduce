"""Import TRIDENT tile coordinates and features as the outputs of stages 2-3 (tiling, embedding).

TRIDENT (mahmoodlab/TRIDENT) writes one HDF5 file per slide and encoder,
``<coords_dir>/features_<encoder>/<slide_id>.h5``, holding ``coords`` [n, 2] (level-0 pixels)
and ``features`` [n, d] float32. This script writes, under ``--data-root``:

  ``tiles/<slide_id>/tiles.parquet``, ``tiles/slides_summary.parquet``   (as stage 2)
  ``embeddings/<encoder>/<slide_id>.cls.npy`` [n, d] fp16, ``.done``      (as stage 3)

so stage 4 (``pipeline/04_build_store.sh``) and everything after it run unchanged.
TRIDENT's Virchow2 features are [CLS, mean patch token] (2560); the CLS half is the PICASSO
input and ``--meanpatch`` also writes the second half as ``<slide_id>.meanpatch.npy``. UNI2
features are the CLS token (1536).

Only slides with features for every requested encoder and identical coordinates are
imported, so the stores of all encoders hold the same tiles and get the same split.
``--max-slides-per-project`` and ``--max-tiles-per-slide`` make a small copy for local tests.

    python scripts/import_trident.py --coords-dir .../contours_hest_0.5_holes/20x_256px_0px_overlap \\
        --projects TCGA-GBM,TCGA-LGG --encoders virchow2,uni2 --data-root data_brain
"""
import argparse, concurrent.futures as cf, json, pathlib, sys, zlib

import h5py
import numpy as np
import polars as pl

TRIDENT_NAMES = {"virchow2": "virchow2", "uni2": "uni_v2"}   # our name -> TRIDENT folder suffix
CLS_DIM = {"virchow2": 1280, "uni2": 1536}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--coords-dir", required=True, type=pathlib.Path, help="TRIDENT <mag>x_<px>px_<overlap>px_overlap folder")
    p.add_argument("--data-root", required=True, type=pathlib.Path)
    p.add_argument("--manifest", default="manifests/tcga_dx_all.csv", help="gives each slide its TCGA project")
    p.add_argument("--projects", default=None, help="comma-separated TCGA project ids, e.g. TCGA-GBM,TCGA-LGG")
    p.add_argument("--encoders", default="virchow2,uni2")
    p.add_argument("--meanpatch", action="store_true", help="also write Virchow2 mean patch tokens")
    p.add_argument("--slides-dir", type=pathlib.Path, default=None, help="where the .svs files are (for stages 6-7)")
    p.add_argument("--max-slides-per-project", type=int, default=None)
    p.add_argument("--max-tiles-per-slide", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--force", action="store_true", help="re-import even if settings changed")
    return p.parse_args()


def features_file(a, enc: str, sid: str) -> pathlib.Path:
    return a.coords_dir / f"features_{TRIDENT_NAMES[enc]}" / f"{sid}.h5"


def import_slide(sid: str, a, encoders: list[str], project: str, svs: str) -> dict:
    coords = meta = None
    for enc in encoders:
        with h5py.File(features_file(a, enc, sid), "r") as f:
            if f["features"].shape[1] not in (CLS_DIM[enc], 2 * CLS_DIM[enc]):
                return {"slide_id": sid, "error": f"{enc} width {f['features'].shape[1]}"}
            if coords is None:
                coords, meta = f["coords"][:], dict(f["coords"].attrs)
            elif not np.array_equal(f["coords"][:], coords):
                return {"slide_id": sid, "error": f"{enc} coordinates differ from {encoders[0]}"}
    keep = np.arange(len(coords))
    if a.max_tiles_per_slide and len(coords) > a.max_tiles_per_slide:
        rng = np.random.default_rng(zlib.crc32(sid.encode()) ^ a.seed)
        keep = np.sort(rng.choice(len(coords), a.max_tiles_per_slide, replace=False))
    tile_l0 = int(round(float(meta["patch_size_level0"])))
    info = {"slide_id": sid, "project": project, "path": svs, "mag": float(meta["level0_magnification"]),
            "dims": [int(meta["level0_width"]), int(meta["level0_height"])], "tile_l0": tile_l0, "n_tiles": len(keep)}

    tiles_out = a.data_root / "tiles" / sid
    done = [a.data_root / "embeddings" / enc / f"{sid}.done" for enc in encoders]
    if all(d.exists() for d in done) and (tiles_out / "tiles.parquet").exists():
        return info
    tiles_out.mkdir(parents=True, exist_ok=True)
    pl.DataFrame({"slide_id": [sid] * len(keep), "x": coords[keep, 0].astype(np.int32), "y": coords[keep, 1].astype(np.int32),
                  "tile_l0": np.full(len(keep), tile_l0, np.int32), "tissue_frac": np.full(len(keep), np.nan, np.float32)}
                 ).write_parquet(tiles_out / "tiles.parquet")
    for enc, marker in zip(encoders, done):
        marker.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(features_file(a, enc, sid), "r") as f:
            x = f["features"][:][keep]
        d = CLS_DIM[enc]
        np.save(marker.parent / f"{sid}.cls.npy", x[:, :d].astype(np.float16))
        if a.meanpatch and enc == "virchow2" and x.shape[1] == 2 * d:
            np.save(marker.parent / f"{sid}.meanpatch.npy", x[:, d:].astype(np.float16))
        marker.write_text("ok\n")
    return info


def main() -> None:
    a = parse_args()
    encoders = a.encoders.split(",")
    for enc in encoders:
        if enc not in TRIDENT_NAMES:
            sys.exit(f"unknown encoder {enc}; choose from {sorted(TRIDENT_NAMES)}")

    settings = {k: str(v) if isinstance(v, pathlib.Path) else v for k, v in vars(a).items() if k not in ("workers", "force")}
    record = a.data_root / "trident_import.json"
    if record.exists() and not a.force:
        old = json.loads(record.read_text())
        changed = [k for k in ("coords_dir", "projects", "max_slides_per_project", "max_tiles_per_slide", "seed") if old.get(k) != settings[k]]
        if changed:
            sys.exit(f"{a.data_root} was imported with other settings ({changed}); use a new --data-root or --force")

    # Slides: in the manifest (gives the project), in the chosen projects, with features for every encoder.
    man = pl.read_csv(a.manifest)
    if a.projects:
        man = man.filter(pl.col("project").is_in(a.projects.split(",")))
    project_of = {f.removesuffix(".svs"): proj for f, proj in zip(man["file_name"], man["project"])}
    have = {enc: {f.stem for f in features_file(a, enc, "x").parent.glob("*.h5")} & set(project_of) for enc in encoders}
    paired = set(project_of).intersection(*have.values())
    for enc in encoders:
        print(f"{enc}: {len(have[enc])} slides with features, {len(have[enc] - paired)} dropped (missing for another encoder)")
    chosen = []
    rng = np.random.default_rng(a.seed)
    for proj in sorted({project_of[s] for s in paired}):
        ids = sorted(s for s in paired if project_of[s] == proj)
        if a.max_slides_per_project:
            ids = sorted(rng.choice(ids, size=min(a.max_slides_per_project, len(ids)), replace=False).tolist())
        chosen += ids
    print(f"importing {len(chosen)} slides with {', '.join(encoders)}")
    svs = {f.stem: str(f) for f in a.slides_dir.rglob("*.svs")} if a.slides_dir else {}

    (a.data_root / "tiles").mkdir(parents=True, exist_ok=True)
    infos = []
    with cf.ProcessPoolExecutor(a.workers) as pool:
        futs = [pool.submit(import_slide, sid, a, encoders, project_of[sid], svs.get(sid, "")) for sid in chosen]
        for k, fut in enumerate(futs):
            info = fut.result()
            infos.append(info)
            if "error" in info:
                print("SKIPPED", info["slide_id"], info["error"])
            if (k + 1) % 100 == 0:
                print(f"{k + 1}/{len(chosen)}", flush=True)
    ok = [i for i in infos if "error" not in i]
    pl.DataFrame(ok).write_parquet(a.data_root / "tiles" / "slides_summary.parquet")
    n_tiles = int(sum(i["n_tiles"] for i in ok))
    record.write_text(json.dumps(settings | {"n_slides": len(ok), "n_tiles": n_tiles}, indent=1) + "\n")
    print(f"imported {len(ok)} slides, {n_tiles:,} tiles, {len(infos) - len(ok)} skipped -> {a.data_root}")


if __name__ == "__main__":
    main()
