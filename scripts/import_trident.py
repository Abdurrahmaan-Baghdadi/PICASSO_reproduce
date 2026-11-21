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

``--manifest`` lists the slides to import, one row per slide (CSV):

  ``slide_id``   the TRIDENT file name without ``.h5`` (or ``file_name``: the slide file, e.g. ``x.svs``)
  ``project``    the slide's label: stratifies the split and is what the comparison probes predict
  ``patient``    optional (or ``case``): all slides of a patient go to the same split

The default is the TCGA manifest (``case`` is the patient). ``--check`` only reports how many
TRIDENT slides the manifest matches, per project, and exits.

Several datasets import into one store: give every TRIDENT folder to ``--coords-dir`` and every
manifest to ``--manifest``. Slide ids must be unique across folders (TCGA barcodes and DICOM
UIDs are), and all folders must come from the same tiling (e.g. 20x, 256 px).

    python scripts/import_trident.py --coords-dir .../contours_hest_0.5_holes/20x_256px_0px_overlap \\
        --projects TCGA-GBM,TCGA-LGG --encoders virchow2,uni2 --data-root data_brain
"""
import argparse, concurrent.futures as cf, json, multiprocessing as mp, os, pathlib, sys, zlib

import h5py
import numpy as np
import polars as pl

TRIDENT_NAMES = {"virchow2": "virchow2", "uni2": "uni_v2"}   # our name -> TRIDENT folder suffix
CLS_DIM = {"virchow2": 1280, "uni2": 1536}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--coords-dir", required=True, type=pathlib.Path, nargs="+",
                   help="TRIDENT <mag>x_<px>px_<overlap>px_overlap folder(s)")
    p.add_argument("--data-root", required=True, type=pathlib.Path)
    p.add_argument("--manifest", default=["manifests/tcga_dx_all.csv"], nargs="+",
                   help="CSV(s) of the slides to import, with project and patient")
    p.add_argument("--projects", default=None, help="comma-separated projects to keep, e.g. TCGA-GBM,TCGA-LGG")
    p.add_argument("--encoders", default="virchow2,uni2")
    p.add_argument("--meanpatch", action="store_true", help="also write Virchow2 mean patch tokens")
    p.add_argument("--slides-dir", type=pathlib.Path, default=None, help="where the .svs files are (for stages 6-7)")
    p.add_argument("--max-slides-per-project", type=int, default=None)
    p.add_argument("--max-tiles-per-slide", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--force", action="store_true", help="re-import even if settings changed")
    p.add_argument("--check", action="store_true", help="report what the manifest matches, import nothing")
    return p.parse_args()


SLIDE_EXTS = (".svs", ".tif", ".tiff", ".ndpi", ".mrxs", ".scn", ".dcm")


def read_manifests(paths: list[str], projects: str | None) -> tuple[dict[str, str], dict[str, str]]:
    """Merged slide id -> project and slide id -> patient of several manifests."""
    project_of, patient_of = {}, {}
    for path in paths:
        proj, pat = read_manifest(path, projects)
        for sid in proj.keys() & project_of.keys():
            if proj[sid] != project_of[sid]:
                sys.exit(f"slide {sid} has project {project_of[sid]} in one manifest and {proj[sid]} in {path}")
        project_of |= proj
        patient_of |= pat
    return project_of, patient_of


def read_manifest(path: str, projects: str | None) -> tuple[dict[str, str], dict[str, str]]:
    """slide id -> project, and slide id -> patient (empty when the manifest has no patient column)."""
    man = pl.read_csv(path, infer_schema_length=0)   # every column as text
    if "slide_id" in man.columns:
        ids = man["slide_id"].to_list()
    elif "file_name" in man.columns:
        ids = [next((f.removesuffix(e) for e in SLIDE_EXTS if f.lower().endswith(e)), f) for f in man["file_name"]]
    else:
        sys.exit(f"{path} needs a slide_id (or file_name) column; it has {man.columns}")
    if "project" not in man.columns:
        sys.exit(f"{path} needs a project column (the slide's label); it has {man.columns}")
    pcol = next((c for c in ("patient", "case") if c in man.columns), None)
    keep = set(projects.split(",")) if projects else None
    project_of, patient_of = {}, {}
    for i, (sid, proj) in enumerate(zip(ids, man["project"])):
        if not sid or not proj or (keep and proj not in keep):
            continue
        if project_of.get(sid, proj) != proj:
            sys.exit(f"{path}: slide {sid} is listed with two projects ({project_of[sid]}, {proj})")
        project_of[sid] = proj
        if pcol and man[pcol][i]:
            patient_of[sid] = man[pcol][i]
    return project_of, patient_of


def features_dir(coords_dir: pathlib.Path, enc: str) -> pathlib.Path:
    return coords_dir / f"features_{TRIDENT_NAMES[enc]}"


def features_file(coords_dir: pathlib.Path, enc: str, sid: str) -> pathlib.Path:
    return features_dir(coords_dir, enc) / f"{sid}.h5"


def import_slide(sid: str, cdir: pathlib.Path, a, encoders: list[str], project: str, patient: str | None, svs: str) -> dict:
    coords = meta = None
    for enc in encoders:
        with h5py.File(features_file(cdir, enc, sid), "r") as f:
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
    info = {"slide_id": sid, "project": project, "patient": patient, "path": svs, "mag": float(meta["level0_magnification"]),
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
        with h5py.File(features_file(cdir, enc, sid), "r") as f:
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

    # Slides: in the manifest (gives the project), in the chosen projects, with features for every encoder.
    project_of, patient_of = read_manifests(a.manifest, a.projects)
    # Slide id -> TRIDENT folder, per encoder; a slide must come from one folder only.
    dir_of: dict[str, pathlib.Path] = {}
    found: dict[str, set[str]] = {enc: set() for enc in encoders}
    for cdir in a.coords_dir:
        for enc in encoders:
            stems = {f.stem for f in features_dir(cdir, enc).glob("*.h5")}
            if not stems:
                sys.exit(f"no .h5 files in {features_dir(cdir, enc)}: is it a TRIDENT "
                         "<mag>x_<px>px_<overlap>px_overlap folder?")
            for sid in stems:
                if dir_of.setdefault(sid, cdir) != cdir:
                    sys.exit(f"slide {sid} is in both {dir_of[sid]} and {cdir}")
            found[enc] |= stems
        n_match = len({f.stem for f in features_dir(cdir, encoders[0]).glob("*.h5")} & set(project_of))
        print(f"{cdir}: {n_match} slides in the manifest(s)")
        if not n_match:
            example = next(iter(sorted(f.stem for f in features_dir(cdir, encoders[0]).glob("*.h5"))))
            sys.exit(f"none of the slides in {cdir} is in {', '.join(a.manifest)}"
                     f"{' for ' + a.projects if a.projects else ''} (e.g. {example}). The manifest gives each slide "
                     "its project, so every dataset needs its slides listed in a manifest (--manifest).")
    have = {enc: found[enc] & set(project_of) for enc in encoders}
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
    if a.check:
        counts = {}
        for s in paired:
            counts[project_of[s]] = counts.get(project_of[s], 0) + 1
        n_pat = len({patient_of[s] for s in paired if s in patient_of})
        print(f"{len(paired)} slides matched in {len(counts)} projects; "
              + (f"{n_pat} patients" if n_pat else "no patient column: the split will be per slide"))
        for proj, n in sorted(counts.items(), key=lambda kv: -kv[1]):
            print(f"  {n:6d}  {proj}")
        return

    settings = {k: str(v) if isinstance(v, pathlib.Path) else v for k, v in vars(a).items() if k not in ("workers", "force", "check")}
    for k in ("coords_dir", "manifest"):   # one value is stored as before, several as a list
        v = [str(x) for x in settings[k]]
        settings[k] = v[0] if len(v) == 1 else v
    record = a.data_root / "trident_import.json"
    if record.exists() and not a.force:
        old = json.loads(record.read_text())
        changed = [k for k in ("coords_dir", "manifest", "projects", "max_slides_per_project", "max_tiles_per_slide", "seed")
                   if old.get(k) != settings[k]]
        if changed:
            sys.exit(f"{a.data_root} was imported with other settings ({changed}); use a new --data-root or --force")
    svs = {}
    if a.slides_dir:
        print(f"listing .svs files under {a.slides_dir}", flush=True)
        svs = {f.stem: str(f) for f in a.slides_dir.rglob("*.svs")}
    workers = max(1, min(a.workers, 32, len(chosen)))   # more processes only add Lustre contention
    print(f"importing {len(chosen)} slides with {', '.join(encoders)}, {workers} processes", flush=True)

    (a.data_root / "tiles").mkdir(parents=True, exist_ok=True)
    infos = []
    every = max(1, min(100, len(chosen) // 10))
    # spawn, not Linux's default fork: Polars' thread pool is already running here, and forked
    # children can deadlock on it (the job then hangs until its time limit). One Polars thread each.
    os.environ.setdefault("POLARS_MAX_THREADS", "1")
    with cf.ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn")) as pool:
        futs = [pool.submit(import_slide, sid, dir_of[sid], a, encoders, project_of[sid], patient_of.get(sid), svs.get(sid, "")) for sid in chosen]
        for k, fut in enumerate(futs):
            info = fut.result()
            infos.append(info)
            if "error" in info:
                print("SKIPPED", info["slide_id"], info["error"], flush=True)
            if (k + 1) % every == 0 or k + 1 == len(chosen):
                print(f"{k + 1}/{len(chosen)}", flush=True)
    ok = [i for i in infos if "error" not in i]
    if not ok:
        sys.exit(f"no slide imported: {len(infos)} skipped (reasons above)")
    pl.DataFrame(ok).write_parquet(a.data_root / "tiles" / "slides_summary.parquet")
    n_tiles = int(sum(i["n_tiles"] for i in ok))
    record.write_text(json.dumps(settings | {"n_slides": len(ok), "n_tiles": n_tiles}, indent=1) + "\n")
    print(f"imported {len(ok)} slides, {n_tiles:,} tiles, {len(infos) - len(ok)} skipped -> {a.data_root}")


if __name__ == "__main__":
    main()
