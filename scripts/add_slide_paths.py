"""Fill in the slide file of each imported slide, for Emb2Img (stages 6-7), without re-importing.

import_trident.py records a slide's file only when it was given --slides-dir. This adds it
afterwards to <data-root>/tiles/slides_summary.parquet (the original is kept as
slides_summary.before_paths.parquet); stores, splits and SAEs are untouched.

  --slides-dir   folder(s) searched recursively for <slide_id>.svs/.tif/.ndpi/... (e.g. TCGA)
  --manifest     manifest(s) with a dicom_path column (scripts/manifest_from_dicom.py, e.g. CCDI):
                 that file opens the DICOM slide (OpenSlide >= 4)

    python scripts/add_slide_paths.py --data-root $DATA_ROOT --slides-dir /path/tcga/slides \\
        --manifest /path/ccdi_manifest.csv
"""
import argparse, os, pathlib, shutil, sys

os.environ.setdefault("POLARS_MAX_THREADS", "2")   # login nodes cap threads per user
import polars as pl

EXTS = (".svs", ".tif", ".tiff", ".ndpi", ".mrxs", ".scn")

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--data-root", required=True, type=pathlib.Path)
p.add_argument("--slides-dir", type=pathlib.Path, nargs="*", default=[])
p.add_argument("--manifest", nargs="*", default=[])
p.add_argument("--dry-run", action="store_true", help="report the matches, write nothing")
a = p.parse_args()

summary_path = a.data_root / "tiles" / "slides_summary.parquet"
summary = pl.read_parquet(summary_path)
ids = set(summary["slide_id"])
found: dict[str, str] = {}
for d in a.slides_dir:
    print(f"listing slide files under {d}", flush=True)
    n = 0
    for root, _, files in os.walk(d):
        for f in files:
            stem, ext = os.path.splitext(f)
            if ext.lower() in EXTS and stem in ids and stem not in found:
                found[stem] = os.path.join(root, f); n += 1
    print(f"  {n} slides matched")
for m in a.manifest:
    man = pl.read_csv(m, infer_schema_length=0)
    if "dicom_path" not in man.columns:
        sys.exit(f"{m} has no dicom_path column (columns: {man.columns})")
    n = 0
    for sid, path in zip(man["slide_id"], man["dicom_path"]):
        if sid in ids and sid not in found and path and os.path.exists(path):
            found[sid] = path; n += 1
    print(f"{m}: {n} slides matched to a DICOM file")

old = dict(zip(summary["slide_id"], summary["path"]))
summary = summary.with_columns(pl.Series("path", [found.get(s) or old[s] or "" for s in summary["slide_id"]]))
report = (summary.group_by("project").agg(with_file=(pl.col("path") != "").sum(), slides=pl.len()).sort("project"))
print(report)
missing = int((summary["path"] == "").sum())
print(f"{summary.height - missing} of {summary.height} slides have a file; {missing} will be left out of Emb2Img")
if a.dry_run:
    sys.exit(0)
backup = summary_path.with_name("slides_summary.before_paths.parquet")
if not backup.exists():
    shutil.copy2(summary_path, backup)
summary.write_parquet(summary_path)
print(f"wrote {summary_path} (original: {backup.name})")
