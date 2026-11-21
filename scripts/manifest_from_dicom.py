"""Manifest for a TRIDENT folder of DICOM slides (e.g. CCDI MCI), for scripts/import_trident.py.

Slides named by DICOM UIDs carry no patient in their name, so this reads it from the DICOM
headers. TRIDENT records the file it read for every slide in ``<job dir>/wsi_states/*.json``
(the job dir is the folder above ``--coords-dir``); only that file's header is read, never the
pixels. Writes one row per ``.h5`` slide:

  ``slide_id``  the TRIDENT name           ``project``  ``--project``, or the label from ``--labels``
  ``patient``   DICOM PatientID            ``dicom_path``, ``study_uid``, ``series_uid``  for reference

``--labels`` is an optional CSV giving each patient (column ``patient``) or slide (``slide_id``)
a ``project``, e.g. the diagnosis; slides it does not cover keep ``--project``. If the DICOM
files moved since TRIDENT ran, ``--dicom-dir`` is searched for them by file name.

    pip install pydicom
    python scripts/manifest_from_dicom.py --coords-dir .../ccdi_mci/wsi_processed/contours_hest_0.5/20x_256px_0px_overlap \\
        --out manifests/ccdi_mci.csv
"""
import argparse, collections, concurrent.futures as cf, csv, json, os, pathlib, sys


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--coords-dir", required=True, type=pathlib.Path, help="TRIDENT <mag>x_<px>px_<overlap>px_overlap folder")
    p.add_argument("--out", required=True, type=pathlib.Path)
    p.add_argument("--project", default="CCDI-MCI", help="label of every slide not in --labels")
    p.add_argument("--labels", type=pathlib.Path, default=None, help="CSV: patient or slide_id, project")
    p.add_argument("--wsi-states", type=pathlib.Path, default=None, help="default: <coords-dir>/../wsi_states")
    p.add_argument("--dicom-dir", type=pathlib.Path, default=None, help="where to look if a recorded file is gone")
    p.add_argument("--workers", type=int, default=16)
    return p.parse_args()


def recorded_paths(states_dir: pathlib.Path) -> dict[str, str]:
    """TRIDENT slide name -> the file it read (latest state per slide)."""
    paths, when = {}, {}
    for f in states_dir.glob("*.json"):
        try:
            st = json.loads(f.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        slide = st.get("slide", {})
        name, path = slide.get("name"), slide.get("abs_path")
        t = st.get("resume", {}).get("last_updated_at", "")
        if name and path and t >= when.get(name, ""):
            paths[name], when[name] = path, t
    return paths


def read_header(path: str) -> dict:
    import pydicom

    tags = ["PatientID", "StudyInstanceUID", "SeriesInstanceUID"]
    if os.path.isdir(path):   # a reader that opens a series folder: any file of it has the patient
        files = sorted(f for f in os.listdir(path) if not f.startswith("."))
        if not files:
            return {"error": "empty folder"}
        path = os.path.join(path, files[0])
    try:
        ds = pydicom.dcmread(path, stop_before_pixels=True, specific_tags=tags)
    except Exception as e:   # unreadable or not DICOM
        return {"error": f"{type(e).__name__}: {e}"}
    return {"patient": str(ds.get("PatientID", "") or ""), "study_uid": str(ds.get("StudyInstanceUID", "") or ""),
            "series_uid": str(ds.get("SeriesInstanceUID", "") or "")}


def main() -> None:
    a = parse_args()
    try:
        import pydicom  # noqa: F401
    except ImportError:
        sys.exit("needs pydicom: pip install pydicom")

    slides = sorted({f.stem for d in a.coords_dir.glob("features_*") for f in d.glob("*.h5")})
    if not slides:
        sys.exit(f"no features_*/*.h5 in {a.coords_dir}")
    states = a.wsi_states or a.coords_dir.parent / "wsi_states"
    recorded = recorded_paths(states) if states.is_dir() else {}
    print(f"{len(slides)} slides; {len(recorded)} recorded in {states}")

    # The DICOM file of every slide: as recorded, else found by name under --dicom-dir.
    path_of = {s: recorded[s] for s in slides if s in recorded and os.path.exists(recorded[s])}
    missing = [s for s in slides if s not in path_of]
    if missing and a.dicom_dir:
        wanted = {pathlib.Path(recorded[s]).name if s in recorded else None: s for s in missing}
        for root, _, files in os.walk(a.dicom_dir):
            for fn in files:
                stem = fn.rsplit(".dcm", 1)[0] if fn.lower().endswith(".dcm") else fn
                s = wanted.get(fn) or (stem if stem in missing else None)
                if s and s not in path_of:
                    path_of[s] = os.path.join(root, fn)
        missing = [s for s in slides if s not in path_of]
    if missing:
        hint = "" if a.dicom_dir else " (pass --dicom-dir if the DICOM files moved)"
        print(f"WARNING {len(missing)} slides without a DICOM file{hint}, e.g. {missing[0]}: their split is per slide")

    with cf.ThreadPoolExecutor(a.workers) as pool:
        headers = dict(zip(path_of, pool.map(read_header, path_of.values())))
    errors = {s: h["error"] for s, h in headers.items() if "error" in h}
    for s, e in list(errors.items())[:5]:
        print(f"WARNING cannot read {path_of[s]}: {e}")

    label_of_patient, label_of_slide = {}, {}
    if a.labels:
        with open(a.labels, newline="") as fd:
            for r in csv.DictReader(fd):
                if r.get("project") and r.get("slide_id"):
                    label_of_slide[r["slide_id"]] = r["project"]
                elif r.get("project") and r.get("patient"):
                    label_of_patient[r["patient"]] = r["project"]

    rows = []
    for s in slides:
        h = headers.get(s, {})
        patient = h.get("patient", "")
        rows.append({"slide_id": s, "project": label_of_slide.get(s) or label_of_patient.get(patient) or a.project,
                     "patient": patient, "dicom_path": path_of.get(s, ""),
                     "study_uid": h.get("study_uid", ""), "series_uid": h.get("series_uid", "")})
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", newline="") as fd:
        w = csv.DictWriter(fd, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    per_patient = collections.Counter(r["patient"] for r in rows if r["patient"])
    no_patient = sum(not r["patient"] for r in rows)
    labels = collections.Counter(r["project"] for r in rows)
    print(f"{len(rows)} slides, {len(per_patient)} patients "
          f"({sum(n > 1 for n in per_patient.values())} with several slides, up to {max(per_patient.values(), default=0)}); "
          f"{no_patient} without a patient; {len(errors)} unreadable")
    print("labels: " + ", ".join(f"{k} {v}" for k, v in labels.most_common(10)) + (" ..." if len(labels) > 10 else ""))
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
