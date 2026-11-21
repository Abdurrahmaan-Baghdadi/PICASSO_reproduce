# Virchow2 vs UNI2 SAEs on brain tumours

The PICASSO SAE takes Virchow2 CLS embeddings. This workflow trains the same SAE on UNI2
embeddings of the same tiles and compares the two, with ablations around the paper recipe.
The SAE stages (4-5) run on either encoder. Emb2Img (stages 6-7) still runs on Virchow2 only:
its pairs, model width and closed-loop score assume Virchow2.

The embeddings come from TRIDENT (mahmoodlab/TRIDENT), which already tiled and encoded the
TCGA-GBM and TCGA-LGG slides with both encoders. `scripts/import_trident.py` turns its HDF5
files into the outputs of stages 2-3, so no slide is read and no GPU is needed until stage 5.

## How the comparison stays fair

- **Same tiles, same split.** Only slides with features for both encoders and identical tile
  coordinates are imported, so both stores hold the same tiles in the same order and get the
  same 80/10/10 split. The split is by patient: 307 of the 880 TCGA-GBM/LGG patients have
  several slides, and a per-slide split put 92 patients in both train and test.
- **Seeds.** The `main` group trains three seeds per encoder. A difference between encoders
  counts only if it is larger than the spread between seeds of one encoder.
- **Scale-free metrics.** The encoders differ in width (1280 vs 1536) and scale, so compare
  explained variance, cosine, dead and active concepts, not raw MSE. The `normalize` group
  z-scores both inputs.
- **Concepts.** `scripts/compare_saes.py` correlates the concept activations of two SAEs on the
  same test tiles. Seed pairs give the baseline; encoder pairs show which concepts are shared.

## Differences from stages 2-3 of this repo

| | Stages 2-3 | TRIDENT import |
|---|---|---|
| Tissue segmentation | Saturation threshold with Otsu | HEST segmenter |
| Tile size | 256 px at 0.5 µm/px, from the slide's mpp | 256 px at 20×, from the magnification (512 level-0 px on 40× slides) |
| Virchow2 | CLS, computed here | First half of TRIDENT's [CLS, mean patch] (2560) |
| UNI2 | not available | TRIDENT's UNI2-h CLS (1536) |

## Local test (laptop, minutes)

```bash
conda create -n picasso python=3.12 -y && conda activate picasso && pip install -e .
export DATA_ROOT=$PWD/data_local OUTPUT_ROOT=$PWD/outputs_local
python scripts/import_trident.py --coords-dir <TRIDENT 20x_256px_0px_overlap folder> \
    --projects TCGA-GBM,TCGA-LGG --encoders virchow2,uni2 --meanpatch --data-root $DATA_ROOT \
    --max-slides-per-project 30 --max-tiles-per-slide 2000
ENCODER=virchow2 bash pipeline/04_build_store.sh --set 'fields=[cls]'
ENCODER=uni2 bash pipeline/04_build_store.sh --set 'fields=[cls]'
ENCODER=virchow2 STORE_DIR=$DATA_ROOT/store/virchow2_meanpatch bash pipeline/04_build_store.sh --set 'fields=[meanpatch]'
python scripts/make_ablations.py configs/ablations/smoke.yaml
bash scripts/run_ablation.sh ablations/smoke.tsv all
python scripts/compare_saes.py ablations/smoke.tsv --device mps --n-rows 10000 --probe-train-rows 20000
```

The smoke grid uses `device: mps` (Apple GPU). Use `cpu` on other machines. The metrics
code keeps its float64 sums on the CPU because MPS has no float64.

## TACC

Everything goes through `slurm/submit.sh`, which reads your settings from `tacc.env`.
On a login node, once:

```bash
cd $WORK && git clone https://github.com/Abdurrahmaan-Baghdadi/PICASSO_reproduce.git && cd PICASSO_reproduce && git checkout uni2-ablations
cp tacc.env.example tacc.env && nano tacc.env    # allocation, system, TRIDENT folder, output folders
bash slurm/submit.sh setup                       # builds the conda env and checks everything below
```

`setup` creates the conda env if it is missing, installs the repo, and checks that torch has
CUDA, that both TRIDENT feature folders are readable, that the queues exist and that the
allocation is accepted. It ends with "All set", or lists what to fix. Then:

```bash
bash slurm/submit.sh test    # dev queues, about an hour: 60 slides, 9 tiny SAEs and the comparison
bash slurm/submit.sh all     # the real run: data -> 21 SAEs -> comparison, chained
```

Both print their job ids and where the results go; follow them with `squeue -u $USER` and
the logs in `slurm_logs/`. `test` writes to `<DATA_ROOT>_test` and `<OUTPUT_ROOT>_test`, so the
real run starts clean. It passed when `<OUTPUT_ROOT>_test/compare/smoke_tacc/` holds
`metrics.csv`, `matching.csv` and `probes.csv`.

- `SYSTEM=ls6` (default) uses `normal` and `gpu-a100`, three runs per node (one per A100), and
  `development` / `gpu-a100-dev` for `test`. `SYSTEM=vista` uses `gg` and `gh`, one run per
  node, and `gh-dev` for `test`. `CPU_PART`, `GPU_PART`, `TEST_CPU_PART` and `TEST_GPU_PART` in
  `tacc.env` override them.
- If an SAE run fails, its job fails and the comparison waits forever
  (`DependencyNeverSatisfied` in `squeue`): `scancel` it, read `<OUTPUT_ROOT>/logs/sae_<run>.log`,
  fix, and rerun `bash slurm/submit.sh all`. Finished slides and SAEs are skipped, so reruns
  also resume timed-out jobs.
- Single stages: `bash slurm/submit.sh data`, `... sae <table.tsv>`, `... compare <table.tsv>`,
  with `DEPEND=<job id>` to wait for another job. Extra arguments go to `sbatch`, e.g. `-t 02:00:00`.
- After editing `configs/ablations/sae_brain.yaml`, rerun
  `python scripts/make_ablations.py configs/ablations/sae_brain.yaml` before `all`.
- `$SCRATCH` is purged when files go unused, so keep `OUTPUT_ROOT` on `$WORK`. Rebuilding
  `DATA_ROOT` is one `data` job.
- Each run loads its train split into RAM (`in_memory: true`): about 25 GB for Virchow2 and
  30 GB for UNI2, so three runs fit on one 256 GB Lonestar6 node.

## Other datasets, and several at once

Any TRIDENT output imports the same way once its slides are listed in a manifest: a CSV with
one row per slide and the columns

| Column | Content |
|---|---|
| `slide_id` | The TRIDENT file name without `.h5` (or `file_name`, the slide file, e.g. `x.svs`) |
| `project` | The slide's label, e.g. diagnosis. It stratifies the split and is what `probes.csv` predicts |
| `patient` | Optional (or `case`). All slides of a patient go to the same split; without it the split is per slide |

Other columns are ignored. The TCGA manifest (`manifests/tcga_dx_all.csv`) is one already.
For DICOM slides named by UIDs (CCDI MCI), `scripts/manifest_from_dicom.py` builds one: it finds
the file TRIDENT read for each slide (`wsi_states/`) and takes the patient from its DICOM header.
Every slide gets the label `CCDI-MCI` unless `--labels` (a CSV of patient or slide_id, project)
gives it one.

To train on several datasets at once, list their TRIDENT folders and manifests in `tacc.env`,
separated by `:`. They import into one store with one patient-level split, stratified by project:

```bash
TRIDENT_DIR=/path/tcga_gbm_lgg/.../20x_256px_0px_overlap:/path/ccdi_mci/.../20x_256px_0px_overlap
MANIFEST=manifests/tcga_dx_all.csv:/path/ccdi_manifest.csv
PROJECTS=
```

`bash slurm/submit.sh setup` reports how many slides each folder matches and the slides per
project. Things to keep in mind:

- Slide ids must be unique across folders, and all folders should come from the same TRIDENT
  tiling (20x, 256 px) and encoders.
- `PROJECTS` filters every manifest. Leave it empty to keep everything a manifest lists and
  TRIDENT has features for (the TCGA folder only holds GBM and LGG).
- With mixed datasets, the probes partly learn which dataset a tile came from, because the
  projects of different datasets never overlap. Compare encoders with that in mind.
- A larger train split needs more RAM (`in_memory: true` loads it): roughly 2.6 KB per tile for
  Virchow2 and 3 KB for UNI2. Three UNI2 runs on one 256 GB Lonestar6 node fit up to about
  25M train tiles; set `PER_NODE=1` (or `in_memory: false` in the grid) above that.

## The grid (`configs/ablations/sae_brain.yaml`, 21 runs)

| Group | Varies | Question |
|---|---|---|
| main | encoder × seed 42, 43, 44 | Paper recipe; seed spread |
| topk | k = 16, 64 | Concepts per tile |
| expansion | 8×, 16× | Dictionary size |
| normalize | z-scored input | Does the scale gap between encoders matter? |
| matryoshka | 10 nested prefixes | Coarse-to-fine concepts |
| epochs | 5 | The paper's epoch count on a small split |
| token | Virchow2 mean patch token | CLS vs patch tokens |

Each other group changes one setting of the `main` seed-42 run. Add a group to the YAML and
rerun `make_ablations.py`; finished runs are kept.

## Outputs

`$OUTPUT_ROOT/sae/<run>/` holds `sae.pt`, `log.jsonl` (train loss, per-epoch validation) and
`eval/summary.json` (test metrics, PCA and k-means baselines). `compare_saes.py` writes to
`$OUTPUT_ROOT/compare/<table>/`:

| File | Content |
|---|---|
| `metrics.csv` | Test metrics of every run |
| `main_summary.csv` | Mean and std over seeds, per encoder |
| `matching.csv` | Best-match correlation of concepts, for seed pairs and encoder pairs |
| `matches_<A>__<B>.parquet` | Best match of every concept, to look at shared and encoder-only concepts |
| `probes.csv` | Logistic probes predicting the manifest's `project` (GBM vs LGG for TCGA), raw embeddings vs SAE codes (tile and slide AUC; mean one-vs-rest AUC with more than two projects) |
