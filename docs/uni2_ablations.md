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

Set up once, on a login node:

```bash
cd $WORK && git clone https://github.com/Pediatric-Brain-Tumor/PICASSO_reproduce.git && cd PICASSO_reproduce
git checkout uni2-ablations
# Miniconda under $WORK; on Vista (ARM) build the env on Vista itself
conda create -n picasso python=3.12 -y && conda activate picasso && pip install -e .
python scripts/make_ablations.py configs/ablations/sae_brain.yaml
```

Then submit the three jobs from the repo root, each after the one before it:

```bash
export DATA_ROOT=$SCRATCH/picasso/data_brain OUTPUT_ROOT=$WORK/picasso/outputs_brain
export TRIDENT_DIR=/scratch/11738/yugdave578/data/wsi_processed/tcga_gbm_lgg/contours_hest_0.5_holes/20x_256px_0px_overlap
export SLIDES_DIR=/scratch/11738/cmssmc/pbt_picasso/tcga/slides
export ALLOCATION=<your TACC project>
bash slurm/submit.sh data                                        # prints the job id
DEPEND=<data job id> bash slurm/submit.sh sae ablations/sae_brain.tsv
DEPEND=<sae array job id> bash slurm/submit.sh compare ablations/sae_brain.tsv
```

- `SYSTEM=ls6` (default) uses `normal` and `gpu-a100`, three runs per node (one per A100).
  `SYSTEM=vista` uses `gg` and `gh`, one run per node. `CPU_PART` / `GPU_PART` override them.
- The jobs inherit the exported variables; `CONDA_SH` and `CONDA_ENV` point at the env if it
  is not `$WORK/miniconda3` / `picasso`.
- Reruns skip finished runs, so a timed-out array can simply be resubmitted.
- `$SCRATCH` is purged when files go unused, so keep `OUTPUT_ROOT` on `$WORK`. Rebuilding
  `DATA_ROOT` is one `data` job.
- Each run loads its train split into RAM (`in_memory: true`): about 25 GB for Virchow2 and
  30 GB for UNI2, so three runs fit on one 256 GB Lonestar6 node.

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
| `probes.csv` | GBM vs LGG logistic probes, raw embeddings vs SAE codes (tile and slide AUC) |
