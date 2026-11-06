# Handoff from chat → Claude Code: PICASSO_reproduce, UNI2 ablations, and the research roadmap

Purpose: carry the goals, decisions and ideas from a long planning chat into the repo, so Claude Code can continue without that chat. Read this first, then reconcile it with the actual repo (it was written from a summary of the repo, not from reading the code).

Suggested location: `docs/HANDOFF_FROM_CHAT.md`, with a one-line pointer to it in `CLAUDE.md`.

## 0. Checked against the code (2026-11-03)

The rest of this document is the handoff as written. Where it disagrees with the code, the code
wins:

- **Splits were slide-level, now patient-level.** `picasso.store` split slides independently. In the
  full TRIDENT set 307 of 880 patients have several slides, and the per-slide split put 92
  patients in both train and test. `split_by: patient` (new default in `configs/data/store.yaml`)
  keeps each patient's slides in one split; `split_by: slide` reproduces the old split exactly.
- **The Mac has a GPU** (Apple M5 Pro, MPS, 48 GB). Local runs use `device: mps`; the limit is
  data size and time, not the lack of a GPU.
- **`scripts/demo.py` and `scripts/concept_intervention.py` do not check `ENCODER`.** They always
  build Virchow2, so a UNI2 SAE would fail on a width mismatch (1536 vs 1280), not with a clear
  message. Only `pipeline/03`, `06` and `07` stop with a clear error for `ENCODER=uni2`.
- **Smoke numbers after the patient split** (re-run 2026-11-03, 9 runs in 1:55 on MPS; the same
  grid gives identical numbers when re-run): SAE explained variance Virchow2 0.27, UNI2 0.22;
  PCA-30 0.58 vs 0.30; seed-vs-seed matching 0.37 (Virchow2) vs 0.25 (UNI2), Virchow2-vs-UNI2
  0.30; tile-level GBM/LGG probe AUC on raw embeddings 0.91 vs 0.79. The Virchow2 mean-patch run
  has negative explained variance (−0.93 → −0.07 over 2 epochs, 91% dead concepts): mean-patch
  tokens vary little around a large mean, so it is undertrained here; watch it on TACC.
- `CLAUDE.md` and `EXPERIMENTS.md` did not exist; `CLAUDE.md` now points here.

## 1. First actions for Claude Code

1. Read `README`, `docs/train_on_subset.md`, `docs/uni2_ablations.md`, `configs/ablations/sae_brain.yaml`, `scripts/import_trident.py`, `scripts/compare_saes.py`, `slurm/`.
2. Run `git status` / `git diff` on branch `uni2-ablations`. Nothing has been committed yet. Propose a commit plan (logical commits: encoder switch, importer, MPS fixes, ablation grid + runner, compare script, SLURM, docs) and ask before committing.
3. Re-run the local smoke test (9 runs) to confirm the environment still works.
4. Report any place where this document disagrees with the code; the code wins.

## 2. Goals

- **Primary (now):** reproduce PICASSO (sparse-autoencoder concept atlas over pathology foundation-model embeddings, plus Emb2Img and concept intervention) on **UNI2** as well as Virchow2, and characterize how the results differ.
- **Workflow:** prepare everything locally (Mac, MPS, small samples), with ablations and alternative designs, then run at scale on UT TACC.
- **Longer term:** use the SAE as the interpretability and steering layer for a spatial generative model over tile-embedding grids (sections 6–7), and eventually apply the pipeline to pediatric brain tumor data.

## 3. Current state (as of the last Code session)

Repo: `~/Documents/Pediatric Brain Tumor Research/PICASSO_reproduce`, branch `uni2-ablations`, uncommitted. Local conda env `picasso` (torch with MPS).

Data: `~/Documents/Pediatric Brain Tumor Research/localDataStore/tcga_gbm_lgg/wsi_processed/contours_hest_0.5_holes/` — TRIDENT 0.3.2 output for TCGA-GBM/LGG (adult brain), no raw slides.

- Patches: 1,702 of 1,704 slides; 20×, 256 px, no overlap, min tissue 0.5; coords in level-0 px (512 px tiles on 40× slides).
- Virchow2: 1,481 slides, `[n, 2560]` = `[CLS (:1280), mean patch token (1280:)]`. PICASSO uses CLS.
- UNI2: 1,702 slides, `[n, 1536]`.
- ~12.1M Virchow2 tiles estimated. 221 slides lack Virchow2 features (interrupted run; lock files remain).
- Datastore `README`, `manifest/`, `clinical/`, `path_reports/` are empty.
- Raw slides on TACC: `/scratch/11738/cmssmc/pbt_picasso/tcga/slides`. TRIDENT outputs on TACC live in another user's `/scratch`; read access must be confirmed.

Done locally:

- `ENCODER=virchow2|uni2` switch for SAE stages; SAE width read from data; `aux_k` defaults to half the embedding width; MPS-safe metrics.
- `scripts/import_trident.py`: TRIDENT h5 → `tiles/<sid>/tiles.parquet`, `slides_summary.parquet`, `embeddings/<enc>/<sid>.*.npy` (fp16) + `.done`. Keeps only slides with both encoders and identical coords, so stores are paired. Subsampling flags for local tests.
- `configs/ablations/sae_brain.yaml` (21 TACC runs, 9 local): paper recipe × 3 seeds per encoder; k = 16, 64; expansion 8×, 16×; z-scored input; Matryoshka; 5 vs 20 epochs; Virchow2 mean-patch instead of CLS. Runner skips finished runs. Per-epoch checkpoints off (would be ~250 GB).
- `scripts/compare_saes.py`: test metrics with seed mean/spread; concept matching seed-vs-seed (baseline) and Virchow2-vs-UNI2 on the same test tiles (self-match sanity check = 1.0); GBM-vs-LGG probes on raw embeddings and SAE codes.
- `slurm/submit.sh data | sae | compare`, Lonestar6 default (3 runs per A100 node), Vista option. Syntax-checked only, never run on TACC.

Not done:

- Emb2Img (stages 3, 6, 7), `demo.py`, `concept_intervention.py` are Virchow2-only and need raw slides; they error clearly for `ENCODER=uni2`. UNI2 support needs an encoder switch in `pairs.py`, model width, and the closed-loop score.
- TRIDENT vs repo preprocessing differences are unverified (HEST segmenter vs Otsu-on-saturation; tile size by magnification vs µm/px; whether TRIDENT's Virchow2 resizes to 224 px like the repo). Check by re-embedding a few tiles with `picasso.embed` on TACC once slides are reachable.

Early hints (tiny sample: 60 slides, 2,000 tiles each, 4× dictionary, 2 epochs; not conclusions):

- UNI2 looks harder to compress: SAE explained variance 0.22 vs 0.30 (Virchow2); 30 PCA components explain 0.29 vs 0.58.
- UNI2 concepts matched less consistently between seeds.
- Slide-level GBM/LGG AUC of 1.0 on 6 test slides is meaningless at that size.

## 4. Immediate roadmap

1. Commit the branch (after user approval).
2. On TACC: set `ALLOCATION=<project>`, confirm read access to TRIDENT outputs, run `slurm/submit.sh data`, then `sae`, then `compare`. Fix SLURM issues as they appear.
3. Analyze the 21-run grid: Virchow2 vs UNI2 on reconstruction, dead/dense latents, seed stability (use seed-vs-seed matching as the noise floor for cross-encoder matching), probe performance on SAE codes vs raw embeddings, and qualitative concept atlases.
4. Verify preprocessing equivalence on a few slides (section 3, "Not done").
5. Add UNI2 to Emb2Img and the intervention scripts, run on TACC with raw slides.
6. Write up the Virchow2-vs-UNI2 comparison: this is the first deliverable.

## 5. Experiment protocol (applies to everything below)

- Fixed slide-level (patient-level) splits; never regenerated mid-study.
- ≥3 seeds; report mean ± std. Seed-vs-seed variation is the baseline any between-condition difference must exceed.
- One shared evaluation suite; each experiment pre-declares its gate.
- Each idea is added as its own experiment behind a config flag, run **isolated** (paper-recipe core + this idea) and **cumulative** (current best stack + this idea). It joins the stack only if it passes its gate.
- Log every run in `EXPERIMENTS.md` (id, hypothesis, config, isolated result, cumulative result, decision).

## 6. Compression / VAE experiments (next research step after the SAE comparison)

Motivation, stated honestly: the SAE does not need a VAE. A VAE (or quantizer) is only useful as a compression layer for the generative model in section 7. It must earn its place against simpler options, and must not destroy what the SAE reads. The UNI2 PCA hint above makes this comparison more interesting: compression may behave very differently per encoder.

Arms (all per encoder, on the same paired tiles):

- **C0 no compression** (raw embeddings).
- **C1 PCA** to 32/64 dims.
- **C2 continuous VAE:** per-tile MLP, `D → 64` (try 32), MSE + (1 − cosine) recon, β-KL with warm-up / free bits, inputs standardized with train stats.
- **C3 VQ-VAE (primary quantized variant):** codebook 1024 (sweep 512–8192), code dim 32–64, EMA updates, commitment β = 0.25, optional cosine lookup, k-means init, dead-code restarts, usage/perplexity logging. Gradient estimator sub-arms: standard straight-through (`z_q = z + (z_q - z).detach()`), rotation trick (Fifty et al. 2024), straight-through Gumbel-softmax ablation.
- **C4 lookup-free:** LFQ (MAGVIT-v2; sign-binarize 16–18 dims, entropy loss) and BSQ (l2-normalize then binarize). Needs a factorized categorical head downstream.
- Fallbacks: residual VQ, FSQ.

Gates: reconstruction (MSE, cosine, explained variance); probe accuracy on decoded vs original; **SAE concept preservation** (per-concept activation correlation, original vs decoded); codebook utilization for quantized arms. A compression arm is kept only if it clearly beats C0/C1 on what the downstream generator needs.

## 7. Longer-term ladder (after section 6)

- **Neighbor-expansion autoregressive generator** over the tile-embedding (or latent) grid: seeds on the tissue mask, BFS distance frontiers generated in parallel, transformer with 2D RoPE and target-position queries, queries attend to visible tiles only. Heads: categorical (VQ), factorized categorical (LFQ/BSQ), Gaussian, MAR-style diffusion. Gate: beat independent-tile and shuffled-tile baselines on spatial coherence.
- **Baselines and order ablations:** independent sampler, flow-matching DiT over latent crops, raster and random orders.
- **SAE in the loop:** concept statistics of generated grids vs real; steering by editing seed-tile concepts, then generating outward; AR likelihood to flag implausible counterfactuals.
- **Cross-modal translation (candidate headline task):** H&E → IHC embeddings or H&E → spatial transcriptomics (e.g. HEST), using paired ground truth for real evaluation.
- **JEPA-style latent prediction** as a comparison.
- **Augmentation utility:** train a MIL classifier on a small cohort with and without generated data.
- **Future:** multi-magnification (VAR-style) generation, serial sections, interpretability beyond SAEs, weight quantization for deployment.

## 8. Shared evaluation suite

Reconstruction (MSE, cosine, explained variance); dead/dense latent fractions; seed stability via concept matching; cross-encoder concept matching; probe accuracy (raw vs codes vs decoded); generator NLL; Fréchet distance / MMD on embeddings; neighbor cosine similarity at distances 1, 2, 4; concept frequency and neighbor co-occurrence; cross-modal metrics; augmentation delta; wall-clock.

## 9. Constraints and cautions

- No GPU on the Mac: local runs are smoke tests only. A full 12M × 1280 fp16 store is ~31 GB.
- Keep checkpoint volume under control on TACC (`$SCRATCH` purge policy; copy results worth keeping to `$WORK` or Corral).
- TCGA-GBM/LGG is adult data. Pediatric data later is a domain shift; concept dictionaries may not transfer and should be re-validated.
- Patient-derived (non-public) data stays on TACC unless the data agreement allows otherwise.
- SAE concepts are hypotheses until checked against annotations or pathologist review.

## 10. Open questions for the user

- TACC allocation name and preferred system (Lonestar6 vs Vista).
- Read access to the TRIDENT outputs in the other user's `/scratch`.
- Should the 221 missing Virchow2 slides be re-run with TRIDENT, or is the paired 1,481-slide set enough?
- Which downstream labels beyond GBM vs LGG are available (clinical, molecular, path reports are empty locally)?
- When pediatric data becomes available, which encoder outputs and format will it come in?
