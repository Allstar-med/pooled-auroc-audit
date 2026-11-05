# pooled-auroc-audit

Code and results for the manuscript *Pooled AUROC measures separation between recordings, not tracking within them*.

A frame-level AUROC pooled over many recordings is an exact mixture of discrimination within recordings ($W$) and between recordings ($X$):

$$E_1 = \alpha W + (1-\alpha) X,$$

where $\alpha$ is the share of positive–negative pairs that come from the same recording. This repository contains the audit that computes $E_1$, $W$, $X$ and $\alpha$ with recording-level (operation-level) inference, the pipeline that trains the 648 model configurations of the study on four public surgical video datasets, the result tables, and the code for every figure.

## Layout

| Folder | Content |
|---|---|
| `src/` | Audit library and training pipeline |
| `results/` | Result tables used in the manuscript |
| `figures/` | Code for Figs 1–6 and Extended Data Fig. 1, and for the numbers quoted in the text |
| `data/` | Anonymous operation-level C1 counts for the Fig. 1 pair diagram |
| `PLAN.md` | Analysis plan with its recorded changes and deviations (Supplementary Note 1) |

### `src/`

| File | Purpose |
|---|---|
| `oea_core.py` | Exact estimands from an operation-by-operation concordance matrix; matched operation bootstrap |
| `audit_all.py` | Audit of a model zoo: estimands, pairwise comparisons, rank agreement, elapsed-time baseline, selection, timing |
| `premature.py` | Premature declaration at a threshold fixed on validation data (90% frame-level sensitivity) |
| `label_structure.py` | Label descriptors computed from test labels alone, and their correlations with the audit outcomes |
| `manifests.py` | Frame manifests for Endoscapes, Cholec80, Cataract-101, HeiCo and SAGES |
| `extract_features.py` | Frozen features from five backbones (ResNet-50, ViT-B/16, DINOv2, SurgeNetXL DINOv2 and DINOv3) |
| `train_heads.py` | Frame and causal linear probes over a regularisation grid |
| `mstcn.py` | Causal multi-stage temporal convolutional network on frozen features |
| `finetune.py` | End-to-end fine-tuning of ResNet-50 and SurgeNetXL DINOv2 |
| `oea_io.py` | Shared writer for models trained over epochs |
| `sages_dedup.py` | Refined SAGES duplicate-footage check (not used for the reported results) |
| `run_all.py` | Stage 1: Endoscapes and Cholec80 manifests, features, linear probes, audit |
| `run_extension.py` | Stage 2: temporal networks, fine-tuning, SAGES and Cataract-101, audit, premature declaration, label structure |

## Reproducing the figures (no GPU, a few minutes)

```bash
pip install numpy scipy pandas matplotlib pillow
cd figures
python fig1.py      # Fig. 1
python figs.py      # Figs 2–6, Extended Data Fig. 1
python stats.py     # numbers quoted in the text -> out/stats.json
```

Figures are written to `figures/out/` as PDF and PNG.

## Re-running the audit from predictions

Row-level predictions, frame identifiers and frame-level annotations are not distributed in this repository or its releases. There is no public `predictions.zip` download. To rerun the audit, obtain authorised access to the source datasets and generate prediction files locally. The Endoscapes C1–C3 analysis also requires the original local frame-screen CSV; the public count summary cannot reconstruct that list. With the required files in `work/preds/` and `work/excluded_frames.csv`:

```bash
cd src
python audit_all.py ../work/results/selected ../work/preds/*[!n].csv --exclude ../work/excluded_frames.csv
python audit_all.py ../work/results/all ../work/preds/*[!n].csv --zoo all --exclude ../work/excluded_frames.csv
python premature.py ../work/results/premature ../work/preds/endo_*[!n].csv --dataset Endoscapes
python premature.py ../work/results/premature_sages ../work/preds/sages_*[!n].csv --dataset SAGES
```

(These commands use Bash glob expansion; on Windows, use Git Bash or pass prediction paths explicitly. The pattern `*[!n].csv` leaves out the `*_validation.csv` logs.) The input format is one row per frame, label and configuration, with the columns `dataset, split, unit, frame_id, t, label, y, score, model, backbone, head, C, sel_E1, sel_W, sel_head`. Any model can be audited by writing its predictions in this format; only predictions and recording identifiers are needed.

### Scope of the included results

The tables in `results/` are the supplied manuscript results. Rendering them reproduces the figures and text statistics from those tables, not the experiment from predictions. The supplied handoff identifies the Endoscapes Kendall-tau confidence intervals as transferred approximations (reported maximum difference: 0.015). They still require recalculation from the original predictions; this difference has not been independently verified here. Figure 3 also derives approximate normal-test p-values from the stored percentile intervals before BH adjustment.

The audit now applies `--exclude` to model-selection and causal-versus-frame-head comparisons as well as the main estimands. The supplied `selection.csv` and `heads.csv` tables have not been recomputed with this correction. Cross-operation AUROC is now computed directly from pair counts, including when no operation contains both classes. A prediction-based rerun is needed before claiming exact agreement between this code and all archived results. No new model training or full prediction-based audit was performed for this public package.

Run synthetic statistical and input-validation tests with `python -m unittest discover -s tests -v`. Synthetic inputs are created in temporary directories, separate from manuscript results.

## Running the full pipeline

The datasets are not redistributed. Obtain them from their creators:

- **Endoscapes2023:** CAMMA, CC BY-NC-SA 4.0.
- **SAGES CVS Challenge 2024:** Hugging Face `CAMMA-public/SAGES_CVS_Challenge_2024`, revision `deef093`.
- **Cholec80:** CAMMA.
- **Cataract-101:** Schoeffmann et al., 2018.

The SurgeNetXL checkpoints `DINOv2_ViTb14_size336_SurgeNetXL.pth` and `DINOv3_ViTb16_size336_SurgeNetXL.pth` are released by their authors.

```bash
pip install -r requirements.txt
cd src
python run_all.py --work ../work --weights WEIGHTS --endoscapes ENDOSCAPES --cholec80 CHOLEC80 --exclude ../work/excluded_frames.csv
python run_extension.py --work ../work --weights WEIGHTS --endoscapes ENDOSCAPES --cholec80 CHOLEC80 \
       --sages SAGES --cataract101 CATARACT101 --exclude ../work/excluded_frames.csv
```

Every step is skipped if its output exists, so an interrupted run can be restarted with the same command. `python run_extension.py ... --smoke` checks the GPU set-up on a few Endoscapes videos. ffmpeg is needed to sample frames from the SAGES and Cataract-101 videos.

The reported results used one RTX 4060 GPU, Python 3.13, PyTorch and timm.

## Notes on the data

- **Endoscapes C1–C3.** These labels are analysed on the 1,663 test keyframes that passed a screen for duplicate and blank images; 136 keyframes were excluded. The frame-level screen list remains local. CVS is analysed on all 1,799 test keyframes. Fig. 1 uses anonymous positive/negative counts per operation, sorted by counts; its order does not disclose source recording identifiers.
- **SAGES test clips.** The test set is the 293 clips in `src/sages_test_primary.csv`, one per footage group.
- **SAGES training clips.** Clips that share footage with a test clip are excluded by the coarse rule (the default `--sages-dedup coarse`).

## Licence

Code: MIT (see `LICENSE`). The anonymous count summary in `data/` is derived from Endoscapes and retains the source attribution and CC BY-NC-SA 4.0 notice. This repository does not grant access to, or redistribute, the source datasets or row-level predictions.
