# Analysis plan for the extended audit

> This is the analysis plan as fixed on 26 October 2026, with the changes recorded before the affected analyses were run. "The published audit" and "the original analyses" refer to an earlier analysis of the Endoscapes and SAGES test sets with three ResNet-18 checkpoints, described in a related manuscript. Deviations not recorded here are listed at the end.

Fixed on 26 October 2026, before any feature of a new model is computed. Changes after that date are listed at the end with their reason.

## Question

The published audit found that pooled frame-level AUROC (E1) on Endoscapes and SAGES 2024 is dominated by cross-operation pairs (α < 3%). It also found that the three original ResNet-18 classifiers did not order frames within an operation better than elapsed time.

A reviewer can attribute both results to weak models. The extension asks whether the same holds for current models: frozen surgical and general foundation backbones with linear heads, trained on the official training splits and selected on the official validation splits.

## Models

There is one zoo per dataset and label. All backbones are frozen, and every head is a logistic regression with class-balanced weights.

| Backbone | Pretraining |
|---|---|
| ResNet-50 | ImageNet-1k |
| ViT-B/16 | ImageNet-21k, fine-tuned on ImageNet-1k |
| DINOv2 ViT-B/14 | LVD-142M, 336 px |
| SurgeNetXL DINOv2 ViT-B/14 | Surgical video, 336 px |
| SurgeNetXL DINOv3 ViT-B/16 | Surgical video, 336 px |

Each backbone gets two heads:

- **Frame head:** the features of the frame only.
- **Causal head:** the frame features concatenated with an exponential moving average (λ = 0.8) of the features of the earlier frames of the same video. It uses no future frames.

The grid is C ∈ {0.001, 0.01, 0.1, 1}. The selected configuration per backbone and label is the one with the highest validation E1, as a conventional pipeline would choose. The configuration with the highest validation W is recorded for P6.

The three original ResNet-18 models of the published audit are added unchanged; they have test predictions only.

## Data

- **Endoscapes-CVS201 keyframes.** Official 120/41/40 video split. Labels are C1, C2, C3 (majority of three raters) and CVS (all three met). The unit is the video.
- **Cholec80 tool presence and phase.** 1 fps, split 1–32 / 33–40 / 41–80, unit = video. SurgeNetXL includes Cholec80 in its pretraining. Its Cholec80 results are therefore reported separately and are not used for any hypothesis below.
- **SAGES 2024 CVS.** Included only if training-split frames and labels are available. Otherwise it is analysed with the published predictions only.

## Hypotheses and decision rules

All analyses use the test split and B = 10,000 matched unit-bootstrap draws (seed 20261101). Intervals for W, E2 and their contrasts are t intervals with n_eff − 1 degrees of freedom; E1 and X intervals are percentile intervals.

**P1 (α).** For every dataset and label, α is below 0.10.
- *Supported* if all point estimates are below 0.10.

**P2 (time baseline).** For the selected configurations, W minus the W of elapsed time is reported with t intervals and Benjamini–Hochberg q within dataset and estimand.
- *Supported in the strong form* if at least half of model–label combinations have q < 0.05 with ΔW > 0. This is the result expected if weak models explained the published finding.
- *Supported in the weak form* if fewer than half do.

Both outcomes are reported; neither is a failure of the study.

**P3 (ranking concordance).**
- Kendall τ between model rankings by E1 and by W is reported per dataset and label, with a bootstrap interval.
- Also reported: the fraction of model pairs whose E1 and W differences have opposite signs, the number of pairs with significant differences of opposite sign, and whether the best model by E1 is also best by W.
- The primary zoo is the selected configurations; the full grid is a sensitivity analysis.
- *Supported* if τ is below 0.8 in at least half of the cells.

**P4 (causal head).** The causal head reaches a higher test W than the frame head of the same backbone.
- Reported descriptively as the paired differences.

**P5 (onset timing).** For every video with at least one positive frame, the onset is the first positive time.
- A model's predicted onset is the first time its score reaches a threshold fixed on the validation split. The threshold minimises the mean absolute onset error, capped at 300 s, with a missed onset counted as 300 s.
- Reported per model: the capped mean absolute error with bootstrap interval, the fraction within 30 s, early and missed onsets, and the fraction of videos without onset that raise an alarm.
- The rankings of models by E1 and by W are compared with the ranking by timing error (Kendall τ).
- *Supported* if τ(W, timing) exceeds τ(E1, timing) in more than half of the cells. This analysis is exploratory, because a fixed threshold also depends on calibration across videos.

**P6 (model selection).** For every backbone and label, the configuration chosen by validation E1 is compared with the one chosen by validation W.
- Reported: how often the two differ, and the test W of each.

## What is not changed

The published estimands, the eleven-item audit and the original analyses are kept as they are. The extension adds models and tasks; it does not re-select anything in the published analyses.

## Changes after fixing the plan

1. **Regularisation grid extended (31 October 2026).** In the first run on Endoscapes, validation E1 increased monotonically as C decreased for every backbone, head and label, so every selected configuration lay at the lower edge of the grid (C = 0.001) and the validation optimum was not reached. The grid was extended to C ∈ {0.00001, 0.00003, 0.0001, 0.0003, 0.001, 0.01, 0.1, 1}, and all heads were refitted. The decision was based only on validation results; selection rules, test analyses and hypotheses are unchanged. Results of the first grid are kept as a sensitivity analysis.
2. **Cholec80 sampled every 5 s (31 October 2026, before any Cholec80 feature was computed).** Cholec80 is analysed at one frame every 5 s instead of 1 fps. This matches the Endoscapes keyframe spacing, so the two datasets are audited at the same temporal resolution, and it keeps the number of near-identical consecutive frames and the computing cost proportionate. Labels, split and analyses are unchanged.
3. **Extension A–D (1 November 2026, after the results of rounds 1–3 were known and before any of the new models was trained or any new dataset was processed).**
   - **A. Trained models.** Two kinds of model are added to the zoo of each dataset. The first is a causal multi-stage temporal convolutional network (MS-TCN; two stages, ten layers, 64 channels) on the frozen features of each backbone, with three seeds. Checkpoints are taken every 10 epochs up to 60. The second is end-to-end fine-tuning of ResNet-50 and SurgeNetXL DINOv2 ViT-B/14 on Endoscapes and Cholec80, with three seeds and 12 epochs, scoring every epoch. Checkpoints play the role of the regularisation grid: the configuration is selected per label on the validation split by E1 (sel_E1) and by W (sel_W). P1–P3 and P6 are applied with unchanged decision rules to the enlarged zoo. In the "all" zoo, only the validation-selected checkpoints of these models enter, together with every regularisation value of the linear heads.
   - **B. Other procedures.** Two datasets are added.
     - *HeiCo* (Heidelberg colorectal; 30 laparoscopic colorectal resections of three types; 14 phases, one-vs-rest). It is sampled every 5 s. The split, fixed here, takes procedures 1–5 of each type for training, 6 for validation and 7–10 for testing.
     - *Cataract-101* (101 microscope-recorded cataract operations; 10 phases, one-vs-rest). It is sampled every 2 s, because the operations last about 8 min and some phases last under 30 s. The split, fixed here by sorted video id, is 60 training, 10 validation and 31 test videos.
     - The full pipeline is applied to both: linear heads, MS-TCN and the audit. P1–P3 and P6 are confirmatory for these datasets.
     - A single in-house colorectal operation with frame-wise phase, instrument and anatomy labels cannot support any within- versus between-operation contrast. It is used only to illustrate label time courses.
   - **C. Premature declaration (exploratory).** The threshold is fixed on validation so that the frame-level sensitivity is 0.90. On the test split, we count the share of frames before the first positive frame of an operation that are flagged, together with related operation-level rates. Two comparisons are reported. The first compares the configurations chosen by validation E1 and by validation W. The second gives the Kendall τ of the premature-rate ranking with the rankings by E1, W and X. The expected direction is τ(W) > τ(E1).
   - **D. Label structure (exploratory).** For each label, the descriptors are computed from the test labels alone: between-operation share of variance, time τ, switches per hour, prevalence and mixed operations. Each is correlated across labels and datasets with the median |E1 − W|, the τ between the E1 and W rankings, and the share of configurations beating elapsed time. The expected direction is that labels that switch on once and stay on (high time τ, few switches) show larger |E1 − W| and lower τ(E1, W).
4. **SAGES CVS 2024 added as a replication dataset (2 November 2026, after the results of A–D were known and before any model was trained on SAGES).**
   - **Why.** The finding that pooled AUROC misranks models and that no model beats elapsed time within operations rests on one dataset, Endoscapes. SAGES CVS 2024 tests whether it holds for the same kind of label in independent data. SAGES consists of 1000 90-s clips from many centres and countries, with C1–C3 rated by three raters every 5 s.
   - **Labels.** C1–C3 by majority of three raters; CVS when all three are met. These are the same rules as for Endoscapes.
   - **Split.**
     - The official 700 training clips and 300 test clips are used.
     - The test set is the 293 primary clips of the published audit, which has one clip per footage group.
     - Training clips whose footage duplicates a test clip are excluded. The rule was revised on 3 November 2026, before any SAGES model was trained, for the reason below.
       - The first rule (at least 5 labelled frames within 6 bits of a frame of the other clip, 64-bit difference hash of the whole frame) flagged 262 of 700 training clips. Visual review showed that most flagged pairs came from different operations: the coarse hash is dominated by the dark circular border and the overall layout of laparoscopic frames.
       - Revised rule. The 64-bit hash is used only to screen candidate pairs. A pair is confirmed when at least 5 informative labelled frames of one clip have a frame of the other clip whose normalised cross-correlation over the central field of view (central 70% of each dimension, 48 × 48 grey levels) reaches the 99.9th percentile of best-match correlations between random train–test clip pairs.
       - The 7 test clips known from the published audit to repeat another test clip serve as positive controls.
       - Confirmed pairs are drawn on review sheets, and a pair may be overruled only after visual review (recorded in sages_dup_override.csv).
     - 100 training clips are held out for validation. They are taken in SHA-256 order of the clip id, skipping clips that share footage with another training clip.
   - **Models.** Linear heads on the five frozen backbones, MS-TCN with three seeds, and fine-tuning of ResNet-50 and SurgeNetXL ViT-B with three seeds. The settings are identical to the other datasets.
   - **Predictions, fixed before training.**
     - (i) No selected configuration has W above that of elapsed time with q < 0.05 for C2, C3 or CVS.
     - (ii) τ(E1, W) < 0.8 for at least three of the four labels, and the best configuration by E1 differs from the best by W for at least three of the four labels.
     - (iii) The between-operation share of label variance exceeds 0.10 for all four labels.
     - Experiment C is repeated on SAGES.
   - **Disclosure.** The SAGES test labels were already analysed descriptively in the published audit, using models trained on Endoscapes only. No SAGES training data had been used before this change.

## Deviations not recorded above

- HeiCo was not analysed because data access had not been granted.
- The refined SAGES deduplication rule of change 4 was not applied; the coarse rule was kept (`--sages-dedup coarse`, the default of `run_extension.py`).
- The three ResNet-18 checkpoints of the earlier analysis were removed from the zoo, so that every configuration was trained and selected under this plan. Endoscapes C1–C3 are analysed on the 1,663 test keyframes that passed the image screen (the local `excluded_frames.csv`, not distributed in this repository).
- P5 (onset timing) was not supported: τ(W, timing) exceeded τ(E1, timing) in 7 of 32 labels.
