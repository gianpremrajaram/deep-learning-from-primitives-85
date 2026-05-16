# Deep Learning From Primitives — COMP0197 CW1 (Grade: 85)

A from-scratch study of **generalisation** in convolutional networks on
CIFAR-10, built under a deliberately strict constraint: **only `torch`,
`torchvision`, `pillow` and the Python standard library** — no
`torchvision.models`, no pretrained weights, no `sklearn`, no
`torch.distributions`, and no high-level loss/scheduler/early-stopping
helpers. Every regularisation technique is implemented from basic tensor
operations.

This repository is the unzipped, unmodified marked submission (CPU-only
PyTorch, Python 3.12) plus added documentation. The two task folders
(`task1/`, `task2/`) contain the original graded code and artefacts and
have **not been edited**.

---

## Repository structure

```
.
├── README.md               # this file
├── AI-USAGE.md             # how generative AI was used and corrected
├── docs/
│   ├── task1_analysis.md   # verbatim Task 1 analysis (as printed by task1/task.py)
│   └── task2_analysis.md   # verbatim Task 2 analysis (as printed by task2/task.py)
├── tools/
│   └── smoke_test.py       # fast correctness checks (no full training needed)
├── task1/                  # ── original submission, untouched ──
│   ├── train.py            # trains baseline + regularised CNNs (75 epochs)
│   ├── task.py             # evaluates checkpoints, draws plot, prints analysis
│   ├── baseline.pth        # trained baseline weights (state_dict)
│   ├── regularised.pth     # trained Dropout+L2 weights (state_dict)
│   ├── histories.pt        # per-epoch train/val loss & accuracy
│   └── generalization_gap.png   # train-vs-val curves (Pillow-rendered)
└── task2/                  # ── original submission, untouched ──
    ├── train.py            # trains CNN with MixUp + label smoothing + early stop
    ├── task.py             # noisy-test evaluation, MixUp montage, prints analysis
    ├── best_model.pth      # best-validation checkpoint (state_dict)
    ├── train_metadata.pt   # early-stopping metadata (stopped/best epoch)
    └── robustness_demo.png # MixUp montage (Pillow-rendered)
```

---

## Task 1 — The generalisation gap & bias–variance trade-off

**Goal.** Demonstrate, in a clean controlled experiment, how explicit
regularisation reshapes the train-vs-validation trajectory.

**Implementation.** Two *architecturally identical* 5-layer CNNs
(3×`Conv2d` + 2×`Linear`, ~620k parameters, no BatchNorm) are trained for
75 epochs with SGD (lr 0.01, momentum 0.9, batch 128) and **no data
augmentation**, so the baseline is free to overfit. The only difference
between the two runs is the regulariser: the baseline uses none; the
regularised model adds `Dropout(0.3)` and L2 `weight_decay=1e-4`. Removing
BatchNorm from *both* models is deliberate — an earlier iteration (v0,
documented in the analysis) showed BatchNorm acts as a strong implicit
regulariser and confounds the comparison.

**Headline result** (reproduced from the submitted checkpoints, eval mode):

| Model       | Train acc | Val acc | Test acc |
|-------------|-----------|---------|----------|
| Baseline    | 100.00%   | 77.72%  | 78.09%   |
| Regularised | 99.99%    | 78.40%  | 77.94%   |

Final test accuracy is essentially tied — the regularisation pay-off is in
the *trajectory* and *loss calibration*, not the endpoint. Baseline
validation loss climbs monotonically from ~0.71 to ~2.10 (growing
confidently wrong) while the regularised model's climbs only to ~1.46. This
is the bias–variance trade-off made concrete: the baseline sits in the
high-variance regime; regularisation trades a sliver of training fit for
better-calibrated predictions. Full reasoning, including the v0→v1 design
correction, is in **[`docs/task1_analysis.md`](docs/task1_analysis.md)**.

## Task 2 — MixUp, label smoothing & robustness (all from scratch)

**Goal.** Build MixUp, label-smoothing cross-entropy and early stopping
from primitives, then measure robustness to input noise.

**Implementation highlights.**

- **MixUp** — a per-batch `λ` is drawn from `Beta(α, α)` *without*
  `torch.distributions`, using the Gamma decomposition
  `g1 / (g1 + g2)` with `g1, g2 ~ Gamma(α, 1)`. Both images **and**
  one-hot labels are blended.
- **Label-smoothing cross-entropy** — implemented as
  `-(smooth · log_softmax(logits)).sum().mean()` with the **log-sum-exp
  trick** (subtract the max logit before exponentiating) to guarantee
  float32 numerical stability. `nn.CrossEntropyLoss` is never used; it
  silently ignores soft targets.
- **Early stopping** — a manual patience counter (patience 10, strict
  `<` improvement) that checkpoints on improvement and **reloads the best
  checkpoint** after the loop, so the evaluated model is the strongest
  generaliser, not the last epoch.

**Headline result** (reproduced from the submitted checkpoint):

| Noise σ | 0.00  | 0.05  | 0.10  | 0.15  | 0.20  |
|---------|-------|-------|-------|-------|-------|
| Test acc| 79.08%| 79.18%| 77.86%| 75.33%| 70.25%|

Accuracy is flat (even marginally up) under mild noise and degrades
gracefully — an 8.83pp drop only at the strongest corruption — indicating
the soft-target training learned smooth, robust features. Training
early-stopped at epoch 22, reloading the best checkpoint from epoch 12.
Full reasoning is in **[`docs/task2_analysis.md`](docs/task2_analysis.md)**.

---

## Running it

Requires the `comp0197-pt` environment (Python 3.12, CPU-only PyTorch).
CIFAR-10 auto-downloads to a git-ignored `./data` on first run. Scripts use
**bare relative paths** and must be run from inside their task folder:

```bash
micromamba activate comp0197-pt

cd task1
python train.py     # trains baseline + regularised → *.pth, histories.pt
python task.py      # → generalization_gap.png + printed analysis

cd ../task2
python train.py     # trains with MixUp+smoothing+early-stop → best_model.pth
python task.py      # → robustness_demo.png + printed analysis
```

The committed `.pth`/`.pt` artefacts are the exact marked checkpoints, so
`task.py` reproduces the published numbers **without retraining**.

## Verification

`tools/smoke_test.py` exercises every core building block — model forward
shapes, the Gamma-based Beta sampler, MixUp blending, the from-scratch
label-smoothing loss (including a large-logit overflow check that validates
the log-sum-exp trick), and a micro training loop — on tiny synthetic
tensors in seconds, **without** any end-to-end CIFAR training. It imports
the original `train.py` modules unmodified. Run:

```bash
micromamba run -n comp0197-pt python tools/smoke_test.py
```

---

## Notes

- Grade: **85**.
- Original submission code in `task1/` and `task2/` is preserved exactly as
  marked; `README.md`, `AI-USAGE.md`, `docs/` and `tools/` are added
  documentation only.
- Generative-AI usage and the specific places where its advice was
  rejected and corrected are documented in
  **[`AI-USAGE.md`](AI-USAGE.md)** and, in detail, in the *GenAI Error
  Correction* sections of the two analysis files.
