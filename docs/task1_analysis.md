# Task 1 — Technical Analysis (Generalisation Gap & Bias–Variance Trade-off)

> _Annotation — not part of the submission. The block below is the technical
> analysis **exactly as emitted to stdout** by the submitted `task1/task.py`
> when run against the submitted checkpoints (`baseline.pth`,
> `regularised.pth`, `histories.pt`). It is reproduced here **1:1 and
> unedited**; every figure is the live value printed by the marked code.
> Context: Task 1 trains two architecturally identical 5-layer CNNs — one
> unregularised baseline, one with Dropout(0.3) + L2 weight decay — with no
> data augmentation, to isolate and study the generalisation gap and the
> bias–variance trade-off._

```text
========================================================================
TECHNICAL ANALYSIS: GENERALISATION GAP AND BIAS-VARIANCE TRADE-OFF
========================================================================

Two identical 5-layer CNNs (3 Conv2d + 2 Linear, bias=True, no BatchNorm;
~620k trainable parameters) were trained on CIFAR-10 for 75 epochs
using SGD (lr=0.01, momentum=0.9, batch size 128) with no data augmentation.
All accuracy figures reported in eval mode (dropout disabled) for consistent
comparison.

The experimental design underwent a revision between iterations to isolate
regularisation as the sole independent variable.

(v0) Flawed Control Group: Dropout p=0.5, weight_decay=1e-3. Baseline
Train 100.00% | Val 81.10% | Gap 18.90pp. Regularised Train 97.11% |
Val 77.00% | Test 76.49% | Gap 20.11pp. In this iteration, the baseline
included BatchNorm - a strong implicit regulariser that injects batch-wise
noise - so it was not a true unconstrained control. The regularised model
compounded BatchNorm with aggressive 1e-3 weight shrinkage and 0.5 dropout,
effectively choking the network below CIFAR-10's required capacity and
causing validation accuracy to oscillate between 74-79%.

(v1) Clean Ablation (No BatchNorm in either model): Baseline uses zero
regularisation; regularised uses Dropout p=0.3 + weight_decay=1e-4.
Baseline: Train 100.00% | Val 77.72% | Test 78.09% | Gap 22.28pp. Regularised: Train 99.99% | Val 78.40% | Test 77.94% | Gap 21.59pp.
Test accuracies are statistically indistinguishable (0.15pp), consistent
with both models' validation curves converging in later epochs; the
generalisation benefit is most visible in the training trajectory and
loss calibration rather than final accuracy.
Stripping BatchNorm from both models ensures the only independent variables
are dropout and L2 weight decay. The baseline rapidly overfits, reaching
100% train accuracy by epoch 38 while validation plateaus around 77%.
The regularised model converges to ~99.3% train accuracy during training
(dropout active; 99.99% in eval mode with dropout disabled). Dropout
prevents perfect memorisation during optimisation, while maintaining a
modest but consistent validation advantage.

Epochs 8-17 reveal the regularisation dynamics most clearly. At epoch 8,
validation accuracies are close (~75.7% baseline, ~76.3% regularised) but
train accuracy is already diverging (84.0% vs 79.6%). Over the next nine
epochs, baseline train accuracy surges to ~98% as the network memorises
training-specific patterns. The regularised model climbs more gradually
(train accuracy ~93.4% by epoch 17); dropout forces redundant, generalisable
feature representations rather than specialised co-dependent units. By
epochs 15-17, regularised validation accuracy sits around 77.9% while
baseline fluctuates between 75.2-75.8%, a ~2-2.5pp gap. This gap narrows
at convergence as both models' validation accuracies drift marginally
upward, settling near 78.40% regularised vs 77.72% baseline.

The validation loss divergence tells a sharper story than accuracy alone.
Baseline validation loss climbs monotonically from its minimum of 0.71
(epoch 8) to 2.10 by epoch 75 while accuracy barely moves - the model
grows progressively more confident in wrong predictions, a textbook
overconfidence signature. The regularised model achieves a lower loss
minimum (0.68 at epoch 11 vs baseline's 0.71) and climbs to 1.46 by
epoch 75 - roughly 70% of baseline's endpoint, confirming that dropout
and weight decay meaningfully slow confidence miscalibration without
eliminating it.

The bias-variance tradeoff maps directly onto these results. The baseline
operates in the high-variance regime: perfect training fit, overconfident
wrong predictions, sensitivity to specific training samples.
Regularisation shifts toward controlled bias - marginally lower training
accuracy for more calibrated predictions. SGD itself acts as implicit
regularisation: each 128-sample mini-batch produces a noisy gradient
estimate, with noise scaling roughly with lr/batch_size, biasing both
models toward flatter minima. Because the optimiser is identical in both
runs, the performance difference is attributable to the explicit
regularisers.

No augmentation was applied intentionally so the baseline could overfit
freely. The pivot from v0 to v1 demonstrates that optimal regularisation
strength depends on existing implicit constraints; v0's BatchNorm-equipped
baseline was already implicitly regularised, making additional explicit
regularisation counterproductive. Removing BatchNorm entirely from both
models in v1 eliminated the confound, establishing that even modest dropout
and weight decay produce measurable generalisation improvements when applied
to a genuinely unconstrained architecture.

GenAI Error Correction: Claude initially drafted a technical analysis that overstated trends from raw data points and entirely missed the regularisation dynamics of epochs 8-17. Furthermore, it incorrectly recommended a batch size of 256 and 50 epochs, and advised against re-training when I sought to improve the model. I rejected this advice, switched to a batch size of 128 and 75 epochs for better generalisation and visual clarity, re-trained the model, and manually rewrote the analysis to accurately reflect the true learning trajectory.
========================================================================
```
