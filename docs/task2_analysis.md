# Task 2 — Technical Analysis (MixUp, Label Smoothing & Robustness)

> _Annotation — not part of the submission. The block below is the technical
> analysis **exactly as emitted to stdout** by the submitted `task2/task.py`
> when run against the submitted checkpoint (`best_model.pth`,
> `train_metadata.pt`). It is reproduced here **1:1 and unedited**; every
> figure is the live value printed by the marked code (including the
> negative -0.10pp at std=0.05, where accuracy rises slightly under mild
> noise — reproduced as printed, not corrected). Context: Task 2 trains one
> 5-layer CNN with from-scratch MixUp (Beta sampled via Gamma decomposition),
> from-scratch label-smoothing cross-entropy (numerically stable log-sum-exp),
> and manual early stopping, then evaluates robustness to additive Gaussian
> noise._

```text
========================================================================
TECHNICAL ANALYSIS: MIXUP, LABEL SMOOTHING, AND ROBUSTNESS
========================================================================

[Architecture and Clean Performance]
A 5-layer CNN (3 Conv2d + 2 Linear) with BatchNorm was trained on
CIFAR-10 using MixUp (alpha=0.2), label smoothing (eps=0.1), and SGD
(lr=0.01, momentum=0.9, batch size=128). No dropout or weight decay was
used. On the uncorrupted test set, the model achieved 79.08%
accuracy.

[Noise Robustness]
Under additive Gaussian noise, accuracy fell from 79.08% on the
uncorrupted test set to 79.18% at noise_std=0.05, 77.86%
at 0.10, 75.33% at 0.15, and 70.25% at 0.20. This is a total drop of 8.83
percentage points at the highest noise level. The decline issmall at low
noise (-0.10 percentage points at std=0.05) and steeper at higher noise
(5.08 percentage points between std=0.15 and 0.20), which suggests the
learned representation is robust to small perturbations but becomes less
reliable once the corruption magnitude exceeds the variation seen during
training.

[MixUp and Memorisation]
In standard training, the model repeatedly sees the original training
examples with fixed labels, which makes it easier to fit those examples
too closely. MixUp changes this by training on convex combinations of
image pairs and their labels, with lambda sampled from Beta(alpha,
alpha). With alpha=0.2, many mixed samples are still close to one
original image, so they remain semantically meaningful, but they are no
longer exact repeats of the training data. Because the network never
receives the same input twice, memorisation of exact training samples
becomes much harder, forcing the network to learn features that inter-
polate smoothly between classes. This encourages smoother decision
boundaries and features that generalise beyond the training set.

[Label Smoothing and Overshooting]
Label smoothing replaces hard one-hot targets with softer targets. With
eps=0.1 in CIFAR-10, the correct class target becomes 0.91 and the
remaining probability mass is spread evenly across the other nine
classes. Under hard targets, the cross-entropy loss only reaches its
minimum when the correct logit grows without bound, which drives weight
magnitudes ever larger. With soft targets, the optimal logit gap is
finite, so the gradients that update the weights naturally shrink as the
network approaches the target distribution. This means the optimiser
stops pushing weights to extreme values, which keeps the model's output
less sensitive to small changes in the input. When combined with MixUp,
the targets are already soft from blending, so label smoothing further
reduces the sharpness of the target distribution and encourages
well-calibrated predictions.

[Early Stopping Dynamics]
Training stopped at epoch 22. The best validation-loss
checkpoint was at epoch 12, with val_loss=0.6745 and
val_acc=79.65%. This shows that the best validation performance was
reached well before the 100-epoch limit, and the following 10 epochs
in the patience window did not improve validation loss further. By
reloading the best checkpoint rather than using the final epoch weights,
the evaluated model reflects the point of strongest generalisation.

[GenAI Error Correction]
Opus 4.6 initially recommended using the torch.nn.functional module to
call F.log_softmax, arguing against removing this abstraction. When
prompted for a from-scratch implementation, it recommended the naive
formula: logits - logits.exp().sum(dim=1, keepdim=True).log(). Upon
further research and documentation review, I realised this approach
incurs a severe risk of float32 overflow, which would produce NaN losses
if any logit exceeded ~88. While our use of label smoothing and batch
normalisation makes reaching this threshold unlikely in this specific
setup, the final implementation controls for these floating-point
operations by applying the log-sum-exp trick. Subtracting the maximum
logit prior to exponentiation guarantees numerical stability regardless
of the raw input scale, ensuring the loss function would be defensible
in real-world production deployments.
========================================================================
```
