# Generative AI Usage

This work was produced with generative-AI assistance, used as a coding and
research aid and **critically reviewed** rather than accepted at face
value. This note is a factual summary; the specific technical corrections
are documented verbatim in the *GenAI Error Correction* sections of
[`docs/task1_analysis.md`](docs/task1_analysis.md) and
[`docs/task2_analysis.md`](docs/task2_analysis.md).

## Tools

- **Claude (Opus 4.6)** — starter code, debugging, and discussion of
  training direction, iterated on substantially against online research and
  lecture material.

## How it was used

- Drafting boilerplate (data loading, training-loop scaffolding) and
  explaining errors.
- Sounding board for design decisions, all independently verified before
  adoption.

## Where its advice was rejected and corrected

- **Task 1 — hyperparameters & analysis.** The model initially recommended
  a batch size of 256 and 50 epochs and advised against re-training. This
  was rejected: batch size 128 / 75 epochs generalised better and gave
  clearer visualisations. An AI-drafted analysis that overstated raw-data
  trends and missed the epoch 8–17 regularisation dynamics was discarded
  and rewritten by hand from the measured results.
- **Task 2 — numerical stability.** The model first recommended using
  `torch.nn.functional.log_softmax`, then, when a from-scratch version was
  requested, the naïve `logits - logits.exp().sum().log()`. This is unsafe:
  float32 overflow produces `NaN` once a logit exceeds ~88. The final
  implementation applies the log-sum-exp trick (subtract the max logit
  before exponentiating) for guaranteed stability.

## Integrity

All design choices, debugging conclusions and the final written analyses
are the author's own. Every measured figure in `docs/` is the live value
printed by the submitted code when run against the submitted checkpoints,
reproduced 1:1 and unedited.
