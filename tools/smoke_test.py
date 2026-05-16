"""Fast correctness checks for COMP0197 CW1 (no end-to-end training).

Exercises every core building block of both tasks on tiny synthetic
tensors so the code can be verified in seconds without downloading
CIFAR-10 or running the 75/100-epoch training loops:

  Task 1 : CNN forward shapes (dropout on/off); a 1-epoch micro training
           loop via the original ``train_model``.
  Task 2 : CNN forward shape; the Gamma-decomposition Beta sampler; MixUp
           image/label blending; the from-scratch label-smoothing loss
           (integer and soft targets, gradient flow, epsilon=0 sanity,
           and a large-logit overflow check that validates the
           log-sum-exp trick); a short real ``train_model`` run with
           patched epoch/patience constants in a throwaway directory.

The original ``task1/train.py`` and ``task2/train.py`` are imported
unmodified (their training code is guarded by ``__main__``). This script
writes nothing outside a temporary directory and exits non-zero on any
failure.

Run:
    micromamba run -n comp0197-pt python tools/smoke_test.py
"""

import importlib.util
import os
import sys
import tempfile

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_passed = 0
_failed = 0


def check(name, condition, detail=''):
    """Record and print a single PASS/FAIL assertion.

    Args:
        name (str): Human-readable check description.
        condition (bool): Truthy means the check passed.
        detail (str): Extra context printed on failure.
    """
    global _passed, _failed
    if condition:
        _passed += 1
        print(f'  PASS  {name}')
    else:
        _failed += 1
        print(f'  FAIL  {name}  {detail}')


def load_module(task_dir, alias):
    """Import a task's ``train.py`` under a unique module name.

    Args:
        task_dir (str): Folder name ('task1' or 'task2') under repo root.
        alias (str): Unique module name to register the import as.

    Returns:
        module: The imported ``train`` module (training code not run; it
            is guarded by ``if __name__ == '__main__'``).
    """
    path = os.path.join(REPO_ROOT, task_dir, 'train.py')
    spec = importlib.util.spec_from_file_location(alias, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[alias] = module
    spec.loader.exec_module(module)
    return module


def test_task1(t1):
    """Validate Task 1 model shapes and a 1-epoch micro training loop.

    Args:
        t1 (module): Imported ``task1/train.py`` module.
    """
    print('Task 1')
    torch.manual_seed(0)
    x = torch.randn(4, 3, 32, 32)

    for use_dropout in (False, True):
        model = t1.CNN(num_classes=10, use_dropout=use_dropout)
        model.eval()
        out = model(x)
        check(
            f'CNN forward shape (use_dropout={use_dropout})',
            out.shape == (4, 10) and torch.isfinite(out).all(),
            f'got {tuple(out.shape)}',
        )

    # Micro training loop on synthetic data via the real train_model.
    imgs = torch.randn(16, 3, 32, 32)
    lbls = torch.randint(0, 10, (16,))
    loader = DataLoader(
        TensorDataset(imgs, lbls), batch_size=8, drop_last=True,
    )
    model = t1.CNN(num_classes=10, use_dropout=True)
    opt = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    hist = t1.train_model(
        model, loader, loader, opt, nn.CrossEntropyLoss(), num_epochs=1,
    )
    ok = (
        set(hist) == {'train_acc', 'val_acc', 'train_loss', 'val_loss'}
        and all(len(hist[k]) == 1 for k in hist)
        and all(torch.isfinite(torch.tensor(hist[k][0])) for k in hist)
    )
    check('train_model 1-epoch micro loop', ok, f'history={hist}')


def test_task2(t2):
    """Validate Task 2 model, MixUp, loss, and a short real train run.

    Args:
        t2 (module): Imported ``task2/train.py`` module.
    """
    print('Task 2')
    torch.manual_seed(0)

    model = t2.CNN(num_classes=10)
    model.eval()
    out = model(torch.randn(4, 3, 32, 32))
    check(
        'CNN forward shape',
        out.shape == (4, 10) and torch.isfinite(out).all(),
        f'got {tuple(out.shape)}',
    )

    # Beta sampler via Gamma decomposition: scalar in [0, 1].
    samples = torch.stack([t2.sample_beta(0.2) for _ in range(500)])
    check(
        'sample_beta range & shape',
        samples.shape == (500,)
        and (samples >= 0).all() and (samples <= 1).all(),
        f'min={samples.min():.3f} max={samples.max():.3f}',
    )

    # MixUp: image shape preserved, labels become valid soft rows.
    imgs = torch.randn(8, 3, 32, 32)
    lbls = torch.randint(0, 10, (8,))
    mi, ml = t2.mixup_batch(imgs, lbls, alpha=0.2, num_classes=10)
    rows_sum_one = torch.allclose(
        ml.sum(dim=1), torch.ones(8), atol=1e-5,
    )
    check(
        'mixup_batch shapes & soft labels',
        mi.shape == imgs.shape and ml.shape == (8, 10)
        and rows_sum_one and (ml >= 0).all(),
        f'img={tuple(mi.shape)} lbl={tuple(ml.shape)}',
    )

    # Label-smoothing loss: integer-target path, finite + gradient flows.
    logits = torch.randn(8, 10, requires_grad=True)
    targets = torch.randint(0, 10, (8,))
    loss = t2.smooth_ce_loss(logits, targets, epsilon=0.1, num_classes=10)
    loss.backward()
    check(
        'smooth_ce_loss int targets (finite + grad)',
        loss.dim() == 0 and torch.isfinite(loss)
        and logits.grad is not None
        and torch.isfinite(logits.grad).all(),
        f'loss={loss.item()}',
    )

    # Soft-target path (MixUp-style targets).
    soft = torch.softmax(torch.randn(8, 10), dim=1)
    loss_soft = t2.smooth_ce_loss(
        torch.randn(8, 10), soft, epsilon=0.1, num_classes=10,
    )
    check(
        'smooth_ce_loss soft targets (finite)',
        loss_soft.dim() == 0 and torch.isfinite(loss_soft),
        f'loss={loss_soft.item()}',
    )

    # epsilon=0 must reduce to plain cross-entropy on hard targets.
    lg = torch.randn(8, 10)
    tg = torch.randint(0, 10, (8,))
    custom = t2.smooth_ce_loss(lg, tg, epsilon=0.0, num_classes=10)
    ref = nn.functional.cross_entropy(lg, tg)
    check(
        'smooth_ce_loss(eps=0) == cross_entropy',
        torch.allclose(custom, ref, atol=1e-5),
        f'custom={custom.item():.6f} ref={ref.item():.6f}',
    )

    # Large-logit overflow check: validates the log-sum-exp trick.
    huge = torch.full((4, 10), 1e4)
    huge[:, 0] = 2e4
    loss_huge = t2.smooth_ce_loss(
        huge, torch.zeros(4, dtype=torch.long), epsilon=0.1,
        num_classes=10,
    )
    check(
        'smooth_ce_loss numerically stable at logit=2e4',
        torch.isfinite(loss_huge),
        f'loss={loss_huge.item()}',
    )

    # Short real train_model run with patched constants in a temp dir.
    imgs = torch.randn(32, 3, 32, 32)
    lbls = torch.randint(0, 10, (32,))
    loader = DataLoader(
        TensorDataset(imgs, lbls), batch_size=8, drop_last=True,
    )
    t2.NUM_EPOCHS, t2.PATIENCE = 2, 1
    cwd = os.getcwd()
    try:
        with tempfile.TemporaryDirectory() as tmp:
            os.chdir(tmp)
            torch.manual_seed(42)
            t2.train_model(t2.CNN(num_classes=10), loader, loader)
            wrote = os.path.exists('best_model.pth') and os.path.exists(
                'train_metadata.pt',
            )
    finally:
        os.chdir(cwd)
    check('train_model writes best checkpoint + metadata', wrote)


def main():
    """Run all checks and exit non-zero if any failed."""
    print('=' * 60)
    print('COMP0197 CW1 — smoke test (no end-to-end training)')
    print('=' * 60)
    t1 = load_module('task1', 't1_train')
    t2 = load_module('task2', 't2_train')
    test_task1(t1)
    test_task2(t2)
    print('=' * 60)
    print(f'RESULT: {_passed} passed, {_failed} failed')
    print('=' * 60)
    sys.exit(1 if _failed else 0)


if __name__ == '__main__':
    main()
