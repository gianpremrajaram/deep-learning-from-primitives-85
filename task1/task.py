# AI usage: NotebookLM for RAG based x-reference of cwk req and lecture slides,
# GPT via Perplexity for research development env options (including pythorch vs
#tensorflow). Claude (opus 4.6) for starter code & later debugging - post-
# architecting end to end solution based on lecture material.

# AI usage error/correction: AI technical analysis draft was very far off.
# Spefically, attempt to evaluate trends from raw output data points were
# overstated and entirely missed epoch 8-17 regularisation dynamic.
# additionally, in in evaluating re-train (which I ultimately opted for, resulting
# in better results) AI advised to not re-train.
# (v0)Dropout p=0.5, weight_decay=1e-3 ->(v1)Dropout p=0.3, weight_decay=1e-4.
# Resulted in overall better results, and clearer learning dynamics to cover in
# the technical analysis.

"""COMP0197 CW1 Task 1 - Evaluation Script

Loads trained baseline and regularised CNN models, evaluates on train/val/test
sets, generates generalization_gap.png (per-epoch accuracy curves using Pillow),
and prints a ~500-word technical analysis of the generalisation gap.

Expects: baseline.pth, regularised.pth, histories.pt (from train.py).
Outputs: generalization_gap.png, terminal-printed analysis.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import torchvision
import torchvision.transforms as transforms
from PIL import Image, ImageDraw, ImageFont

# ── Constants (must match train.py) ────────────────────────────────────────
BATCH_SIZE = 128
NUM_CLASSES = 10
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


# ── Model (identical to train.py) ─────────────────────────────────────────

class CNN(nn.Module):
    """5-layer CNN (3 Conv2d + 2 Linear) for CIFAR-10 classification.

    Architecture:
        Conv1(3->32, bias=True) -> ReLU -> MaxPool
        Conv2(32->64, bias=True) -> ReLU -> MaxPool
        Conv3(64->128, bias=True) -> ReLU -> MaxPool
        Flatten(128*4*4=2048)
        FC1(2048->256) -> ReLU -> [Dropout]
        FC2(256->10)

    Args:
        num_classes (int): Number of output classes. Default: 10.
        use_dropout (bool): If True, applies Dropout(0.3) after FC1 ReLU.
    """

    def __init__(self, num_classes=10, use_dropout=False):
        """Initialise all layers explicitly.

        Args:
            num_classes (int): Output dimension (number of classes).
            use_dropout (bool): Whether to insert Dropout after FC1 ReLU.
        """
        super().__init__()
        # Layer 1: Conv2d  (B,3,32,32) -> (B,32,16,16) after pool
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=True)

        # Layer 2: Conv2d  (B,32,16,16) -> (B,64,8,8) after pool
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=True)

        # Layer 3: Conv2d  (B,64,8,8) -> (B,128,4,4) after pool
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=True)

        # Layer 4: Linear  (B,2048) -> (B,256)
        self.fc1 = nn.Linear(128 * 4 * 4, 256)

        # Layer 5: Linear  (B,256) -> (B,num_classes)
        self.fc2 = nn.Linear(256, num_classes)

        # Shared non-parametric ops
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        self.use_dropout = use_dropout
        if use_dropout:
            self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        """Forward pass through the network.

        Args:
            x (torch.Tensor): Input images, shape (B, 3, 32, 32).

        Returns:
            torch.Tensor: Raw logits, shape (B, num_classes).
        """
        x = self.pool(self.relu(self.conv1(x)))   # -> (B,32,16,16)
        x = self.pool(self.relu(self.conv2(x)))   # -> (B,64,8,8)
        x = self.pool(self.relu(self.conv3(x)))   # -> (B,128,4,4)
        x = x.view(x.size(0), -1)                           # -> (B,2048)
        x = self.relu(self.fc1(x))                           # -> (B,256)
        if self.use_dropout:
            x = self.dropout(x)
        x = self.fc2(x)                                     # -> (B,10)
        return x


# ── Data ───────────────────────────────────────────────────────────────────

def load_data():
    """Load CIFAR-10 with the same train/val split used during training.

    Returns:
        tuple[DataLoader, DataLoader, DataLoader]:
            (train_loader, val_loader, test_loader).
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    full_train = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform,
    )
    test_set = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform,
    )

    generator = torch.Generator().manual_seed(42)
    train_set, val_set = random_split(
        full_train, [45000, 5000], generator=generator,
    )

    train_loader = DataLoader(
        train_set, batch_size=BATCH_SIZE, shuffle=False, drop_last=False,
    )
    val_loader = DataLoader(
        val_set, batch_size=BATCH_SIZE, shuffle=False, drop_last=False,
    )
    test_loader = DataLoader(
        test_set, batch_size=BATCH_SIZE, shuffle=False, drop_last=False,
    )
    return train_loader, val_loader, test_loader


# ── Evaluation ─────────────────────────────────────────────────────────────

def evaluate(model, loader):
    """Compute classification accuracy on a data loader.

    Args:
        model (CNN): Trained model (must already be in eval mode).
        loader (DataLoader): Batches of (images, labels).

    Returns:
        float: Accuracy in [0, 1].
    """
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    return correct / total


# ── Pillow plot ────────────────────────────────────────────────────────────

def _get_font(size=14):
    """Load a default font at the given size with graceful fallback.

    Args:
        size (int): Desired font size in points.

    Returns:
        PIL.ImageFont: Loaded font object.
    """
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def draw_plot(histories, filename='generalization_gap.png'):
    """Generate per-epoch accuracy curves for baseline and regularised models.

    Draws a line chart with 4 curves (baseline train/val, regularised
    train/val), axes, grid, tick labels, axis titles, and a legend.
    Uses only Pillow (no matplotlib).

    Args:
        histories (dict): Must contain 'baseline_history' and
            'regularised_history', each a dict with list-valued keys
            'train_acc' and 'val_acc'.
        filename (str): Output PNG path (relative).
    """
    # ── Layout ─────────────────────────────────────────────────────
    W, H = 920, 640
    ml, mr, mt, mb = 75, 175, 50, 60   # margins
    pw = W - ml - mr                     # plot width in pixels
    ph = H - mt - mb                     # plot height in pixels

    f_title = _get_font(18)
    f_label = _get_font(14)
    f_tick = _get_font(11)

    # ── Data series ────────────────────────────────────────────────
    bl = histories['baseline_history']
    rg = histories['regularised_history']
    n = len(bl['train_acc'])

    series = [
        ('Baseline Train',    bl['train_acc'],  (210, 50, 50)),    # red
        ('Baseline Val',      bl['val_acc'],    (50, 80, 210)),    # blue
        ('Regularised Train', rg['train_acc'],  (230, 140, 20)),   # orange
        ('Regularised Val',   rg['val_acc'],    (30, 160, 70)),    # green
    ]

    # ── Y-axis auto-scale ─────────────────────────────────────────
    all_acc = [v for _, data, _ in series for v in data]
    y_lo = max(0.0, min(all_acc) - 0.03)
    y_hi = min(1.0, max(all_acc) + 0.02)
    if y_hi <= y_lo:
        y_lo, y_hi = 0.0, 1.0

    def xpx(epoch):
        """Map epoch index (0-based) to x pixel coordinate."""
        return ml + int(epoch / max(n - 1, 1) * pw)

    def ypx(acc):
        """Map accuracy value to y pixel coordinate."""
        return mt + int((1.0 - (acc - y_lo) / (y_hi - y_lo)) * ph)

    # ── Canvas ─────────────────────────────────────────────────────
    img = Image.new('RGB', (W, H), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    # ── Title ──────────────────────────────────────────────────────
    title = 'Generalisation Gap: Training vs Validation Accuracy'
    tb = draw.textbbox((0, 0), title, font=f_title)
    draw.text(((W - tb[2] + tb[0]) // 2, 12), title,
              fill=(0, 0, 0), font=f_title)

    # ── Horizontal grid + Y ticks ──────────────────────────────────
    n_yticks = 8
    for i in range(n_yticks + 1):
        val = y_lo + i * (y_hi - y_lo) / n_yticks
        py = ypx(val)
        draw.line([(ml, py), (ml + pw, py)], fill=(225, 225, 225))
        lbl = f'{val:.2f}'
        bb = draw.textbbox((0, 0), lbl, font=f_tick)
        draw.text((ml - (bb[2] - bb[0]) - 8, py - (bb[3] - bb[1]) // 2),
                  lbl, fill=(100, 100, 100), font=f_tick)

    # ── Vertical grid + X ticks ────────────────────────────────────
    step = max(1, n // 10)
    ticks = list(range(0, n, step))
    if n - 1 not in ticks:
        ticks.append(n - 1)
    for ep in ticks:
        px = xpx(ep)
        draw.line([(px, mt), (px, mt + ph)], fill=(225, 225, 225))
        lbl = str(ep + 1)
        bb = draw.textbbox((0, 0), lbl, font=f_tick)
        draw.text((px - (bb[2] - bb[0]) // 2, mt + ph + 6),
                  lbl, fill=(100, 100, 100), font=f_tick)

    # ── Axes ───────────────────────────────────────────────────────
    draw.line([(ml, mt + ph), (ml + pw, mt + ph)], fill=(0, 0, 0), width=2)
    draw.line([(ml, mt), (ml, mt + ph)], fill=(0, 0, 0), width=2)

    # ── X-axis label ───────────────────────────────────────────────
    xl = 'Epoch'
    xb = draw.textbbox((0, 0), xl, font=f_label)
    draw.text((ml + pw // 2 - (xb[2] - xb[0]) // 2, H - 28),
              xl, fill=(0, 0, 0), font=f_label)

    # ── Y-axis label (rotated) ─────────────────────────────────────
    yl_img = Image.new('RGBA', (150, 25), (255, 255, 255, 0))
    yl_draw = ImageDraw.Draw(yl_img)
    yl_draw.text((0, 0), 'Accuracy', fill=(0, 0, 0, 255), font=f_label)
    yl_rot = yl_img.rotate(90, expand=True)
    img.paste(yl_rot, (2, mt + ph // 2 - yl_rot.height // 2), yl_rot)

    # ── Data lines ─────────────────────────────────────────────────
    for _, data, color in series:
        pts = [(xpx(i), ypx(data[i])) for i in range(len(data))]
        for j in range(len(pts) - 1):
            draw.line([pts[j], pts[j + 1]], fill=color, width=2)

    # ── Legend ──────────────────────────────────────────────────────
    lx = ml + pw + 18
    ly = mt + 20
    for i, (name, _, color) in enumerate(series):
        cy = ly + i * 28
        draw.rectangle([(lx, cy + 2), (lx + 22, cy + 14)], fill=color)
        draw.text((lx + 28, cy), name, fill=(0, 0, 0), font=f_tick)

    img.save(filename)
    print(f'Saved {filename}')


# ── Analysis ───────────────────────────────────────────────────────────────

def print_analysis(bl_train, bl_val, bl_test, rg_train, rg_val, rg_test,
                   n_epochs):
    """Print ~500-word technical analysis referencing actual measured results.

    Args:
        bl_train (float): Baseline training accuracy.
        bl_val (float): Baseline validation accuracy.
        bl_test (float): Baseline test accuracy.
        rg_train (float): Regularised training accuracy.
        rg_val (float): Regularised validation accuracy.
        rg_test (float): Regularised test accuracy.
        n_epochs (int): Number of training epochs.
    """
    bl_gap = bl_train - bl_val
    rg_gap = rg_train - rg_val

    print(f"""
{'=' * 72}
TECHNICAL ANALYSIS: GENERALISATION GAP AND BIAS-VARIANCE TRADE-OFF
{'=' * 72}

Two identical 5-layer CNNs (3 Conv2d + 2 Linear, bias=True, no BatchNorm;
~620k trainable parameters) were trained on CIFAR-10 for {n_epochs} epochs
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
Baseline: Train {bl_train:.2%} | Val {bl_val:.2%} | Test {bl_test:.2%} \
| Gap {bl_gap * 100:.2f}pp. Regularised: Train {rg_train:.2%} | Val \
{rg_val:.2%} | Test {rg_test:.2%} | Gap {rg_gap * 100:.2f}pp.
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
upward, settling near {rg_val:.2%} regularised vs {bl_val:.2%} baseline.

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

GenAI Error Correction: Claude initially drafted a technical analysis that \
overstated trends from raw data points and entirely missed the \
regularisation dynamics of epochs 8-17. Furthermore, it incorrectly \
recommended a batch size of 256 and 50 epochs, and advised against \
re-training when I sought to improve the model. I rejected this advice, \
switched to a batch size of 128 and 75 epochs for better generalisation \
and visual clarity, re-trained the model, and manually rewrote the \
analysis to accurately reflect the true learning trajectory.
{'=' * 72}
""")


# ── Main ───────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    train_loader, val_loader, test_loader = load_data()

    # ── Load and evaluate baseline ─────────────────────────────────
    baseline = CNN(num_classes=NUM_CLASSES, use_dropout=False)
    baseline.load_state_dict(
        torch.load('baseline.pth', map_location='cpu', weights_only=True),
    )
    baseline.eval()
    bl_train = evaluate(baseline, train_loader)
    bl_val = evaluate(baseline, val_loader)
    bl_test = evaluate(baseline, test_loader)

    # ── Load and evaluate regularised ──────────────────────────────
    regularised = CNN(num_classes=NUM_CLASSES, use_dropout=True)
    regularised.load_state_dict(
        torch.load('regularised.pth', map_location='cpu', weights_only=True),
    )
    regularised.eval()
    rg_train = evaluate(regularised, train_loader)
    rg_val = evaluate(regularised, val_loader)
    rg_test = evaluate(regularised, test_loader)

    print(f'Baseline    -> Train: {bl_train:.4f}  '
          f'Val: {bl_val:.4f}  Test: {bl_test:.4f}')
    print(f'Regularised -> Train: {rg_train:.4f}  '
          f'Val: {rg_val:.4f}  Test: {rg_test:.4f}')

    # ── Generate plot from training histories ──────────────────────
    histories = torch.load('histories.pt', map_location='cpu',
                           weights_only=False)
    draw_plot(histories)

    # ── Print analysis ─────────────────────────────────────────────
    n_epochs = len(histories['baseline_history']['train_acc'])
    print_analysis(bl_train, bl_val, bl_test,
                   rg_train, rg_val, rg_test, n_epochs)
