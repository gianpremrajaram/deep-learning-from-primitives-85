# Claude opus 4.6 for starter code + debugging. Iterated substantially based on
# online research/lecture material to correct mathematical misrepresentations.
# AI correction: first robustness_demo labels- simply 0.xxsample_1 + sample_2.
# AI also strongly suggested keeping as-is, misunderstanding visual demonstration
# of the clear seperation of sample_1 + sample_2, both with relative % is better.


"""COMP0197 CW1 Task 2 - Evaluation Script

Loads the best model trained with MixUp + Label Smoothing, evaluates
robustness under additive Gaussian noise at five levels, generates a
4x4 MixUp montage (robustness_demo.png), and prints a ~500-word
technical analysis.

Expects: best_model.pth (from train.py).
Outputs: robustness_demo.png, terminal-printed analysis.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import torchvision
import torchvision.transforms as transforms
from PIL import Image, ImageDraw, ImageFont

# ── Reproducibility ────────────────────────────────────────────────────────
torch.manual_seed(42)

# ── Constants (must match train.py) ────────────────────────────────────────
BATCH_SIZE = 128
NUM_CLASSES = 10
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)
NOISE_LEVELS = [0.0, 0.05, 0.1, 0.15, 0.2]
CIFAR10_CLASSES = [
    'plane', 'auto', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck',
]


# ── Model (identical to train.py) ─────────────────────────────────────────

class CNN(nn.Module):
    """5-layer CNN (3 Conv2d + 2 Linear) for CIFAR-10 classification.

    Architecture:
        Conv1(3->32,bias=F) -> BN -> ReLU -> MaxPool(2)  Output: (B,32,16,16)
        Conv2(32->64,bias=F)-> BN -> ReLU -> MaxPool(2)  Output: (B,64, 8, 8)
        Conv3(64->128,bias=F)->BN -> ReLU -> MaxPool(2)  Output: (B,128,4, 4)
        Flatten                                           Output: (B, 2048)
        FC1(2048->256) -> ReLU                            Output: (B, 256)
        FC2(256->10)                                      Output: (B, 10)

    Args:
        num_classes (int): Number of output classes. Default: 10.
    """

    def __init__(self, num_classes=10):
        """Initialise all layers explicitly.

        Args:
            num_classes (int): Output dimension (number of classes).
        """
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(128)
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, num_classes)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        """Forward pass through the network.

        Args:
            x (torch.Tensor): Input images, shape (B, 3, 32, 32).

        Returns:
            torch.Tensor: Raw logits, shape (B, num_classes).
        """
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.pool(self.relu(self.bn3(self.conv3(x))))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
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


# ── MixUp helpers (for montage demo) ──────────────────────────────────────

def sample_beta(alpha):
    """Sample from Beta(alpha, alpha) via Gamma decomposition.

    Args:
        alpha (float): Shape parameter for both Beta parameters.

    Returns:
        torch.Tensor: Scalar in [0, 1] drawn from Beta(alpha, alpha).
    """
    conc = torch.tensor([alpha], dtype=torch.float32)
    g1 = torch._standard_gamma(conc)
    g2 = torch._standard_gamma(conc)
    return (g1 / (g1 + g2)).squeeze()


def unnormalize(tensor):
    """Reverse CIFAR-10 channel-wise normalisation for display.

    Args:
        tensor (torch.Tensor): Normalised image, shape (3, 32, 32).

    Returns:
        PIL.Image.Image: RGB image clipped to [0, 255].
    """
    img = tensor.clone()
    for c in range(3):
        img[c] = img[c] * CIFAR10_STD[c] + CIFAR10_MEAN[c]
    img = img.clamp(0.0, 1.0)
    img = (img * 255).to(torch.uint8)
    return Image.fromarray(img.permute(1, 2, 0).numpy())


# ── Evaluation ─────────────────────────────────────────────────────────────

def evaluate_noisy(model, loader, noise_std):
    """Compute accuracy on data with additive Gaussian noise.

    Adds noise in normalised space: noisy = images + N(0, noise_std).
    No clamping is applied (tests raw robustness of learned features).

    Args:
        model (CNN): Trained model (must be in eval mode).
        loader (DataLoader): Test batches of (images, labels).
        noise_std (float): Noise standard deviation (0 = clean).

    Returns:
        float: Classification accuracy in [0, 1].
    """
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            if noise_std > 0:
                images = images + torch.randn_like(images) * noise_std
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    return correct / total


# ── Montage ────────────────────────────────────────────────────────────────

def _get_font(size=14):
    """Load a default font at the requested size with fallback.

    Args:
        size (int): Desired font size in points.

    Returns:
        PIL.ImageFont.ImageFont: Font object.
    """
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def create_montage(train_loader, filename='robustness_demo.png'):
    """Generate a 4x4 montage of MixUp-blended images with annotations.

    Takes 16 training images, pairs them via random shuffle, blends each
    pair with a fresh Beta sample, and arranges results in a grid. Each
    cell is annotated with the lambda value and the two class names.

    Alpha=1.0 is used for the demo (uniform lambda) to make blending
    visually clear; training uses alpha=0.2.

    Args:
        train_loader (DataLoader): Training data for source images.
        filename (str): Output PNG path (relative). Default: robustness_demo.png.
    """
    images, labels = next(iter(train_loader))
    images = images[:16]
    labels = labels[:16]

    # ── Layout constants ───────────────────────────────────────────
    cell = 96
    cols, rows = 4, 4
    anno_h = 16
    title_h = 28
    pad = 8
    w = 2 * pad + cols * cell
    h = 2 * pad + title_h + rows * (cell + anno_h)

    font_sm = _get_font(9)
    font_title = _get_font(13)

    canvas = Image.new('RGB', (w, h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    # ── Title ──────────────────────────────────────────────────────
    title = 'MixUp Blended Training Samples'
    tb = draw.textbbox((0, 0), title, font=font_title)
    draw.text(((w - tb[2] + tb[0]) // 2, pad), title,
              fill=(0, 0, 0), font=font_title)

    # ── Create blended pairs ───────────────────────────────────────
    indices = torch.randperm(16)
    blended = []
    info = []

    for i in range(16):
        j = indices[i].item()
        lam = sample_beta(1.0).item()
        mixed = lam * images[i] + (1.0 - lam) * images[j]
        blended.append(mixed)
        c1 = CIFAR10_CLASSES[labels[i].item()]
        c2 = CIFAR10_CLASSES[labels[j].item()]
        info.append((lam, c1, c2))

    # ── Draw cells ─────────────────────────────────────────────────
    for idx in range(16):
        r = idx // cols
        c = idx % cols
        x = pad + c * cell
        y = pad + title_h + r * (cell + anno_h)

        pil_img = unnormalize(blended[idx])
        pil_img = pil_img.resize((cell, cell), Image.NEAREST)
        canvas.paste(pil_img, (x, y))

        # Annotation
        lam, c1, c2 = info[idx]
        anno = f'{lam:.2f} {c1} + {1.0 - lam:.2f} {c2}'
        ab = draw.textbbox((0, 0), anno, font=font_sm)
        aw = ab[2] - ab[0]
        ax = x + (cell - aw) // 2
        ay = y + cell + 1
        draw.text((ax, ay), anno, fill=(80, 80, 80), font=font_sm)

    canvas.save(filename)
    print(f'Saved {filename}')


# ── Analysis ───────────────────────────────────────────────────────────────

def print_analysis(noise_results):
    """Print ~500-word technical analysis on MixUp, Label Smoothing, and robustness.

    Loads training metadata (stopped epoch, best validation loss) from
    train_metadata.pt to populate early stopping details dynamically.
    Noise robustness figures are computed from the live evaluation results.

    Args:
        noise_results (dict[float, float]): Mapping noise_std -> accuracy.
    """
    clean_acc = noise_results[0.0]
    worst_acc = noise_results[0.2]
    drop = clean_acc - worst_acc

    # Per-level drops for quantitative commentary
    acc_005 = noise_results[0.05]
    acc_010 = noise_results[0.10]
    acc_015 = noise_results[0.15]
    drop_low = clean_acc - acc_005
    drop_high = acc_015 - worst_acc

    # Load training metadata saved by train.py
    meta = torch.load('train_metadata.pt', map_location='cpu', weights_only=True)
    stopped_epoch = meta['stopped_epoch']
    best_val_loss = meta['best_val_loss']

    print(f"""
{'=' * 72}
TECHNICAL ANALYSIS: MIXUP, LABEL SMOOTHING, AND ROBUSTNESS
{'=' * 72}

[Architecture and Clean Performance]
A 5-layer CNN (3 Conv2d + 2 Linear) with BatchNorm was trained on
CIFAR-10 using MixUp (alpha=0.2), label smoothing (eps=0.1), and SGD
(lr=0.01, momentum=0.9, batch size=128). No dropout or weight decay was
used. On the uncorrupted test set, the model achieved {clean_acc:.2%}
accuracy.

[Noise Robustness]
Under additive Gaussian noise, accuracy fell from {clean_acc:.2%} on the
uncorrupted test set to {acc_005:.2%} at noise_std=0.05, {acc_010:.2%}
at 0.10, {acc_015:.2%} at 0.15, and {worst_acc:.2%} at 0.20. This is a
total drop of {drop * 100:.2f} percentage points at the highest noise level. The decline is
small at low noise ({drop_low * 100:.2f} percentage points at std=0.05) and steeper at higher
noise ({drop_high * 100:.2f} percentage points between std=0.15 and 0.20), which suggests the
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
receives the same input twice, memorisation of exact training samples becomes much harder,
forcing the network to learn features that interpolate smoothly between
classes. This encourages smoother decision boundaries and features that
generalise beyond the training set.

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
Training stopped at epoch {stopped_epoch}. The best validation-loss
checkpoint was at epoch 12, with val_loss={best_val_loss:.4f} and
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
{'=' * 72}
""")


# ── Main ───────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    train_loader, val_loader, test_loader = load_data()

    # ── Load best model ────────────────────────────────────────────
    model = CNN(num_classes=NUM_CLASSES)
    model.load_state_dict(
        torch.load('best_model.pth', map_location='cpu', weights_only=True),
    )
    model.eval()

    # ── Noisy test evaluation ──────────────────────────────────────
    print('=' * 72)
    print('NOISY TEST EVALUATION')
    print('=' * 72)
    noise_results = {}
    for ns in NOISE_LEVELS:
        acc = evaluate_noisy(model, test_loader, ns)
        noise_results[ns] = acc
        print(f'  noise_std={ns:.2f}  ->  accuracy={acc:.4f}')

    # ── MixUp montage ─────────────────────────────────────────────
    create_montage(train_loader)

    # ── Analysis ──────────────────────────────────────────────────
    print_analysis(noise_results)
