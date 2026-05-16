# # Claude (opus 4.6) for starter code & later debugging. Iterated substantially based on
# online research/lecture material to correct training direction.


"""COMP0197 CW1 Task 2 - Training Script

Trains a CNN on CIFAR-10 with custom MixUp data augmentation and Label
Smoothing cross-entropy loss, plus manual early stopping. All three
techniques are implemented from basic tensor operations.

MixUp (alpha=0.2): blends image pairs and one-hot labels per batch using
lambda sampled from Beta(alpha, alpha) via Gamma decomposition.

Label Smoothing (epsilon=0.1): softens targets before computing cross-entropy
as -(smooth_targets * log_softmax(logits)).sum(dim=1).mean().

Early Stopping (patience=10): saves best checkpoint on validation loss
improvement (strict <), reloads best weights after training loop ends.

Outputs: best_model.pth
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torchvision
import torchvision.transforms as transforms

# ── Reproducibility ────────────────────────────────────────────────────────
torch.manual_seed(42)

# ── Hyperparameters ────────────────────────────────────────────────────────
NUM_EPOCHS = 100
BATCH_SIZE = 128
LEARNING_RATE = 0.01
MOMENTUM = 0.9
NUM_CLASSES = 10
MIXUP_ALPHA = 0.2
SMOOTHING_EPSILON = 0.1
PATIENCE = 10
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


# ── Model ──────────────────────────────────────────────────────────────────

class CNN(nn.Module):
    """5-layer CNN (3 Conv2d + 2 Linear) for CIFAR-10 classification.

    Architecture:
        Conv1(3->32,bias=F) -> BN -> ReLU -> MaxPool(2)  Output: (B,32,16,16)
        Conv2(32->64,bias=F)-> BN -> ReLU -> MaxPool(2)  Output: (B,64, 8, 8)
        Conv3(64->128,bias=F)->BN -> ReLU -> MaxPool(2)  Output: (B,128,4, 4)
        Flatten                                           Output: (B, 2048)
        FC1(2048->256) -> ReLU                            Output: (B, 256)
        FC2(256->10)                                      Output: (B, 10)

    No dropout: Task 2 regularisation is purely MixUp + Label Smoothing +
    Early Stopping. Conv bias=False because BatchNorm absorbs the bias.

    Args:
        num_classes (int): Number of output classes. Default: 10.
    """

    def __init__(self, num_classes=10):
        """Initialise all layers explicitly.

        Args:
            num_classes (int): Output dimension (number of classes).
        """
        super().__init__()
        # Layer 1: Conv2d  (bias=False: BatchNorm absorbs the bias)
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)

        # Layer 2: Conv2d
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)

        # Layer 3: Conv2d
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(128)

        # Layer 4: Linear
        self.fc1 = nn.Linear(128 * 4 * 4, 256)

        # Layer 5: Linear
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
        x = self.pool(self.relu(self.bn1(self.conv1(x))))   # (B,32,16,16)
        x = self.pool(self.relu(self.bn2(self.conv2(x))))   # (B,64, 8, 8)
        x = self.pool(self.relu(self.bn3(self.conv3(x))))   # (B,128,4, 4)
        x = x.view(x.size(0), -1)                           # (B, 2048)
        x = self.relu(self.fc1(x))                           # (B, 256)
        x = self.fc2(x)                                     # (B, 10)
        return x


# ── MixUp ──────────────────────────────────────────────────────────────────

def sample_beta(alpha):
    """Sample from Beta(alpha, alpha) via Gamma decomposition.

    Uses the identity: if g1 ~ Gamma(a,1) and g2 ~ Gamma(a,1),
    then g1 / (g1 + g2) ~ Beta(a, a). Avoids torch.distributions.

    Args:
        alpha (float): Shape parameter for both Beta parameters.

    Returns:
        torch.Tensor: Scalar in [0, 1] drawn from Beta(alpha, alpha).
    """
    conc = torch.tensor([alpha], dtype=torch.float32)
    g1 = torch._standard_gamma(conc)
    g2 = torch._standard_gamma(conc)
    return (g1 / (g1 + g2)).squeeze()


def mixup_batch(images, labels, alpha, num_classes):
    """Apply MixUp to a mini-batch: blend both images and one-hot labels.

    Pairs are formed by a random permutation of the batch. A single
    lambda is sampled per batch from Beta(alpha, alpha).

    Args:
        images (torch.Tensor): Batch of images, shape (B, 3, 32, 32).
        labels (torch.Tensor): Integer class labels, shape (B,).
        alpha (float): Beta distribution shape parameter.
        num_classes (int): Number of classes for one-hot encoding.

    Returns:
        tuple[torch.Tensor, torch.Tensor]:
            mixed_images (B, 3, 32, 32) and mixed_labels (B, num_classes).
    """
    lam = sample_beta(alpha)
    indices = torch.randperm(images.size(0))

    mixed_images = lam * images + (1.0 - lam) * images[indices]

    y1 = torch.zeros(images.size(0), num_classes).scatter_(
        1, labels.unsqueeze(1), 1.0,
    )
    y2 = torch.zeros(images.size(0), num_classes).scatter_(
        1, labels[indices].unsqueeze(1), 1.0,
    )
    mixed_labels = lam * y1 + (1.0 - lam) * y2

    return mixed_images, mixed_labels


# ── Label Smoothing Cross-Entropy ──────────────────────────────────────────

def smooth_ce_loss(logits, targets, epsilon, num_classes):
    """Cross-entropy loss with label smoothing, handling int or soft targets.

    When targets are integer class indices (1D): converts to one-hot, then
    applies smoothing. When targets are already soft vectors (2D, from
    MixUp): applies smoothing directly on the soft distribution.

    Smoothing: smooth = (1 - epsilon) * targets_as_probs + epsilon / K
    Loss:      -(smooth * log_softmax(logits, dim=1)).sum(dim=1).mean()

    With epsilon=0 this reduces to standard cross-entropy.

    Args:
        logits (torch.Tensor): Raw model output, shape (B, num_classes).
        targets (torch.Tensor): Integer labels (B,) or soft vectors
            (B, num_classes).
        epsilon (float): Smoothing factor in [0, 1].
        num_classes (int): Number of classes.

    Returns:
        torch.Tensor: Scalar mean loss.
    """
    if targets.dim() == 1:
        target_probs = torch.zeros_like(logits).scatter_(
            1, targets.unsqueeze(1), 1.0,
        )
    else:
        target_probs = targets

    smooth = (1.0 - epsilon) * target_probs + epsilon / num_classes
    # Compute log-softmax from scratch using the log-sum-exp trick.
    # Subtracting the maximum logit (m) before exponentiation guarantees
    # the largest exponentiated value is e^0 = 1, alleviating the risk
    # of float32 overflow (which produces NaN loss) if logits exceed ~88.
    m = logits.max(dim=1, keepdim=True).values
    log_probs = logits - m - (logits - m).exp().sum(dim=1, keepdim=True).log()
    return -(smooth * log_probs).sum(dim=1).mean()


# ── Data ───────────────────────────────────────────────────────────────────

def load_data():
    """Download CIFAR-10 and split into train (45k) / val (5k) loaders.

    Uses ToTensor + channel-wise normalisation. Split uses a dedicated
    generator seed (42), independent of the global torch seed.

    Returns:
        tuple[DataLoader, DataLoader]: (train_loader, val_loader).
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    full_train = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform,
    )

    generator = torch.Generator().manual_seed(42)
    train_set, val_set = random_split(
        full_train, [45000, 5000], generator=generator,
    )

    train_loader = DataLoader(
        train_set, batch_size=BATCH_SIZE, shuffle=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_set, batch_size=BATCH_SIZE, shuffle=False, drop_last=True,
    )
    return train_loader, val_loader


# ── Training with early stopping ───────────────────────────────────────────

def train_model(model, train_loader, val_loader):
    """Train with MixUp + Label Smoothing and manual early stopping.

    Training loop applies MixUp to each batch, then computes smoothed CE
    loss on the blended soft labels. Validation uses clean (unmixed) data
    with standard CE (epsilon=0) for the early stopping signal.

    Best checkpoint is saved on strict validation loss improvement (<).
    After the loop ends, the best checkpoint is always reloaded.

    Args:
        model (CNN): Network to train.
        train_loader (DataLoader): Training batches.
        val_loader (DataLoader): Validation batches.
    """
    optimizer = optim.SGD(
        model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM,
    )

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(NUM_EPOCHS):
        # ── Train phase (MixUp + Label Smoothing) ──────────────────
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            mixed_images, mixed_labels = mixup_batch(
                images, labels, MIXUP_ALPHA, NUM_CLASSES,
            )

            optimizer.zero_grad()
            outputs = model(mixed_images)
            loss = smooth_ce_loss(
                outputs, mixed_labels, SMOOTHING_EPSILON, NUM_CLASSES,
            )
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)

        # ── Validation phase (clean data, standard CE) ─────────────
        model.eval()
        val_loss_sum = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                outputs = model(images)
                loss = smooth_ce_loss(
                    outputs, labels, epsilon=0.0, num_classes=NUM_CLASSES,
                )
                val_loss_sum += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        val_loss = val_loss_sum / len(val_loader)
        val_acc = val_correct / val_total

        print(f'  Epoch [{epoch + 1:3d}/{NUM_EPOCHS}]  '
              f'Train Loss: {train_loss:.4f}  |  '
              f'Val Loss: {val_loss:.4f}  Val Acc: {val_acc:.4f}',
              end='')

        # ── Early stopping ─────────────────────────────────────────
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'best_model.pth')
            print(f'  *checkpoint*')
        else:
            patience_counter += 1
            print(f'  (patience {patience_counter}/{PATIENCE})')

        if patience_counter >= PATIENCE:
            print(f'\nEarly stopping at epoch {epoch + 1}')
            break

    # Reload best checkpoint
    model.load_state_dict(
        torch.load('best_model.pth', map_location='cpu', weights_only=True),
    )
    print(f'\nReloaded best checkpoint (val_loss={best_val_loss:.4f})')

    # Persist early-stopping metadata for task.py
    torch.save({
        'stopped_epoch': epoch + 1,
        'best_val_loss': best_val_loss,
    }, 'train_metadata.pt')
    print('Done. Saved: best_model.pth, train_metadata.pt')


# ── Main ───────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    train_loader, val_loader = load_data()

    print('=' * 72)
    print('TASK 2: MixUp + Label Smoothing + Early Stopping')
    print(f'  MixUp alpha={MIXUP_ALPHA}  |  Smoothing eps={SMOOTHING_EPSILON}')
    print(f'  Patience={PATIENCE}  |  Max epochs={NUM_EPOCHS}')
    print(f'  SGD lr={LEARNING_RATE}, momentum={MOMENTUM}, weight_decay=0')
    print('=' * 72)

    torch.manual_seed(42)
    model = CNN(num_classes=NUM_CLASSES)
    train_model(model, train_loader, val_loader)
