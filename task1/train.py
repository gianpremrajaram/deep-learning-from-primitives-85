# Claude (opus 4.6) for starter code & later debugging. Iterated substantially
# based on online research.
# AI mistake corrected: epoch number and batch size, initial advice was 50&256.
# after reviewing results I opted for increasing epoch to 75 for better visuals
# 128 yielded resulted in better performance (I knew generalisation to unseen
# was the main challenge). Suspect 256 initially recommended based on general
# substack/reddit recommendations which were not well-suited to current arch +
# CIFAR dataset.

"""COMP0197 CW1 Task 1 - Training Script

Trains baseline and regularised CNN models on CIFAR-10 to investigate the
generalisation gap and bias-variance trade-off.

Baseline: no regularisation (no dropout, weight_decay=0).
Regularised: Dropout(0.3) + weight_decay=1e-4 in SGD.
Both share the same 5-layer architecture (3 Conv2d + 2 Linear) with
toggleable regularisation, ensuring a clean controlled experiment.

Outputs: baseline.pth, regularised.pth, histories.pt
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
NUM_EPOCHS = 75
BATCH_SIZE = 128
LEARNING_RATE = 0.01
MOMENTUM = 0.9
NUM_CLASSES = 10
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


# ── Model ──────────────────────────────────────────────────────────────────

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
    """Download CIFAR-10 and create train/val data loaders.

    Applies ToTensor + channel-wise normalisation (no augmentation).
    Splits 50k training images into 45k train / 5k val with a fixed
    generator seed, independent of the global torch seed.

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


# ── Training loop ──────────────────────────────────────────────────────────

def train_model(model, train_loader, val_loader, optimizer, criterion,
                num_epochs):
    """Train for a fixed number of epochs, recording per-epoch metrics.

    Args:
        model (CNN): Network to train.
        train_loader (DataLoader): Training batches (B, 3, 32, 32).
        val_loader (DataLoader): Validation batches.
        optimizer (torch.optim.Optimizer): SGD optimiser.
        criterion (nn.Module): Loss function (CrossEntropyLoss).
        num_epochs (int): Total training epochs.

    Returns:
        dict: History with keys 'train_acc', 'val_acc', 'train_loss',
              'val_loss', each a list[float] of length num_epochs.
    """
    history = {
        'train_acc': [], 'val_acc': [],
        'train_loss': [], 'val_loss': [],
    }

    for epoch in range(num_epochs):
        # ── Train phase ────────────────────────────────────────────
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_acc = correct / total
        train_loss = running_loss / len(train_loader)

        # ── Validation phase ───────────────────────────────────────
        model.eval()
        val_loss_sum = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss_sum += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        val_acc = val_correct / val_total
        val_loss = val_loss_sum / len(val_loader)

        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)

        print(f'  Epoch [{epoch + 1:3d}/{num_epochs}]  '
              f'Train Loss: {train_loss:.4f}  Acc: {train_acc:.4f}  |  '
              f'Val Loss: {val_loss:.4f}  Acc: {val_acc:.4f}')

    return history


# ── Main ───────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    train_loader, val_loader = load_data()
    criterion = nn.CrossEntropyLoss()

    # ── Baseline (no regularisation) ───────────────────────────────
    print('=' * 72)
    print('BASELINE MODEL  (no dropout, weight_decay=0)')
    print('=' * 72)
    torch.manual_seed(42)
    baseline = CNN(num_classes=NUM_CLASSES, use_dropout=False)
    baseline_opt = optim.SGD(
        baseline.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM,
    )
    baseline_hist = train_model(
        baseline, train_loader, val_loader, baseline_opt, criterion,
        NUM_EPOCHS,
    )
    torch.save(baseline.state_dict(), 'baseline.pth')

    # ── Regularised (Dropout + L2) ─────────────────────────────────
    print('\n' + '=' * 72)
    print('REGULARISED MODEL  (dropout=0.3, weight_decay=1e-4)')
    print('=' * 72)
    torch.manual_seed(42)
    regularised = CNN(num_classes=NUM_CLASSES, use_dropout=True)
    reg_opt = optim.SGD(
        regularised.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM,
        weight_decay=1e-4,
    )
    reg_hist = train_model(
        regularised, train_loader, val_loader, reg_opt, criterion,
        NUM_EPOCHS,
    )
    torch.save(regularised.state_dict(), 'regularised.pth')

    # ── Save training histories ────────────────────────────────────
    torch.save({
        'baseline_history': baseline_hist,
        'regularised_history': reg_hist,
    }, 'histories.pt')

    print('\nDone. Saved: baseline.pth, regularised.pth, histories.pt')
