"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import config


class CNNClassifier(nn.Module):
    """
    Operates on 1-D diffusion-refined feature vectors. Implemented as a
    1D-conv + dense stack so it still benefits from local feature
    interactions (as a "CNN" implies) while accepting a flat feature vector
    input.
    """

    def __init__(self, feature_dim=config.FEATURE_DIM, num_classes=config.NUM_CLASSES):
        super().__init__()

        self.conv_block = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),

            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
        )

        conv_out_dim = 64 * (feature_dim // 4)

        self.classifier = nn.Sequential(
            nn.Linear(conv_out_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        # x: [batch, feature_dim] -> add channel dim for Conv1d: [batch, 1, feature_dim]
        x = x.unsqueeze(1)
        x = self.conv_block(x)
        x = x.view(x.size(0), -1)
        logits = self.classifier(x)
        return logits  # raw logits; see module docstring

    def predict_proba(self, x):
        logits = self.forward(x)
        return F.softmax(logits, dim=-1)


def train_cnn(refined_features, labels, val_features=None, val_labels=None,
              device=config.DEVICE, verbose=True):
    """
    refined_features : FloatTensor [N, feature_dim] -- diffusion-refined features
    labels           : LongTensor [N] -- integer class labels (0..NUM_CLASSES-1)
    Returns the trained CNNClassifier.
    """
    dataset = TensorDataset(refined_features, labels)
    loader = DataLoader(dataset, batch_size=config.CNN_BATCH_SIZE, shuffle=True, drop_last=True)

    model = CNNClassifier().to(device)
    criterion = nn.CrossEntropyLoss()  # equivalent to Softmax + categorical cross-entropy
    optimizer = optim.Adam(model.parameters(), lr=config.CNN_LR)

    for epoch in range(config.CNN_EPOCHS):
        model.train()
        epoch_loss, correct, total = 0.0, 0, 0

        for feats, y in loader:
            feats, y = feats.to(device), y.to(device)

            optimizer.zero_grad()
            logits = model(feats)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * feats.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += feats.size(0)

        train_acc = 100.0 * correct / total
        msg = (f"[CNN] Epoch {epoch + 1}/{config.CNN_EPOCHS} "
               f"- loss: {epoch_loss / total:.4f} - train_acc: {train_acc:.2f}%")

        if val_features is not None and val_labels is not None:
            val_acc = evaluate_cnn(model, val_features, val_labels, device)
            msg += f" - val_acc: {val_acc:.2f}%"

        if verbose:
            print(msg)

    return model


@torch.no_grad()
def evaluate_cnn(model, features, labels, device=config.DEVICE):
    model.eval()
    features, labels = features.to(device), labels.to(device)
    logits = model(features)
    preds = logits.argmax(dim=1)
    acc = 100.0 * (preds == labels).float().mean().item()
    return acc
