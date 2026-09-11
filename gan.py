"""
GAN stage of the pipeline.

The Discriminator is trained adversarially (standard DCGAN-style setup) on
the spectrogram images. Once trained, the Discriminator's penultimate layer
is used as a fixed feature extractor: for every input spectrogram it
produces a FEATURE_DIM-dimensional vector, which is passed on to the
Diffusion stage for further refinement.

Activation : LeakyReLU (as specified)
Optimizer  : Adam, lr = 0.0002
Loss       : Binary Cross Entropy
Epochs     : 30
Batch size : 16
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import config


def _conv_out_size(size, n_downsamples):
    return size // (2 ** n_downsamples)


class Generator(nn.Module):
    """Simple transpose-conv generator, used only to provide the adversarial
    training signal for the Discriminator (which is the component we
    actually reuse for feature extraction)."""

    def __init__(self, latent_dim=config.GAN_LATENT_DIM,
                 img_channels=config.IMG_CHANNELS, img_size=config.IMG_SIZE):
        super().__init__()
        self.init_size = img_size // 8  # 3 upsampling stages of factor 2
        self.fc = nn.Linear(latent_dim, 128 * self.init_size ** 2)

        self.net = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, momentum=0.8),
            nn.LeakyReLU(config.GAN_LEAKY_SLOPE, inplace=True),

            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, momentum=0.8),
            nn.LeakyReLU(config.GAN_LEAKY_SLOPE, inplace=True),

            nn.Upsample(scale_factor=2),
            nn.Conv2d(64, img_channels, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, z):
        out = self.fc(z)
        out = out.view(out.size(0), 128, self.init_size, self.init_size)
        return self.net(out)


class Discriminator(nn.Module):
    """
    Convolutional discriminator. `forward(x)` returns (validity, features):
        - validity: real/fake probability (used for adversarial training)
        - features: FEATURE_DIM-dim vector from the penultimate layer
                    (used downstream as the GAN-extracted feature vector)
    """

    def __init__(self, img_channels=config.IMG_CHANNELS, img_size=config.IMG_SIZE,
                 feature_dim=config.FEATURE_DIM):
        super().__init__()

        def block(in_ch, out_ch, bn=True):
            layers = [nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1)]
            if bn:
                layers.append(nn.BatchNorm2d(out_ch, momentum=0.8))
            layers.append(nn.LeakyReLU(config.GAN_LEAKY_SLOPE, inplace=True))
            layers.append(nn.Dropout2d(0.25))
            return layers

        self.conv_blocks = nn.Sequential(
            *block(img_channels, 32, bn=False),  # /2
            *block(32, 64),                      # /4
            *block(64, 128),                      # /8
        )

        ds_size = _conv_out_size(img_size, 3)
        flat_dim = 128 * ds_size * ds_size

        # Penultimate layer -> this is the "intermediate feature" we extract
        self.feature_layer = nn.Sequential(
            nn.Linear(flat_dim, feature_dim),
            nn.LeakyReLU(config.GAN_LEAKY_SLOPE, inplace=True),
        )
        self.validity_layer = nn.Sequential(
            nn.Linear(feature_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        out = self.conv_blocks(x)
        out = out.view(out.size(0), -1)
        features = self.feature_layer(out)
        validity = self.validity_layer(features)
        return validity, features


def train_gan(dataset, device=config.DEVICE, verbose=True):
    """
    Trains the GAN adversarially on `dataset` (labels are ignored -- GAN
    training here is unsupervised). Returns the trained Discriminator,
    which doubles as the feature extractor for the next stage.
    """
    loader = DataLoader(dataset, batch_size=config.GAN_BATCH_SIZE, shuffle=True, drop_last=True)

    G = Generator().to(device)
    D = Discriminator().to(device)

    criterion = nn.BCELoss()
    opt_G = optim.Adam(G.parameters(), lr=config.GAN_LR, betas=(0.5, 0.999))
    opt_D = optim.Adam(D.parameters(), lr=config.GAN_LR, betas=(0.5, 0.999))

    for epoch in range(config.GAN_EPOCHS):
        epoch_d_loss, epoch_g_loss = 0.0, 0.0
        n_batches = 0

        for imgs, _ in loader:
            imgs = imgs.to(device)
            bsz = imgs.size(0)
            real_labels = torch.ones(bsz, 1, device=device)
            fake_labels = torch.zeros(bsz, 1, device=device)

            # ---- Train Discriminator ----
            opt_D.zero_grad()
            real_validity, _ = D(imgs)
            d_real_loss = criterion(real_validity, real_labels)

            z = torch.randn(bsz, config.GAN_LATENT_DIM, device=device)
            fake_imgs = G(z).detach()
            fake_validity, _ = D(fake_imgs)
            d_fake_loss = criterion(fake_validity, fake_labels)

            d_loss = 0.5 * (d_real_loss + d_fake_loss)
            d_loss.backward()
            opt_D.step()

            # ---- Train Generator ----
            opt_G.zero_grad()
            z = torch.randn(bsz, config.GAN_LATENT_DIM, device=device)
            gen_imgs = G(z)
            gen_validity, _ = D(gen_imgs)
            g_loss = criterion(gen_validity, real_labels)
            g_loss.backward()
            opt_G.step()

            epoch_d_loss += d_loss.item()
            epoch_g_loss += g_loss.item()
            n_batches += 1

        if verbose:
            print(f"[GAN] Epoch {epoch + 1}/{config.GAN_EPOCHS} "
                  f"- D_loss: {epoch_d_loss / n_batches:.4f} "
                  f"- G_loss: {epoch_g_loss / n_batches:.4f}")

    D.eval()
    return D


@torch.no_grad()
def extract_gan_features(discriminator, dataset, device=config.DEVICE, batch_size=64):
    """
    Runs every sample in `dataset` through the trained Discriminator and
    collects the penultimate-layer feature vectors and their labels.
    Returns (features: FloatTensor [N, FEATURE_DIM], labels: LongTensor [N]).
    """
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    discriminator.eval()

    all_feats, all_labels = [], []
    for imgs, labels in loader:
        imgs = imgs.to(device)
        _, feats = discriminator(imgs)
        all_feats.append(feats.cpu())
        all_labels.append(labels)

    return torch.cat(all_feats, dim=0), torch.cat(all_labels, dim=0)
