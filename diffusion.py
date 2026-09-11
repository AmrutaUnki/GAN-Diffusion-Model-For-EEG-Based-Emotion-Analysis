"""
Diffusion stage of the pipeline.

Rather than diffusing raw images, this diffusion model operates directly on
the FEATURE_DIM-dimensional feature vectors extracted by the trained GAN
Discriminator. It learns to predict the Gaussian noise added at each forward
diffusion step (standard DDPM objective). At inference time, a *partial*
forward-noise + reverse-denoise cycle (DIFF_REFINE_STEPS out of
DIFF_TIMESTEPS) is used to "refine" each GAN feature vector -- this smooths
out noise/outlier structure in the raw GAN features while keeping them close
to the original representation (a full T-step reverse process starting from
pure noise would instead generate an unrelated feature vector, which is not
what "refinement" calls for here).

Activation : ReLU (as specified)
Optimizer  : Adam, lr = 0.001
Loss       : Mean Squared Error
Epochs     : 30
Batch size : 16
"""

import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import config


def _sinusoidal_time_embedding(timesteps, dim):
    """Standard transformer-style sinusoidal embedding for the diffusion timestep."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000) * torch.arange(half, dtype=torch.float32) / half
    ).to(timesteps.device)
    args = timesteps.float()[:, None] * freqs[None, :]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2 == 1:
        emb = nn.functional.pad(emb, (0, 1))
    return emb


class DenoiseNet(nn.Module):
    """MLP noise-prediction network: input = (noisy feature vector, timestep),
    output = predicted noise (same shape as the feature vector)."""

    def __init__(self, feature_dim=config.FEATURE_DIM, time_emb_dim=64, hidden_dim=512):
        super().__init__()
        self.time_emb_dim = time_emb_dim

        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.net = nn.Sequential(
            nn.Linear(feature_dim + hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, feature_dim),
        )

    def forward(self, x_t, t):
        t_emb = _sinusoidal_time_embedding(t, self.time_emb_dim)
        t_emb = self.time_mlp(t_emb)
        h = torch.cat([x_t, t_emb], dim=-1)
        return self.net(h)


class FeatureDiffusion:
    """Wraps the forward (noising) process, training loop, and the partial
    reverse (denoising) process used for feature refinement."""

    def __init__(self, feature_dim=config.FEATURE_DIM, timesteps=config.DIFF_TIMESTEPS,
                 device=config.DEVICE):
        self.device = device
        self.T = timesteps

        betas = torch.linspace(config.DIFF_BETA_START, config.DIFF_BETA_END, timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.betas = betas.to(device)
        self.alphas = alphas.to(device)
        self.alphas_cumprod = alphas_cumprod.to(device)
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod).to(device)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod).to(device)

        self.model = DenoiseNet(feature_dim=feature_dim).to(device)

    def q_sample(self, x0, t, noise=None):
        """Forward diffusion: sample x_t given x_0 and timestep t."""
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_ac = self.sqrt_alphas_cumprod[t][:, None]
        sqrt_1m_ac = self.sqrt_one_minus_alphas_cumprod[t][:, None]
        return sqrt_ac * x0 + sqrt_1m_ac * noise, noise

    def train(self, features, verbose=True):
        """
        features: FloatTensor [N, feature_dim] -- the GAN-extracted features.
        Trains the noise-prediction network with the standard DDPM MSE loss.
        """
        dataset = TensorDataset(features)
        loader = DataLoader(dataset, batch_size=config.DIFF_BATCH_SIZE, shuffle=True, drop_last=True)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=config.DIFF_LR)

        self.model.train()
        for epoch in range(config.DIFF_EPOCHS):
            epoch_loss = 0.0
            n_batches = 0
            for (x0,) in loader:
                x0 = x0.to(self.device)
                bsz = x0.size(0)
                t = torch.randint(0, self.T, (bsz,), device=self.device).long()

                x_t, noise = self.q_sample(x0, t)
                predicted_noise = self.model(x_t, t)
                loss = criterion(predicted_noise, noise)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            if verbose:
                print(f"[Diffusion] Epoch {epoch + 1}/{config.DIFF_EPOCHS} "
                      f"- MSE loss: {epoch_loss / n_batches:.6f}")

        self.model.eval()

    @torch.no_grad()
    def refine(self, features, refine_steps=config.DIFF_REFINE_STEPS):
        """
        Partial noise + denoise cycle used to refine GAN-extracted features.

        1) Forward-diffuse each feature vector from t=0 to t=refine_steps.
        2) Run the learned reverse process back down to t=0.

        This keeps the refined vector close to the original GAN feature
        (unlike full T-step generation from pure noise), while allowing the
        diffusion model to smooth/denoise fine-grained structure.
        """
        self.model.eval()
        x = features.to(self.device)
        bsz = x.size(0)

        t_start = torch.full((bsz,), refine_steps - 1, device=self.device, dtype=torch.long)
        x_t, _ = self.q_sample(x, t_start)

        for step in reversed(range(refine_steps)):
            t = torch.full((bsz,), step, device=self.device, dtype=torch.long)
            predicted_noise = self.model(x_t, t)

            alpha_t = self.alphas[t][:, None]
            alpha_cumprod_t = self.alphas_cumprod[t][:, None]
            beta_t = self.betas[t][:, None]

            coef1 = 1.0 / torch.sqrt(alpha_t)
            coef2 = beta_t / torch.sqrt(1.0 - alpha_cumprod_t)
            mean = coef1 * (x_t - coef2 * predicted_noise)

            if step > 0:
                noise = torch.randn_like(x_t)
                x_t = mean + torch.sqrt(beta_t) * noise
            else:
                x_t = mean

        return x_t.cpu()
