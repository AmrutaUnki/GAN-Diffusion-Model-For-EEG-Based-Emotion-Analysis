import argparse
import random
import numpy as np
import torch
from sklearn.model_selection import train_test_split
import config
from dataset import SpectrogramDataset
from gan import train_gan, extract_gan_features
from diffusion import FeatureDiffusion
from cnn import train_cnn, evaluate_cnn


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dummy", action="store_true",
                        help="Use synthetic spectrograms instead of config.DATA_ROOT "
                             "(useful for verifying the pipeline runs end-to-end).")
    parser.add_argument("--data_root", type=str, default=None,
                        help="Override config.DATA_ROOT.")
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument("--gan_epochs", type=int, default=config.GAN_EPOCHS)
    parser.add_argument("--diff_epochs", type=int, default=config.DIFF_EPOCHS)
    parser.add_argument("--cnn_epochs", type=int, default=config.CNN_EPOCHS)
    args = parser.parse_args()

    # allow quick epoch overrides for smoke-testing without editing config.py
    config.GAN_EPOCHS = args.gan_epochs
    config.DIFF_EPOCHS = args.diff_epochs
    config.CNN_EPOCHS = args.cnn_epochs

    set_seed(args.seed)
    print(f"[INFO] Using device: {config.DEVICE}")

    data_root = args.data_root or config.DATA_ROOT
     print(f"[INFO] Loading spectrograms from: {data_root}")
    full_dataset = SpectrogramDataset(root_dir=data_root)
     all_labels = np.array([lbl for _, lbl in full_dataset.samples])

    indices = np.arange(len(full_dataset))
    train_idx, test_idx = train_test_split(
        indices, test_size=0.2, stratify=all_labels, random_state=args.seed
    )
    train_subset = torch.utils.data.Subset(full_dataset, train_idx)
    test_subset = torch.utils.data.Subset(full_dataset, test_idx)
    print(f"[INFO] Train samples: {len(train_subset)} | Test samples: {len(test_subset)}")

    # -----------------------------------------------------------------
    # 2) Stage 1 - GAN: train adversarially, then extract features
    # -----------------------------------------------------------------
    print("\n===== Stage 1: GAN training =====")
    discriminator = train_gan(train_subset, device=config.DEVICE)

    print("\n[INFO] Extracting GAN features for train/test sets...")
    train_gan_feats, train_labels = extract_gan_features(discriminator, train_subset, device=config.DEVICE)
    test_gan_feats, test_labels = extract_gan_features(discriminator, test_subset, device=config.DEVICE)

    # -----------------------------------------------------------------
    # 3) Stage 2 - Diffusion: train on GAN features, then refine them
    # -----------------------------------------------------------------
    print("\n===== Stage 2: Diffusion training =====")
    diffusion = FeatureDiffusion(device=config.DEVICE)
    diffusion.train(train_gan_feats)

    print("\n[INFO] Refining GAN features with the trained diffusion model...")
    train_refined_feats = diffusion.refine(train_gan_feats)
    test_refined_feats = diffusion.refine(test_gan_feats)

    # -----------------------------------------------------------------
    # 4) Stage 3 - CNN: classify refined features into 8 classes
    # -----------------------------------------------------------------
    print("\n===== Stage 3: CNN classifier training =====")
    cnn_model = train_cnn(
        train_refined_feats, train_labels,
        val_features=test_refined_feats, val_labels=test_labels,
        device=config.DEVICE,
    )

    final_acc = evaluate_cnn(cnn_model, test_refined_feats, test_labels, device=config.DEVICE)
    print(f"\n[RESULT] Final test accuracy (GAN+Diffusion+CNN hybrid): {final_acc:.2f}%")


if __name__ == "__main__":
    main()
