import os
import numpy as np
import torch
from torch.utils.data import Dataset

import config


class SpectrogramDataset(Dataset):
    """
    Expects a directory structure like:

        root/
            class_0/*.png (or .npy)
            class_1/*.png
            ...
            class_7/*.png

    Each image is loaded, resized to (config.IMG_SIZE, config.IMG_SIZE),
    normalized to [-1, 1], and returned as a (C, H, W) float tensor along
    with its integer class label.
    """

    def __init__(self, root_dir=None, transform=None):
        self.root_dir = root_dir or config.DATA_ROOT
        self.transform = transform
        self.samples = []  # list of (filepath, label)

        # Auto-detect class folder names rather than hardcoding them, since
        # folder naming can vary slightly (typos, casing, etc.). Only
        # directories are treated as classes; files at the root are ignored.
        all_entries = sorted(os.listdir(self.root_dir))
        classes = [
            e for e in all_entries
            if os.path.isdir(os.path.join(self.root_dir, e))
        ]

        if len(classes) != config.NUM_CLASSES:
            print(
                f"[WARNING] Expected {config.NUM_CLASSES} class folders, "
                f"found {len(classes)}: {classes}\n"
                f"          Double-check the folder names under {self.root_dir}"
            )

        self.classes = classes
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        print(f"[INFO] Detected classes ({len(classes)}): {classes}")

        for cls_name, label in self.class_to_idx.items():
            cls_dir = os.path.join(self.root_dir, cls_name)
            for fname in os.listdir(cls_dir):
                if fname.lower().endswith((".png", ".jpg", ".jpeg", ".npy", ".bmp", ".tif", ".tiff")):
                    self.samples.append((os.path.join(cls_dir, fname), label))

        print(f"[INFO] Total samples found: {len(self.samples)}")
        if len(self.samples) == 0:
            raise RuntimeError(
                f"No image files found under {self.root_dir}. "
                f"Check that each class folder contains .png/.jpg/.npy files."
            )

    def __len__(self):
        return len(self.samples)

    def _load_image(self, path):
        if path.endswith(".npy"):
            arr = np.load(path).astype(np.float32)
        else:
            from PIL import Image
            img = Image.open(path).convert(
                "L" if config.IMG_CHANNELS == 1 else "RGB"
            )
            img = img.resize((config.IMG_SIZE, config.IMG_SIZE))
            arr = np.asarray(img, dtype=np.float32)
            if config.IMG_CHANNELS == 1:
                arr = arr[None, :, :]           # (1, H, W)
            else:
                arr = arr.transpose(2, 0, 1)    # (C, H, W)
        # normalize to [-1, 1]
        arr = (arr / 127.5) - 1.0
        return arr

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        arr = self._load_image(path)
        tensor = torch.from_numpy(arr).float()
        if self.transform:
            tensor = self.transform(tensor)
        return tensor, label



        return len(self.labels)

    def __getitem__(self, idx):
        return torch.from_numpy(self.data[idx]).float(), int(self.labels[idx])
