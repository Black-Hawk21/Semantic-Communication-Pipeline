"""
dataset.py — ImageNet 64x64 dataset loader
"""

import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


# ─────────────────────────────────────────────
# ImageNet 64x64 Dataset
# ─────────────────────────────────────────────

class ImageNet32Dataset(Dataset):
    """
    Loader for pre-extracted ImageNet 64x64 images stored as PNG/JPEG files.

    Expected directory structure
    ────────────────────────────
    root/
      train/
        class_0/
          img1.png
          img2.jpeg
          ...
        class_1/
          ...
      val/
        class_0/
          ...
        class_1/
          ...

    Class folders are sorted alphabetically; the folder index (0-based)
    is used as the integer label.
    """

    def __init__(
        self,
        root:      str,
        split:     str = "train",
        transform: Optional[object] = None,
    ):
        assert split in ("train", "val"), "split must be 'train' or 'val'"
        self.transform = transform
        self.samples: list  # list of (image_path, label) tuples

        self.samples = self._load(Path(root), split)

    # ── internal ──────────────────────────────

    def _load(self, root: Path, split: str) -> list:
        folder = root / split
        if not folder.exists():
            raise FileNotFoundError(f"Split folder not found: {folder}")

        # Sort class directories for a consistent label assignment
        class_dirs = sorted([d for d in folder.iterdir() if d.is_dir()])
        if not class_dirs:
            raise FileNotFoundError(
                f"No class subdirectories found in {folder}. "
                "Expected folders named e.g. 'class_0', 'class_1', ..."
            )

        samples = []
        for label, class_dir in enumerate(class_dirs):
            image_files = sorted([
                p for p in class_dir.iterdir()
                if p.suffix.lower() in (".png", ".jpg", ".jpeg")
            ])
            for img_path in image_files:
                samples.append((img_path, label))

        if not samples:
            raise FileNotFoundError(f"No PNG/JPEG images found under {folder}.")

        return samples

    # ── Dataset API ───────────────────────────

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img, label


# ─────────────────────────────────────────────
# Transforms
# ─────────────────────────────────────────────

def get_transforms(split: str = "train"):
    """
    Returns standard transforms for train / val.
    Output tensors are in [-1, 1] to match the Decoder's Tanh output.
    """
    if split == "train":
        return transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.ToTensor(),                             # [0, 1]
            transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),  # → [-1, 1]
        ])
    else:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),
        ])


# ─────────────────────────────────────────────
# DataLoader factory
# ─────────────────────────────────────────────

def get_dataloader(
    root:        str,
    split:       str   = "train",
    batch_size:  int   = 128,
    num_workers: int   = 4,
    pin_memory:  bool  = True,
) -> DataLoader:
    """
    Returns a DataLoader for the ImageNet 64x64 dataset.

    Args:
        root        : Path to the dataset root directory.
        split       : 'train' or 'val'.
        batch_size  : Mini-batch size.
        num_workers : Number of data-loading workers.
        pin_memory  : Use pinned memory (faster GPU transfers).
    """
    dataset = ImageNet32Dataset(
        root      = root,
        split     = split,
        transform = get_transforms(split),
    )
    return DataLoader(
        dataset,
        batch_size  = batch_size,
        shuffle     = (split == "train"),
        num_workers = num_workers,
        pin_memory  = pin_memory,
        drop_last   = (split == "train"),
    )
