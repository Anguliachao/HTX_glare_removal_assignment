"""SD1 dataset parsing and preprocessing utilities for notebook usage."""

from __future__ import annotations

import argparse
import os
import random
from typing import List, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

TARGET_SIZE = 512
PAD_SIZE = 24
PADDED_SIZE = TARGET_SIZE + (PAD_SIZE * 2)


def _list_png_files(data_dir: str) -> List[str]:
    """Recursively list PNG files under a directory in sorted order."""
    if not os.path.isdir(data_dir):
        raise ValueError(f"Directory does not exist: {data_dir}")

    file_paths: List[str] = []
    for root, _, files in os.walk(data_dir):
        for filename in files:
            if filename.lower().endswith(".png"):
                file_paths.append(os.path.join(root, filename))

    file_paths.sort()
    return file_paths


def _to_grayscale(panel: np.ndarray, image_path: str, panel_name: str) -> np.ndarray:
    """Convert a panel to grayscale while handling common OpenCV channel shapes."""
    if panel.ndim == 2:
        return panel
    if panel.ndim != 3:
        raise RuntimeError(
            f"Unexpected panel dimensions for {panel_name} in {image_path}: {panel.shape}"
        )

    channels = panel.shape[2]
    if channels == 4:
        return cv2.cvtColor(panel, cv2.COLOR_BGRA2GRAY)
    if channels == 3:
        return cv2.cvtColor(panel, cv2.COLOR_BGR2GRAY)
    if channels == 1:
        return panel[:, :, 0]

    raise RuntimeError(
        f"Unsupported channel count ({channels}) for {panel_name} in {image_path}"
    )


def _center_crop_or_resize(image: np.ndarray, target_size: int = TARGET_SIZE) -> np.ndarray:
    """Center crop to target size when possible, otherwise resize to target size."""
    height, width = image.shape[:2]
    if height == target_size and width == target_size:
        return image

    if height >= target_size and width >= target_size:
        top = (height - target_size) // 2
        left = (width - target_size) // 2
        return image[top : top + target_size, left : left + target_size]

    return cv2.resize(
        image, (target_size, target_size), interpolation=cv2.INTER_AREA
    )


def apply_train_augmentation(
    glare_gray: np.ndarray, gt_gray: np.ndarray, use_padding: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply paired train-only augmentations for glare and GT images."""
    glare_work = glare_gray
    gt_work = gt_gray

    if use_padding:
        glare_work = cv2.copyMakeBorder(
            glare_work,
            PAD_SIZE,
            PAD_SIZE,
            PAD_SIZE,
            PAD_SIZE,
            cv2.BORDER_REPLICATE,
        )
        gt_work = cv2.copyMakeBorder(
            gt_work,
            PAD_SIZE,
            PAD_SIZE,
            PAD_SIZE,
            PAD_SIZE,
            cv2.BORDER_REPLICATE,
        )

    if random.random() < 0.5:
        glare_work = np.fliplr(glare_work)
        gt_work = np.fliplr(gt_work)

    if random.random() < 0.5:
        glare_work = np.flipud(glare_work)
        gt_work = np.flipud(gt_work)

    if use_padding:
        max_offset = PADDED_SIZE - TARGET_SIZE
        top = random.randint(0, max_offset)
        left = random.randint(0, max_offset)

        glare_work = glare_work[top : top + TARGET_SIZE, left : left + TARGET_SIZE]
        gt_work = gt_work[top : top + TARGET_SIZE, left : left + TARGET_SIZE]

    return np.ascontiguousarray(glare_work), np.ascontiguousarray(gt_work)


class SD1Dataset(Dataset):
    """Minimal SD1 dataset for glare-removal training and validation."""

    def __init__(self, data_dir: str, split: str) -> None:
        if split not in {"train", "val"}:
            raise ValueError("split must be either 'train' or 'val'")

        self.data_dir = data_dir
        self.split = split
        self.enable_augmentation = False
        self.file_paths = _list_png_files(data_dir)

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image_path = self.file_paths[index]
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

        if image is None:
            raise RuntimeError(f"Failed to read image: {image_path}")

        if image.ndim < 2:
            raise RuntimeError(f"Invalid image shape for {image_path}: {image.shape}")

        height, width = image.shape[:2]
        if width % 3 != 0:
            raise RuntimeError(
                f"Image width is not divisible by 3 for SD1 parsing: {image_path} (w={width})"
            )
        if height <= 0:
            raise RuntimeError(f"Invalid image height for {image_path}: {height}")

        panel_width = width // 3
        gt_panel = image[:, 0:panel_width]
        glare_panel = image[:, panel_width : 2 * panel_width]

        gt_gray = _to_grayscale(gt_panel, image_path, "GT panel")
        glare_gray = _to_grayscale(glare_panel, image_path, "Glare panel")

        gt_gray = _center_crop_or_resize(gt_gray, TARGET_SIZE)
        glare_gray = _center_crop_or_resize(glare_gray, TARGET_SIZE)

        if self.split == "train" and self.enable_augmentation:
            glare_gray, gt_gray = apply_train_augmentation(
                glare_gray, gt_gray, use_padding=True
            )

        glare_tensor = torch.from_numpy(glare_gray.astype(np.float32) / 255.0).unsqueeze(
            0
        )
        gt_tensor = torch.from_numpy(gt_gray.astype(np.float32) / 255.0).unsqueeze(0)

        return glare_tensor, gt_tensor


def visualize_sample(dataset: SD1Dataset, index: int = 0) -> None:
    """Visualize one parsed sample (glare input and ground truth) from a dataset."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required for visualize_sample(). "
            "Install it or visualize tensors manually in the notebook."
        ) from exc

    glare_tensor, gt_tensor = dataset[index]
    glare_img = glare_tensor.squeeze(0).cpu().numpy()
    gt_img = gt_tensor.squeeze(0).cpu().numpy()

    figure, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(glare_img, cmap="gray", vmin=0.0, vmax=1.0)
    axes[0].set_title("Glare Input")
    axes[0].axis("off")

    axes[1].imshow(gt_img, cmap="gray", vmin=0.0, vmax=1.0)
    axes[1].set_title("Ground Truth")
    axes[1].axis("off")

    figure.tight_layout()
    plt.show()


def get_dataloaders(
    train_dir: str,
    val_dir: str,
    batch_size: int,
    val_batch_size: int,
    num_workers: int = 0,
    enable_augment: bool = True,
) -> Tuple[DataLoader, DataLoader]:
    """Create train/val dataloaders for SD1."""
    train_dataset = SD1Dataset(data_dir=train_dir, split="train")
    train_dataset.enable_augmentation = enable_augment

    val_dataset = SD1Dataset(data_dir=val_dir, split="val")
    val_dataset.enable_augmentation = False

    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader


def _build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI parser for quick local dataset verification."""
    parser = argparse.ArgumentParser(description="SD1 dataset quick verification")
    parser.add_argument("--train_dir", type=str, required=True, help="Path to train images")
    parser.add_argument("--val_dir", type=str, required=True, help="Path to val images")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size for loaders")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader workers")
    parser.add_argument(
        "--disable_augment",
        action="store_true",
        help="Disable train augmentation for debugging",
    )
    return parser


if __name__ == "__main__":
    args = _build_arg_parser().parse_args()
    train_loader, val_loader = get_dataloaders(
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        enable_augment=not args.disable_augment,
    )

    print(f"Train dataset length: {len(train_loader.dataset)}")
    print(f"Val dataset length: {len(val_loader.dataset)}")

    if len(train_loader.dataset) > 0:
        glare_sample, gt_sample = train_loader.dataset[0]
        print(f"One train sample shapes: glare={glare_sample.shape}, gt={gt_sample.shape}")
    else:
        print("Train dataset is empty.")

    if len(val_loader.dataset) > 0:
        glare_sample, gt_sample = val_loader.dataset[0]
        print(f"One val sample shapes: glare={glare_sample.shape}, gt={gt_sample.shape}")
    else:
        print("Val dataset is empty.")

    try:
        glare_batch, gt_batch = next(iter(train_loader))
        print(f"One train batch shapes: glare={glare_batch.shape}, gt={gt_batch.shape}")
    except StopIteration:
        print("Train loader has no batches.")
