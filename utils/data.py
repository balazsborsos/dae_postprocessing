import os
import cv2
import torch
import random
import numpy as np

from typing import Union, Tuple
from pathlib import Path
from PIL import Image

from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split


class DataAugmenter:
    def __init__(self):

        self.max_patches = 2

    def random_region_deletion(self,
                               mask: torch.Tensor,
                               patch_size: Tuple[int, int] = (32, 32),
                               max_patches: int = 10) -> torch.Tensor:
        """
        Randomly delete patches of a specified size from the input mask.

        Args:
            mask (torch.Tensor): Input image.
            patch_size (tuple): Size of the patches to delete.
            max_patches (int): Maximum number of patches to delete.

        Returns:
            torch.Tensor: Image with patches randomly deleted.
        """
        corrupted_mask = mask.clone()
        # rand_values = torch.rand(mask.shape[1:])

        # Calculate the maximum starting coordinates for patches
        num_x = int(mask.shape[0] / patch_size[0])
        num_y = int(mask.shape[1] / patch_size[1])

        possible_patches = []
        for x in range(0, num_x):
            for y in range(0, num_y):
                if mask[x * patch_size[0]:x * patch_size[0] + patch_size[0],
                   y * patch_size[1]:y * patch_size[1] + patch_size[1]].any() == 1:
                    possible_patches.append((x, y))

        if not possible_patches:
            return mask
        num_patches = random.randint(0, max_patches) if len(possible_patches) > max_patches else len(possible_patches)
        sample = random.sample(possible_patches, num_patches)
        for (x, y) in sample:
            corrupted_mask[x * patch_size[0]:x * patch_size[0] + patch_size[0],
            y * patch_size[1]:y * patch_size[1] + patch_size[1]] = 0

        return corrupted_mask

    def random_circle_addition(self,
                               mask: torch.Tensor,
                               circle_radius: int = 8,
                               max_circles: int = 10) -> torch.Tensor:
        """
        Randomly delete patches of a specified size from the input mask.

        Args:
            mask (torch.Tensor): Input image.
            circle_radius (int): Radius of circles to add.
            max_circles (int): Maximum number of possible circles.

        Returns:
            torch.Tensor: Image with randomly added circles.
        """
        corrupted_mask = mask.clone()
        num_circles = random.randint(0, max_circles)

        for i in range(num_circles):
            x, y = random.randint(0, mask.shape[0]), random.randint(0, mask.shape[1])
            X_grid, Y_grid = np.ogrid[:mask.shape[0], :mask.shape[1]]
            dist_from_center = np.sqrt((X_grid - x) ** 2 + (Y_grid - y) ** 2)

            corrupted_mask[dist_from_center <= circle_radius] = 1

        return corrupted_mask

    def random_swap_labels_with_distance(self,
                                         mask: torch.Tensor,
                                         num_swaps: int = 100,
                                         max_distance: int = 20) -> torch.Tensor:
        """
        Swap out an arbitrary number of pixels close to segmentation borders with random distance.

        Args:
            mask (torch.Tensor): Input image.
            num_swaps (int): Number of pixels to swap.
            max_distance (int): Maximum distance from segmentation border to swap.

        Returns:
            torch.Tensor: Image with randomly swapped pixels.
        """
        # Find the edges of the mask
        corrupted_mask = mask.clone()
        mask_n = corrupted_mask.numpy()
        max_x, max_y = mask_n.shape
        mask_edges = cv2.Canny(mask_n.astype(np.uint8), 0, 1)

        # Get the indices of the edge pixels
        edge_indices = np.argwhere(mask_edges > 0)

        # Randomly select 'num_swaps' pixels to swap
        swap_indices = np.random.choice(len(edge_indices), num_swaps, replace=False)

        # Swap foreground and background labels at selected edge pixels with random distance
        for idx in swap_indices:
            x, y = edge_indices[idx]
            swap_distance = np.random.randint(1, max_distance + 1)
            corrupted_mask[min(x + swap_distance, max_x - 1), y] = corrupted_mask[max(x - swap_distance, 0), y]
            corrupted_mask[max(x - swap_distance, 0), y] = abs(1 - corrupted_mask[min(x + swap_distance, max_x - 1), y])
        # corrupted_mask = torch.Tensor(mask)
        return corrupted_mask

    def __call__(self, input: torch.Tensor) -> torch.Tensor:
        aug = self.random_swap_labels_with_distance(input, num_swaps=300, max_distance=20)
        aug = self.random_region_deletion(aug, patch_size=(32, 32), max_patches=15)
        aug = self.random_circle_addition(aug, circle_radius=8, max_circles=10)
        return aug


class LungMasksDataset(Dataset):
    def __init__(
            self,
            mask_dir: Path,
            mlset: str,
            train_ratio: float = 0.8,
            **kwargs
    ):

        self.mask_dir = mask_dir
        self.mlset = mlset
        self.mask_paths = os.listdir(self.mask_dir)
        self.train_ratio = train_ratio
        self.augmentations = DataAugmenter()
        self.convert_to_tensor = transforms.ToTensor()
        self.img_size = kwargs['img_size']

        if self.mlset == 'training':
            self.masks, _ = train_test_split(self.mask_paths, train_size=self.train_ratio, shuffle=True)
        else:
            _, self.masks = train_test_split(self.mask_paths, train_size=self.train_ratio, shuffle=False)

    def __len__(self):
        return len(self.masks)

    def __getitem__(self, index):

        mask_path = os.path.join(self.mask_dir, self.masks[index])
        mask = np.array(Image.open(mask_path).convert("L")) / 255
        mask = torch.Tensor(cv2.resize(mask, self.img_size))

        corrupted_mask = self.augmentations(mask)

        return corrupted_mask[None], mask[None]


def get_dataloader(
        path: Union[str, Path],
        mlset: str,
        *args,
        **kwargs
) -> DataLoader:
    dataset = LungMasksDataset(path, mlset=mlset, **kwargs)
    dataloader = DataLoader(
        dataset,
        shuffle=(mlset == "training"),
        batch_size=kwargs.get("batch_size", 1),
        num_workers=kwargs.get("num_workers", 0),
    )
    return dataloader
