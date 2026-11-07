"""dataset.py - CamVid dataset for U-Net semantic segmentation.

Pipeline for one sample:
    image : load RGB -> resize (bilinear) -> to tensor [0, 1] -> (augment) -> normalize
    mask  : load RGB -> resize (nearest)  -> RGB colors -> class indices   -> (augment)

Image :
    load RGB : we load the picture as a 3 channels image (this is useful if the image is in grayscale for example)
    resize (binilear) : we resize the images to 352×480, to reduce the training cost. The bilinear mode
    calculate a mean of the neighboring pixels.
    normalize (to tensor [0,1]) : using min-max normalisation
    augment (only during training) : flip, zoom/crop, luminosity ... To create variants of the same image.
    On small datsets like the camVid, it limits overfitting.
    standardize : channel by channel. Makes the training more stable.

Mask :
    Some common steps
    RGB colors -> class indices : Turning each colour to a correponding category (0, 1, 2, 3).
    Unknown colours become 255.
    and we do the mapping to the classes tables. We end up assigning a class to each pixel, having
    a [H,W] tensor, exactly what the CrossEntropyLoss expects.
"""


import os
import random

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader

IGNORE_INDEX = 255 # will be used in the loss function
IMG_EXTS = (".png", ".jpg", ".jpeg")


# ImageNet statistics because we will be using a pretrained encoder
# the .view() is necessary so these values will be applied to the correponding channels of the image
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


# --------------------------------------------------------------------------- #
# Color <-> class index mapping
# --------------------------------------------------------------------------- #
class ClassMapper:
    """Converts RGB masks to class-index masks (and back) using class_dict.csv.

    Classes listed in `ignore_names` (e.g. "Void") are NOT given an index:
    their pixels become IGNORE_INDEX. The remaining classes are numbered
    0..num_classes-1 in the order they appear in the CSV.
    """

    def __init__(self, class_dict_path, ignore_names=("Void",)):
        cd = pd.read_csv(class_dict_path)
        cd.columns = [c.strip().lower() for c in cd.columns]
        cd["name"] = cd["name"].strip()

        ignore = {n.lower() for n in ignore_names}
        keep = cd[~cd["name"].lower().isin(ignore)].reset_index(drop=True)

        self.names = keep["name"].tolist()  # index -> name
        self.palette = keep[["r", "g", "b"]].to_numpy(dtype=np.uint8)  # index -> RGB
        self.num_classes = len(self.names)

        # Lookup table: encoded color -> class index. Unknown colors -> IGNORE_INDEX.
        self.lut = np.full(256 ** 3, IGNORE_INDEX, dtype=np.uint8)
        self.lut[self._encode(self.palette)] = np.arange(self.num_classes, dtype=np.uint8)

    @staticmethod
    def _encode(rgb):
        """RGB -> one unique integer per color."""
        rgb = rgb.astype(np.uint32)
        return (rgb[:, :, 0] << 16) | (rgb[:, :, 1] << 8) | rgb[:, :, 2]

    def rgb_to_index(self, rgb):
        """HxWx3 uint8 array -> HxW uint8 array of class indices."""
        return self.lut[self._encode(rgb)]

    def index_to_rgb(self, idx):
        """HxW array of class indices -> HxWx3 uint8 RGB image (for visualization)."""
        idx = np.asarray(idx)
        out = np.zeros((*idx.shape, 3), dtype=np.uint8)
        valid = idx != IGNORE_INDEX
        out[valid] = self.palette[idx[valid]]
        return out


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
class CamVidDataset(Dataset):
    """
    Args:
        img_dir, mask_dir : folders with the images and the RGB masks.
        mapper            : a ClassMapper instance.
        size              : (height, width) after resizing. U-Net downsamples 5 times,
                            so both values should be divisible by 32 (352 and 480 are).
        mask_suffix       : suffix added to the image name to get the mask name
                            (e.g. "0001TP_006690.png" -> "0001TP_006690_L.png").
        augment           : True for the training set only.
    """

    def __init__(self, img_dir, mask_dir, mapper, size=(352, 480), mask_suffix="_L", augment=False):
        self.mapper = mapper
        self.size = size
        self.augment = augment

        # Index the masks by their "image-equivalent" name
        mask_paths = {}
        for f in sorted(os.listdir(mask_dir)):
            if not f.lower().endswith(IMG_EXTS):
                continue
            stem = os.path.splitext(f)[0]
            if mask_suffix and stem.endswith(mask_suffix):
                stem = stem[: -len(mask_suffix)]
            mask_paths[stem] = os.path.join(mask_dir, f)

        # Pair each image with its mask by name (safer than relying on sorted order)
        self.samples, missing = [], []
        for f in sorted(os.listdir(img_dir)):
            if not f.lower().endswith(IMG_EXTS):
                continue
            stem = os.path.splitext(f)[0]
            if stem in mask_paths:
                self.samples.append((os.path.join(img_dir, f), mask_paths[stem]))
            else:
                missing.append(f)
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} images have no mask (e.g. {missing[:3]}). "
                f"Check mask_suffix={mask_suffix!r} and the folder names."
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        img_path, mask_path = self.samples[i]
        h, w = self.size

        img = Image.open(img_path).convert("RGB").resize((w, h), Image.BILINEAR)
        # NEAREST never creates new colors, so the RGB -> index lookup stays exact
        mask = Image.open(mask_path).convert("RGB").resize((w, h), Image.NEAREST)

        img = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0  # [3, H, W] in [0, 1]
        mask = torch.from_numpy(self.mapper.rgb_to_index(np.array(mask))).long()  # [H, W] indices

        if self.augment:
            img, mask = self._augment(img, mask)

        img = (img - IMAGENET_MEAN) / IMAGENET_STD
        return img, mask

    @staticmethod
    def _augment(img, mask):
        """Geometric transforms are applied to image AND mask, color transforms to the image only."""
        # Horizontal flip
        if random.random() < 0.5:
            img, mask = img.flip(-1), mask.flip(-1)

        # Random scale + crop, then resize back to the original size
        if random.random() < 0.5:
            _, h, w = img.shape
            scale = random.uniform(0.6, 1.0)
            ch, cw = int(h * scale), int(w * scale)
            top, left = random.randint(0, h - ch), random.randint(0, w - cw)
            img = img[:, top:top + ch, left:left + cw]
            mask = mask[top:top + ch, left:left + cw]
            img = F.interpolate(img[None], size=(h, w), mode="bilinear", align_corners=False)[0]
            mask = F.interpolate(mask[None, None].float(), size=(h, w), mode="nearest")[0, 0].long()

        # Brightness / contrast jitter
        if random.random() < 0.5:
            brightness, contrast = random.uniform(0.8, 1.2), random.uniform(0.8, 1.2)
            m = img.mean()
            img = (((img - m) * contrast + m) * brightness).clamp(0.0, 1.0) # clamp(0.0, 1.0) garantees all values stay in the [0,1] range

        return img, mask


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def denormalize(img):
    """Undo the ImageNet normalization(standardization): [3, H, W] tensor -> HxWx3 float array in [0, 1]."""
    img = (img.cpu() * IMAGENET_STD + IMAGENET_MEAN).clamp(0, 1)
    return img.permute(1, 2, 0).numpy()


def get_dataloaders(data_dir, splits, batch_size=8, size=(352, 480),
                    mask_suffix="_L", num_workers=2, ignore_names=("Void",)):
    """
    Args:
        data_dir : root folder of the dataset (contains class_dict.csv).
        splits   : {"train": (images_folder, masks_folder), "val": (...), "test": (...)}
    Returns:
        loaders (dict of DataLoader), mapper (ClassMapper)
    """
    mapper = ClassMapper(os.path.join(data_dir, "class_dict.csv"), ignore_names=ignore_names)

    loaders = {}
    for split, (img_folder, mask_folder) in splits.items():
        is_train = split == "train" # to do the augmentation only if it is true
        ds = CamVidDataset(
            os.path.join(data_dir, img_folder),
            os.path.join(data_dir, mask_folder),
            mapper, size=size, mask_suffix=mask_suffix, augment=is_train,
        )
        loaders[split] = DataLoader(
            ds, batch_size=batch_size, shuffle=is_train, num_workers=num_workers,
            pin_memory=torch.cuda.is_available(), drop_last=is_train,
        )
    return loaders, mapper

