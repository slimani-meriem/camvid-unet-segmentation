"""metrics.py - IoU, Dice and mIoU computed from a confusion matrix.

cm[i, j] = number of pixels of true class i predicted as class j, accumulated over
the whole split (not image by image: rare classes are absent from most images).
For a class k: TP = cm[k, k], FN = rest of row k, FP = rest of column k.

    IoU  = TP / (TP + FP + FN)        Dice = 2 TP / (2 TP + FP + FN)

mIoU only averages the classes present in the ground truth.
"""

import numpy as np

# 31 CamVid classes -> 11 "standard" classes (the SegNet benchmark: Badrinarayanan et al., 2017).
# Void is already excluded by dataset.py.
CAMVID_11_GROUPS = {
    "Sky":        ["Sky"],
    "Building":   ["Building", "Wall", "Archway", "Bridge", "Tunnel"],
    "Pole":       ["Column_Pole", "TrafficCone"],
    "Road":       ["Road", "LaneMkgsDriv", "LaneMkgsNonDriv"],
    "Pavement":   ["Sidewalk", "ParkingBlock", "RoadShoulder"],
    "Tree":       ["Tree", "VegetationMisc"],
    "SignSymbol": ["SignSymbol", "TrafficLight", "Misc_Text"],
    "Fence":      ["Fence"],
    "Car":        ["Car", "SUVPickupTruck", "Truck_Bus", "Train", "OtherMoving"],
    "Pedestrian": ["Pedestrian", "Child", "CartLuggagePram", "Animal"],
    "Bicyclist":  ["Bicyclist", "MotorcycleScooter"],
}


# --------------------------------------------------------------------------- 
# Accumulating the confusion matrix
# --------------------------------------------------------------------------- 
def confusion_from_batch(pred, target, num_classes):
    """pred, target: [B, H, W] int64 tensors -> [C, C] confusion matrix (ignored pixels skipped)."""
    import torch
    from .dataset import IGNORE_INDEX

    valid = (target != IGNORE_INDEX) & (target < num_classes)
    idx = target[valid] * num_classes + pred[valid]
    return torch.bincount(idx, minlength=num_classes ** 2).reshape(num_classes, num_classes)


def compute_confusion(model, loader, num_classes, device, amp=True):
    """Run the model on a whole loader and return the confusion matrix as a numpy array."""
    import torch

    model.eval()
    cm = torch.zeros(num_classes, num_classes, dtype=torch.long, device=device)
    with torch.no_grad():
        for img, mask in loader:
            img = img.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
                logits = model(img)
            cm += confusion_from_batch(logits.argmax(dim=1), mask, num_classes)
    return cm.cpu().numpy()


# --------------------------------------------------------------------------- 
# Scores
# --------------------------------------------------------------------------- 
def scores_from_confusion(cm):
    """Per-class IoU / Dice and global scores from a [C, C] confusion matrix."""
    cm = np.asarray(cm, dtype=np.float64)
    tp = np.diag(cm)
    gt = cm.sum(axis=1)     # pixels of each true class (support)
    pred = cm.sum(axis=0)   # pixels predicted as each class
    fp, fn = pred - tp, gt - tp

    with np.errstate(divide="ignore", invalid="ignore"):
        iou = tp / (tp + fp + fn)
        dice = 2 * tp / (2 * tp + fp + fn)

    present = gt > 0
    return {
        "miou": float(iou[present].mean()),
        "mdice": float(dice[present].mean()),
        "pixel_acc": float(tp.sum() / cm.sum()),
        "iou": iou, "dice": dice, "support": gt, "present": present,
    }


def group_confusion(cm, names, groups=CAMVID_11_GROUPS):
    """Merge a C x C confusion matrix into a smaller one (31 -> 11) WITHOUT retraining.

    With P the [C, C_new] 0/1 membership matrix: cm_new = P^T @ cm @ P.
    Classes that belong to no group (none with CAMVID_11_GROUPS) would be dropped from rows AND columns.
    Returns (cm_new, new_names).
    """
    cm = np.asarray(cm)
    new_names = list(groups)
    old_to_new = {}
    for j, members in enumerate(groups.values()):
        for m in members:
            if m not in names:
                raise ValueError(f"'{m}' is in the grouping but not in the class list.")
            if m in old_to_new:
                raise ValueError(f"'{m}' belongs to two groups.")
            old_to_new[m] = j

    P = np.zeros((len(names), len(new_names)), dtype=cm.dtype)
    for i, n in enumerate(names):
        if n in old_to_new:
            P[i, old_to_new[n]] = 1
    return P.T @ cm @ P, new_names