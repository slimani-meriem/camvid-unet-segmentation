"""train.py - fine-tune a U-Net (pretrained ResNet34 encoder) on CamVid (31 classes).

Run from the repository root, e.g. on Colab:
    python -m src.train --data-dir data/CamVid --epochs 30

Two phases (see model.py):
    epochs 1..freeze_epochs : encoder frozen (and kept in eval mode), only the decoder learns
    afterwards              : everything trains, encoder at lr * encoder_lr_mult

"""

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from .dataset import IGNORE_INDEX, get_dataloaders
from .metrics import compute_confusion, scores_from_confusion
from .model import ENCODER_NAME, build_model, count_params, get_param_groups, set_encoder_trainable

SPLITS = {
    "train": ("train", "train_labels"),
    "val": ("val", "val_labels"),
    "test": ("test", "test_labels"),
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default="data/CamVid")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--freeze-epochs", type=int, default=3, help="epochs with the encoder frozen")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3, help="learning rate of the decoder / head")
    p.add_argument("--encoder-lr-mult", type=float, default=0.1, help="encoder lr = lr * this")
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--loss", choices=["ce", "ce_dice"], default="ce")
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-amp", action="store_true", help="disable mixed precision")
    p.add_argument("--ckpt-dir", default="checkpoints", help="best model is saved here (git-ignored)")
    p.add_argument("--results-dir", default="results", help="history (json) and curves (png) are saved here")
    p.add_argument("--run-name", default=None)
    return p.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_loss(name):
    ce = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)  # ignored pixels (Void) don't contribute
    if name == "ce":
        return ce

    import segmentation_models_pytorch as smp
    dice = smp.losses.DiceLoss(
        mode="multiclass",
        ignore_index=IGNORE_INDEX
    )

    return lambda logits, target: ce(logits, target) + dice(logits, target)


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, use_amp, encoder_frozen):
    model.train()
    if encoder_frozen:
        model.encoder.eval()
    running, seen = 0.0, 0
    for img, mask in loader:
        img = img.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits = model(img)
        loss = criterion(logits.float(), mask)  # loss in float32 for numerical stability
        scaler.scale(loss).backward() # to resolve potential underflow problem
        scaler.step(optimizer)
        scaler.update()
        running += loss.item() * img.size(0)
        seen += img.size(0)
    return running / seen


def save_curves(history, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = [h["epoch"] for h in history]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(epochs, [h["train_loss"] for h in history])
    ax[0].set(title="Training loss", xlabel="epoch")
    ax[1].plot(epochs, [h["val_miou"] for h in history], label="mIoU")
    ax[1].plot(epochs, [h["val_mdice"] for h in history], label="mDice")
    ax[1].set(title="Validation (31 classes, present classes only)", xlabel="epoch")
    ax[1].legend()
    for a in ax:
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda" and not args.no_amp
    run_name = args.run_name or "unet_resnet34"
    ckpt_dir, results_dir = Path(args.ckpt_dir), Path(args.results_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    print(f"device={device} | amp={use_amp} | run={run_name}")

    loaders, mapper = get_dataloaders(args.data_dir, SPLITS, batch_size=args.batch_size, num_workers=args.num_workers)
    C = mapper.num_classes
    print(f"{C} classes | " + " | ".join(f"{k}: {len(v.dataset)} images" for k, v in loaders.items()))

    model = build_model(num_classes=C).to(device)
    total, _ = count_params(model)
    print(f"encoder={ENCODER_NAME} | {total / 1e6:.1f} M parameters")

    criterion = build_loss(args.loss)
    optimizer = torch.optim.AdamW(get_param_groups(model, args.lr, args.encoder_lr_mult, args.weight_decay))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    history, best_miou = [], -1.0
    for epoch in range(1, args.epochs + 1):
        frozen = epoch <= args.freeze_epochs
        set_encoder_trainable(model, not frozen)
        if epoch == args.freeze_epochs + 1:
            _, trainable = count_params(model)
            print(f"--> encoder unfrozen ({trainable / 1e6:.1f} M trainable parameters)")

        t0 = time.time()
        train_loss = train_one_epoch(model, loaders["train"], criterion, optimizer, scaler, device, use_amp, frozen)
        scheduler.step()

        scores = scores_from_confusion(compute_confusion(model, loaders["val"], C, device, amp=use_amp))
        row = {"epoch": epoch, "frozen": frozen, "train_loss": train_loss, "val_miou": scores["miou"],
               "val_mdice": scores["mdice"], "val_pixel_acc": scores["pixel_acc"], "seconds": time.time() - t0}
        history.append(row)

        is_best = scores["miou"] > best_miou
        if is_best:
            best_miou = scores["miou"]
            torch.save({"model": model.state_dict(), "encoder": ENCODER_NAME, "class_names": mapper.names,
                        "epoch": epoch, "val_miou": best_miou, "args": vars(args)}, ckpt_dir / f"{run_name}_best.pth")
        print(f"epoch {epoch:3d}/{args.epochs} {'[frozen]' if frozen else '        '} "
              f"loss {train_loss:.4f} | val mIoU {scores['miou']:.4f} mDice {scores['mdice']:.4f} "
              f"acc {scores['pixel_acc']:.4f} | {row['seconds']:.0f}s {'*' if is_best else ''}")

        # written every epoch: nothing is lost if Colab disconnects
        with open(results_dir / f"{run_name}_history.json", "w") as f:
            json.dump({"args": vars(args), "history": history, "best_val_miou": best_miou}, f, indent=2)

    save_curves(history, results_dir / f"{run_name}_curves.png")
    print(f"done. best val mIoU = {best_miou:.4f} -> {ckpt_dir / (run_name + '_best.pth')}")


if __name__ == "__main__":
    main()