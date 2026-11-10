"""model.py - U-Net with a ResNet34 encoder pretrained on ImageNet.

The encoder weights come from the Hugging Face Hub: `timm` downloads
"timm/resnet34.a1_in1k" and `segmentation_models_pytorch` plugs it into a U-Net
decoder (the "tu-" prefix stands for "timm universal encoder").
These weights expect ImageNet mean/std, which is what dataset.py applies.

Fine-tuning (see train.py): the decoder is randomly initialised, the encoder is not.
    1) warm-up   : encoder frozen, only the decoder learns
    2) fine-tune : everything is unfrozen, with a smaller learning rate for the encoder
"""

import segmentation_models_pytorch as smp

ENCODER_NAME = "tu-resnet34.a1_in1k"


def build_model(num_classes):
    """U-Net -> logits of shape [B, num_classes, H, W] (no softmax: CrossEntropyLoss does it)."""
    kwargs = dict(
        encoder_name=ENCODER_NAME,
        in_channels=3,
        classes=num_classes
    )

    try:
        return smp.Unet(encoder_weights="imagenet", **kwargs)
    except (KeyError, ValueError, TypeError):
        return smp.Unet(encoder_weights=True, **kwargs) # if not able to get the weights with "imagenet"


def set_encoder_trainable(model, trainable):
    """Freeze (False) or unfreeze (True) every encoder parameter."""
    for p in model.encoder.parameters():
        p.requires_grad = trainable


def get_param_groups(model, lr, encoder_lr_mult=0.1, weight_decay=1e-4):
    """Two parameter groups: decoder + head at `lr`, encoder at `lr * encoder_lr_mult`."""
    encoder_params = list(model.encoder.parameters())
    encoder_ids = {id(p) for p in encoder_params}
    other_params = [p for p in model.parameters() if id(p) not in encoder_ids]
    return [
        {"params": other_params, "lr": lr, "weight_decay": weight_decay},
        {"params": encoder_params, "lr": lr * encoder_lr_mult, "weight_decay": weight_decay},
    ]


def count_params(model):
    """(total, trainable) number of parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable