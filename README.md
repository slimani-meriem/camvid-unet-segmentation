# CamVid semantic segmentation with a fine-tuned U-Net
 
Semantic segmentation of driving scenes ([CamVid](https://www.kaggle.com/datasets/carlolepelaars/camvid)) with a U-Net whose encoder is a ResNet34 pretrained on ImageNet and fine-tuned on the task. Built with PyTorch.
 
> **Status: work in progress.** The data pipeline, model, metrics and training script are implemented, and the first training runs are in progress. **There are no results in this repository yet**, and the evaluation script is not written. The sections below describe what the code does and the choices behind it; the last section lists what comes next.
 
## Why this project
 
I wanted to go beyond running a segmentation tutorial and understand the decisions that actually change the outcome on a small, imbalanced dataset: how the classes are defined, how the pretrained encoder is fine-tuned, and how the metrics are computed. This repository is where I work through them, so I try to document the reasons behind each choice, not only the choice itself.
 
## Dataset
 
I use the Kaggle version of CamVid (701 street-level images, 960×720, with pixel-wise RGB masks and a `class_dict.csv`):
 
| Split | Images |
|---|---|
| train | 369 |
| val | 100 |
| test | 232 |
 
A few points worth knowing:
 
- **31 classes, not 11.** The masks come with 32 classes. `Void` (unlabelled pixels) is ignored in both the loss and the metrics, which leaves **31 classes to learn**. Most papers use the 11-class version introduced with SegNet instead, so I also report 11-class scores (see [Metrics](#metrics)). Training on 31 classes keeps the fine-grained analysis, but many of those classes cover very few pixels.
- **Resizing.** Images are resized to 352×480 (both divisible by 32, as a 5-level U-Net requires). Masks use nearest-neighbour interpolation so that class indices are never blended. The aspect ratio changes slightly (4:3 becomes about 1.36:1).
- **Augmentation (training split only):** horizontal flip, random scale-and-crop, and brightness/contrast jitter. Geometric transforms are applied to the image and the mask together; colour transforms only to the image.
- **Normalisation** uses ImageNet statistics, which is what the pretrained encoder expects.
## Model
 
`segmentation_models_pytorch` U-Net with a ResNet34 encoder. The encoder weights (`timm/resnet34.a1_in1k`) are downloaded from the Hugging Face Hub through `timm`. **The decoder comes from the library**: I did not write it myself (see the perspectives below).
 
## Training strategy
 
The decoder and the segmentation head start from random weights, whereas the encoder already contains useful visual features. Training everything at once would send large, noisy gradients from the random decoder into the pretrained encoder. So fine-tuning is done in two phases:
 
1. **Warm-up (first n epochs, `--freeze-epochs`)**: encoder frozen (and kept in eval mode so its BatchNorm statistics do not move). Only the decoder learns.
2. **Fine-tuning (remaining epochs)**: everything is unfrozen, with a learning rate 10× smaller for the encoder (`1e-4`) than for the decoder (`1e-3`).
Other settings: AdamW (weight decay `1e-4`), cosine learning-rate decay, mixed precision (fp16 forward pass, loss in fp32), batch size 8, seed 42. The loss is cross-entropy with ignored pixels excluded; Cross-Entropy + Dice is available with `--loss ce_dice`. The best checkpoint is selected on the **validation** mIoU. The test set is meant to be used only once, for the final evaluation.
 
## Metrics
 
- IoU and Dice are computed **per class from a confusion matrix accumulated over the whole split**, not averaged image by image: rare classes are absent from most images, so a per-image average would be meaningless.
- **mIoU is averaged only over the classes present in the ground truth** of the split. A class that never appears has no defined IoU, and counting it as 0 would artificially lower the score.
- **31 → 11 classes.** The 11-class scores are obtained by merging the rows and columns of the 31×31 confusion matrix after evaluation (no retraining). The grouping follows the one used in MathWorks' `camvidPixelLabelIDs` (itself based on the SegNet procedure) and reproduced in [lih627/CamVid](https://github.com/lih627/CamVid). For example, `Car` also absorbs `SUVPickupTruck`, `Truck_Bus`, `Train` and `OtherMoving`. Groupings vary slightly between codebases, so any 11-class number should be read with this table in mind.
- Training on 31 classes and merging afterwards is not the same as training directly on 11 classes, so the 11-class figure is an order of magnitude to situate the model.
The metric functions are unit-tested on hand-computed examples (see `src/metrics.py`); the training and model code have not been validated beyond the first runs.
 
## Running it
 
```bash
pip install -r requirements.txt
```
 
Download CamVid from Kaggle and extract it into `data/CamVid/` (the folder must contain `train/`, `train_labels/`, `val/`, `val_labels/`, `test/`, `test_labels/` and `class_dict.csv`). Then:
 
```bash
python -m src.train --data-dir data/CamVid --epochs 30
```
 
Useful options: `--freeze-epochs`, `--encoder-lr-mult`, `--loss {ce,ce_dice}`, `--ckpt-dir`, `--results-dir`. A GPU is strongly recommended; I train on Google Colab (T4). Checkpoints are not versioned.
 
## Project structure
 
```
src/
  dataset.py   # CamVid dataset, RGB mask -> class index mapping, augmentation, dataloaders
  model.py     # U-Net with pretrained ResNet34 encoder, freezing and per-group learning rates
  metrics.py   # confusion matrix, IoU / Dice / mIoU, 31 -> 11 class grouping
  train.py     # two-phase fine-tuning, validation, checkpointing
  evaluate.py  # test-set evaluation (to do)
notebooks/     # Colab setup, dataset exploration, training
```
 
## Perspectives
 
1. **Complete the evaluation and publish results.** Evaluate the best checkpoint once on the test set with 31 and 11 classes, report per-class IoU and Dice, add the confusion matrix and qualitative predictions (image / ground truth / prediction), and potentially compare with published numbers.
2. **Study the class imbalance.** Measure the pixel frequency of each class, relate it to the per-class IoU, and compare Cross-Entropy, Cross-Entropy + Dice and a class-weighted loss, to see how much the rare classes improve and at what cost for the frequent ones.
3. **Go deeper on the model and the fine-tuning.** Replace the library decoder with my own implementation of the U-Net decoder, ablate the fine-tuning choices (number of frozen epochs, encoder/decoder learning-rate ratio, training from scratch versus pretrained), and try other pretrained encoders.

 