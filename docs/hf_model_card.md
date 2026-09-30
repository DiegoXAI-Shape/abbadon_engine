---
license: other
license_name: research-and-education-only
library_name: pytorch
tags:
  - torchscript
  - libtorch
  - image-segmentation
  - image-classification
  - knowledge-distillation
  - convnextv2
  - sam2
  - shortcut-learning
---

# Abbadon — Daowa-maad & Mendicant Bias

Two TorchScript models that work as a pipeline to classify **cats vs dogs by looking at the animal,
not at the background**:

- **Daowa-maad** — a pet segmentation oracle distilled from **SAM 2**.
- **Mendicant Bias** — a cat/dog classifier that receives the image **plus** Daowa-maad's mask as
  soft attention.

```
image ─► Daowa-maad ─► sigmoid mask ─┐
  │                                   ├─► Mendicant Bias ─► cat / dog
  └──────── normalized RGB ───────────┘
```

## Why not just a ResNet?

A plain CNN reaches high accuracy on cats vs dogs in a couple of epochs — but Grad-CAM showed it was
right for the wrong reasons: **"dog if there is grass, cat if there is a sofa"**. Most dog photos are
taken in parks and most cat photos indoors, so the network learned the background instead of the
animal (*shortcut learning*).

Abbadon attacks that bias by giving the classifier an explicit hint of **where the animal is**, while
still letting it disagree with that hint when the hint is wrong.

## Files

| File | Model | Input | Output |
|---|---|---|---|
| `Daowa_Oracle_Frozen.pt` | Daowa-maad | RGB `[1, 3, 384, 384]` | logits `[1, 1, 384, 384]` (apply sigmoid) |
| `mendicant_bias_cpp.pt` | Mendicant Bias | `forward(rgb [1, 3, 384, 384], mask [1, 1, 384, 384])` | logits `[1, 2]` — `0 = cat`, `1 = dog` |

Both are exported with `torch.jit.trace`, so they run from Python **or C++ (LibTorch)** without the
original model code.

> Use `Daowa_Oracle_Frozen.pt` together with Mendicant Bias: it is the checkpoint that produced the
> masks Mendicant Bias was trained on.

## Preprocessing (both models, same tensor)

1. Read as RGB (`cv2.imread` + `BGR → RGB`)
2. Resize to **384×384**, bilinear (`cv2.INTER_LINEAR`)
3. `/ 255`, then ImageNet normalization — mean `[0.485, 0.456, 0.406]`, std `[0.229, 0.224, 0.225]`
4. HWC → CHW, add batch dimension

The mask given to Mendicant Bias is Daowa-maad's **sigmoid output without thresholding**
(soft probabilities in `[0, 1]`).

## Usage (Python)

```python
import cv2, torch
from huggingface_hub import hf_hub_download

repo = "DiegoXAI-Shape/abbadon-engine"
daowa = torch.jit.load(hf_hub_download(repo, "Daowa_Oracle_Frozen.pt")).eval()
mendicant = torch.jit.load(hf_hub_download(repo, "mendicant_bias_cpp.pt")).eval()

img = cv2.cvtColor(cv2.imread("pet.jpg"), cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (384, 384), interpolation=cv2.INTER_LINEAR)
x = torch.from_numpy(img).permute(2, 0, 1).float().div(255)
mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
x = ((x - mean) / std).unsqueeze(0)

with torch.no_grad():
    mask = torch.sigmoid(daowa(x))       # [1, 1, 384, 384]
    logits = mendicant(x, mask)          # [1, 2]

print(["cat", "dog"][logits.argmax(1).item()])
```

## Usage (C++ / LibTorch)

See **[abbadon_engine](https://github.com/DiegoXAI-Shape/abbadon_engine)**: a C++ inference engine
that runs the full pipeline (OpenCV preprocessing → Daowa-maad → Mendicant Bias) on the GPU.

## Training

Training code: **[Abbadon](https://github.com/DiegoXAI-Shape/Abbadon)**.

### Daowa-maad (segmentation)

- **Encoder:** ConvNeXtV2-Tiny (`convnextv2_tiny.fcmae_ft_in22k_in1k`, via `timm`).
- **Data:** Oxford-IIIT Pet. Its trimaps have 3 classes (pet / background / border); the border class
  was a small, noisy minority that made the model unsure and produced very thick borders, so the task
  was reframed as **binary: pet vs not pet**, like SAM.
- **Teacher:** SAM 2. Since SAM 2 is promptable (not automatic), a detector (YOLOv8) found the cat or
  dog and its box center was used as SAM 2's point prompt. This produced extra masks to grow the
  dataset (it fails on some unusual poses where the center falls outside the animal).
- **Knowledge distillation:** Daowa-maad learns from SAM 2's soft predictions — Hinton's
  *dark knowledge* — not only from hard labels (KL divergence, temperature 2.0).
- **Loss schedule (burn-in):** the weights shift linearly during training — KL to the teacher
  1.0 → 0.1, BCE 1.0 → 0, Dice and Boundary loss 0 → 1. Early on it trusts the teacher; later it
  focuses on the ground truth and the object's shape and borders.
- **Adversarial fine-tuning (hard negatives):** a later stage mixed Oxford-IIIT Pet positives with
  **hard negatives from ADE20K** — "trap textures" that look like a pet but are not (e.g. fur coats),
  whose correct mask is all zeros. A batch sampler kept a fixed positive/negative ratio per batch,
  and the loss was Dice + Boundary (signed distance maps), without the teacher. Checkpoints were
  selected by `0.7 × IoU(negatives) + 0.3 × IoU(positives)`, prioritizing **not hallucinating pets**.
  The released `Daowa_Oracle_Frozen.pt` already includes this adversarial stage, and it is the
  oracle that was frozen to train Mendicant Bias.
- **Training details:** AMP (bfloat16), gradient clipping, gradient accumulation.

### Mendicant Bias (classification)

- **Backbone:** ConvNeXtV2-Atto (`timm`), modified to accept **4 channels** (RGB + mask).
- **Data:** Dogs vs Cats (Kaggle). Classes: `Cat = 0`, `Dog = 1`.
- **Attention Gate:** before the classifier, a small conv block looks at RGB + mask and predicts a
  correction in `[-1, 1]` (`tanh`) that is **added** to Daowa-maad's mask. The classifier can
  "disobey" its teacher where the mask is wrong, but an **L2 penalty** on the correction
  (`λ = 0.01`) makes disobeying costly, so it only does it when it pays off.
- **Drop-RGB (p = 0.15):** during training the colors are randomly zeroed, forcing the classifier to
  decide from the **shape** of the mask alone. This fights color/texture shortcuts.
- **Validation accuracy:** 99.43%.

## Limitations

- Trained only on cats and dogs. Other animals, or images with several pets, are out of distribution.
- Daowa-maad works at 384×384; masks upscaled to large photos lose fine detail (fur, whiskers).
- The SAM 2 pseudo-labels inherit the detector-center heuristic's failures on unusual poses.
- Mendicant Bias depends on Daowa-maad's mask; a very wrong mask can mislead it.

## Intended use & license

**Research and education only** — this is a personal portfolio project, not a commercial product.

- **Code** ([abbadon_engine](https://github.com/DiegoXAI-Shape/abbadon_engine),
  [Abbadon](https://github.com/DiegoXAI-Shape/Abbadon)): MIT License.
- **Weights:** trained on datasets with non-commercial terms (Dogs vs Cats / Asirra, ADE20K), so
  **commercial use of the weights is not permitted** by those terms.

## Acknowledgements

This project stands on the work of others:

- **Oxford-IIIT Pet Dataset** (CC BY-SA 4.0) — O. M. Parkhi, A. Vedaldi, A. Zisserman,
  C. V. Jawahar. *Cats and Dogs.* IEEE CVPR, 2012.
- **Dogs vs Cats** (Kaggle), built from **Asirra** — J. Elson, J. R. Douceur, J. Howell, J. Saul.
  *Asirra: A CAPTCHA that Exploits Interest-Aligned Manual Image Categorization.* ACM CCS, 2007.
- **ADE20K** — B. Zhou, H. Zhao, X. Puig, S. Fidler, A. Barriuso, A. Torralba.
  *Scene Parsing through ADE20K Dataset.* IEEE CVPR, 2017.
- **SAM 2** (Apache 2.0) — N. Ravi et al. *SAM 2: Segment Anything in Images and Videos.* Meta AI, 2024.
- **YOLOv8** (Ultralytics, AGPL-3.0) — used only to generate SAM 2 prompts during data preparation.
- **ConvNeXt V2** backbones via [timm](https://github.com/huggingface/pytorch-image-models).

```bibtex
@InProceedings{parkhi12a,
  author    = "Omkar M. Parkhi and Andrea Vedaldi and Andrew Zisserman and C. V. Jawahar",
  title     = "Cats and Dogs",
  booktitle = "IEEE Conference on Computer Vision and Pattern Recognition",
  year      = "2012",
}
```
