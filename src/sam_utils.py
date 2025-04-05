# src/sam_utils.py

import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from transformers import SamModel, SamProcessor


def load_sam_model(model_name, device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SamModel.from_pretrained(model_name).to(device)
    processor = SamProcessor.from_pretrained(model_name)
    return model, processor, device


def run_sam_inference(image_path, model, processor, device):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    masks = outputs.pred_masks.squeeze().cpu().numpy()
    return image, masks


def visualize_segmentation(image, mask, title="Segmentation Mask"):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(mask[0], cmap="cool")
    plt.title(title)
    plt.axis("off")
    plt.show()


def dice_coefficient(pred_mask, true_mask, smooth=1e-6):
    pred_mask = (pred_mask > 0.5).astype(np.float32)
    true_mask = (true_mask > 0.5).astype(np.float32)
    intersection = np.sum(pred_mask * true_mask)
    sum_masks = np.sum(pred_mask) + np.sum(true_mask)
    return (2. * intersection + smooth) / (sum_masks + smooth)

def load_mask(mask_path):

    mask = Image.open(mask_path).convert("L")  # Grayscale
    mask = np.array(mask) / 255.0
    return (mask > 0.5).astype(np.float32)


def visualize_prediction_vs_gt(image, pred_mask, true_mask, pred_title="Predicted Mask"):
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(image)
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(pred_mask, cmap="cool")
    plt.title(pred_title)
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(true_mask, cmap="gray")
    plt.title("Ground Truth Mask")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

def iou_score(pred_mask, true_mask, pred_thresh=0.5, true_thresh=128, smooth=1e-6):
    if np.max(true_mask) <= 1:
        true_thresh = 0.5

    pred_mask_bin = (pred_mask > pred_thresh).astype(np.float32)
    true_mask_bin = (true_mask > true_thresh).astype(np.float32)

    intersection = np.sum(pred_mask_bin * true_mask_bin)
    union = np.sum(np.logical_or(pred_mask_bin, true_mask_bin))

    return (intersection + smooth) / (union + smooth)