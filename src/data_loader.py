# src/data_loader.py

import os
import random
import numpy as np
from PIL import Image
import cv2
from datasets import load_dataset

def download_and_subset_dataset(name="sxj1215/inaturalist", size=5000, save_dir="iNaturalist_subset"):
    ds = load_dataset(name)
    subset = ds["train"].shuffle(seed=42).select(range(size))
    subset.save_to_disk(save_dir)
    return subset

def save_images_from_subset(subset, output_dir="iNaturalist_subset_images"):
    os.makedirs(output_dir, exist_ok=True)
    for i, item in enumerate(subset):
        image = item["images"][0]
        label = item["messages"]
        image_path = os.path.join(output_dir, f"{label}_{i}.jpg")
        image.save(image_path)

def apply_gaussian_blur(image, kernel_size=10):
    return cv2.GaussianBlur(np.array(image), (kernel_size, kernel_size), 0)

def apply_motion_blur(image, kernel_size=30):
    kernel = np.zeros((kernel_size, kernel_size))
    kernel[int((kernel_size - 1) / 2), :] = np.ones(kernel_size) / kernel_size
    return cv2.filter2D(np.array(image), -1, kernel)

def blur_image_folder(input_dir, output_dir="iNaturalist_blurry_images"):
    os.makedirs(output_dir, exist_ok=True)
    for filename in os.listdir(input_dir):
        if filename.endswith((".jpg", ".png")):
            img_path = os.path.join(input_dir, filename)
            img = Image.open(img_path)
            blurred = apply_gaussian_blur(img) if random.choice([True, False]) else apply_motion_blur(img)
            Image.fromarray(blurred).save(os.path.join(output_dir, filename))