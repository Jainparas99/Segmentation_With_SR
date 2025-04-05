
**Improving Segementation with Super Resolution**

**Author:- Paras Jain**

A current Masters student in RIT in the field of Artificial Intelligence

Course Capstone

Introduction :-

In this project I intend to enhance segementation for blurry images and also for animal which get camouflaged easily with their sourroundings making it tough for SAM models to segment them with their background.


Model Setup

![alt text](<figures/Screenshot 2025-02-22 at 12.34.23 AM.png>)

The model is setup with an SR model as a wrapper class over the SAM model. This will enhance the image before it get input to the model.  


Baseline Figures

![alt text](figures/TINYSAM.png)

Segmentation Using Tiny SAM

![alt text]





Setup Instructions
git clone https://github.com/yourusername/sam-enhancement.git
cd sam-enhancement
!pip install -r requirements.txt

The Dataset can be downloaded from HuggingFace
from datasets import load_dataset
ds = load_dataset("sxj1215/inaturalist")
