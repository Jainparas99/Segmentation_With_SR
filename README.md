
Improving Segementation with Super Resolution
-- Author Paras Jain
-- Course Capstone

Model Setup

![alt text](<figures/Screenshot 2025-02-22 at 12.34.23 AM.png>)


Results 

![alt text](figures/TINYSAM.png)



Project Structure
Directory/
 src/
      sam_utils.py        # Load SAM, run inference, visualize, compute metrics
      sr_utils.py         # Super-resolution wrapper using EDSR
      deblur_utils.py 
      data_loader.py    
  
 notebooks/
      SAM_Original.ipynb  # SAM on blurry image
      SAM_Baseline.ipynb  # SLIM-SAM on blurry image
      SR_SAM.ipynb        # SLIM-SAM on SR image
      Deblur_SAM.ipynb    # SLIM-SAM on Deblurred image
  
 data/
      iNaturalist_blurry_images/
      ground_truth_masks/
  
 figures/               
 requirements.txt
 README.md

Setup Instructions
git clone https://github.com/yourusername/sam-enhancement.git
cd sam-enhancement
!pip install -r requirements.txt

The Dataset can be downloaded from HuggingFace
from datasets import load_dataset
ds = load_dataset("sxj1215/inaturalist")