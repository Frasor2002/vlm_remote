import os
import json
import torch
import random
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

DATA_PATH = os.path.join("mnt", "cimec-storage6", "users", "francesco.sorrentino","data")


def convert_to_heatmap(sal_tensor: torch.Tensor, outlier_perc: float = 1.0) -> Image.Image:
  """
  Converts a raw attribution tensor into a Captum-style heatmap PIL Image.
  """
  sal_2d = torch.sum(torch.abs(sal_tensor), dim=0).detach().cpu().numpy()
  
  if outlier_perc > 0:
    vmax = np.percentile(sal_2d, 100 - outlier_perc)
    vmin = sal_2d.min()
    sal_2d = np.clip(sal_2d, vmin, vmax)
      
  vmax = sal_2d.max()
  vmin = sal_2d.min()
  if vmax > vmin:
    sal_2d = (sal_2d - vmin) / (vmax - vmin)
  else:
    sal_2d = np.zeros_like(sal_2d)
      
  cmap = plt.get_cmap('jet')
  heatmap_rgba = cmap(sal_2d)
  
  heatmap_rgb = (heatmap_rgba[:, :, :3] * 255).astype(np.uint8)
  
  return Image.fromarray(heatmap_rgb)


def load_saved_dataset(data_name):
  dataset_dir = os.path.join(DATA_PATH, data_name)
  
  if not os.path.exists(dataset_dir):
    raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")
      
  samples = []
  
  sample_folders = [f for f in os.listdir(dataset_dir) if f.startswith("sample_")]
  sample_folders.sort(key=lambda x: int(x.split('_')[1]))
  
  for folder in sample_folders:
    sample_dir = os.path.join(dataset_dir, folder)
    
    img_path = os.path.join(sample_dir, "img.png")
    sal_path = os.path.join(sample_dir, "sal.png")
    info_path = os.path.join(sample_dir, "info.json")
    
    # Load Images as PIL (keep them as PIL for the HF pipeline)
    try:
      img_pil = Image.open(img_path).convert("RGB")
      sal_pil = Image.open(sal_path).convert("RGB")
    except (FileNotFoundError, OSError) as e:
      print(f"Warning: Skipping {folder} due to missing or corrupted image files. ({e})")
      continue
        
    try:
      with open(info_path, "r") as f:
        info_dict = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
      print(f"Warning: Skipping {folder} due to missing or invalid info.json. ({e})")
      continue
        
    samples.append({
      "img": img_pil,
      "saliency": sal_pil,
      "label": info_dict.get("ground_truth_label", "Unknown"),
      "pred": info_dict.get("prediction", "Unknown")
    })
      
  print(f"Successfully loaded {len(samples)} samples from '{dataset_dir}'")
  return samples
