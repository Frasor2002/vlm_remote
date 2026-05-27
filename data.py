import os
import json
import torch
from PIL import Image

DATA_PATH = os.path.join("mnt", "cimec-storage6", "users", "francesco.sorrentino","data")


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
    mask_path = os.path.join(sample_dir, "mask.pt")
    
    # Load Images as PIL (keep them as PIL for the HF pipeline)
    try:
      img_pil = Image.open(img_path).convert("RGB")
      sal_pil = Image.open(sal_path).convert("RGB")
      mask_tensor = torch.load(mask_path)
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
      "mask": mask_tensor,
      "label": info_dict["ground_truth_label"],
      "pred": info_dict["prediction"],
      "category": info_dict["category"]
    })
      
  print(f"Successfully loaded {len(samples)} samples from '{dataset_dir}'")
  return samples


