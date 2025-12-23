import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from src.model import VAE
from src.dataset import CelebADataset
import json
import os
from multiprocessing import freeze_support

# --- CONFIGURATION ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_PATH = "checkpoints/best_vae.pth"
CONFIG_PATH = "config/config.json"
RESULTS_DIR = "results/failure_gallery"

# Ensure these paths match your folder structure
IMG_DIR = "data/celeba/img_align_celeba/img_align_celeba"
ATTR_PATH = "data/celeba/list_attr_celeba.csv"

def load_config():
    """Loads hyperparameter configuration from JSON."""
    if not os.path.exists(CONFIG_PATH):
        raise FileNotFoundError(f"Config file not found at {CONFIG_PATH}")
    with open(CONFIG_PATH, "r") as f:
        return json.load(f)

def find_worst_samples():
    """
    Scans the test set to find images with the highest reconstruction loss.
    Saves a grid comparing the worst originals vs their reconstructions.
    """
    print(f" Using device: {DEVICE}")
    
    # 1. Load Configuration
    cfg = load_config()
    
    # 2. Load Model
    print(" Loading Model...")
    model = VAE(latent_dim=cfg['latent_dim']).to(DEVICE)
    if os.path.exists(CHECKPOINT_PATH):
        model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
        print(f" Model weights loaded from {CHECKPOINT_PATH}")
    else:
        raise FileNotFoundError(f" Checkpoint not found at {CHECKPOINT_PATH}")
    
    model.eval()

    # 3. Load Test Data
    print(" Loading Test Dataset...")
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])
    
    try:
        dataset = CelebADataset(IMG_DIR, ATTR_PATH, split='test', transform=transform)
    except Exception as e:
        print(f" Warning: 'test' split not found ({e}). Using 'val' split instead.")
        dataset = CelebADataset(IMG_DIR, ATTR_PATH, split='val', transform=transform)
        
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0)
    
    # 4. Scan for failures
    print(f" Scanning {len(dataset)} images to find worst reconstructions...")
    
    scored_images = [] # List to store: {'loss': float, 'orig': tensor, 'recon': tensor}
    
    # Create results directory
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)

    with torch.no_grad():
        for i, batch in enumerate(loader):
            batch = batch.to(DEVICE)
            
            # Forward pass
            recon, _, _ = model(batch)
            
            mse_per_pixel = F.mse_loss(recon, batch, reduction='none')
            
            loss_per_img = mse_per_pixel.view(mse_per_pixel.size(0), -1).sum(dim=1)
            
            # Store results
            for j in range(len(batch)):
                scored_images.append({
                    "loss": loss_per_img[j].item(),
                    "orig": batch[j].cpu(), 
                    "recon": recon[j].cpu()
                })
            
            # Log progress
            if (i + 1) % 50 == 0:
                print(f"   Processed batch {i + 1}/{len(loader)}...")

    # 5. Sort by Loss (Descending: Worst first)
    print(" Sorting images by loss...")
    scored_images.sort(key=lambda x: x["loss"], reverse=True)
    
    # 6. Save Top Failures
    N_FAILURES = 20
    print(f" Saving top {N_FAILURES} worst failures to {RESULTS_DIR}...")
    
    top_failures = scored_images[:N_FAILURES]
    
    grid_list = []
    for idx, item in enumerate(top_failures):
        grid_list.append(item["orig"])
        grid_list.append(item["recon"])
        
    final_grid = torch.stack(grid_list)
    save_path = os.path.join(RESULTS_DIR, "failure_gallery.png")
    
    save_image(final_grid, save_path, nrow=8, normalize=True, padding=2, pad_value=1)
    
    print(f"\n DONE! Failure gallery saved at: {save_path}")
    print("   (Left: Original, Right: Reconstruction)")

if __name__ == "__main__":
    freeze_support()
    try:
        find_worst_samples()
    except Exception as e:
        print(f"\n Critical Error: {e}")