import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from src.model import VAE
from src.dataset import CelebADataset
import json
import os
import time
from multiprocessing import freeze_support

# --- CONFIGURATION ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_PATH = "checkpoints/best_vae.pth"
CONFIG_PATH = "config/config.json"

IMG_DIR = "data/celeba/img_align_celeba/img_align_celeba"
ATTR_PATH = "data/celeba/list_attr_celeba.csv"

def load_config():
    """Loads hyperparameter configuration from JSON."""
    with open(CONFIG_PATH, "r") as f:
        return json.load(f)

def load_model(config):
    """Initializes the VAE and loads trained weights."""
    model = VAE(latent_dim=config['latent_dim']).to(DEVICE)
    if os.path.exists(CHECKPOINT_PATH):
        model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
        print(f" Model loaded successfully from {CHECKPOINT_PATH}")
    else:
        raise FileNotFoundError(f" Checkpoint not found at {CHECKPOINT_PATH}")
    model.eval()
    return model

def get_test_loader(batch_size=32):
    """Loads the Test dataset."""
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])
    
    print(" Loading Test Dataset...")
    try:
        dataset = CelebADataset(IMG_DIR, ATTR_PATH, split='test', transform=transform)
    except Exception as e:
        print(f" Warning: Could not load 'test' split ({e}). Falling back to 'val'.")
        dataset = CelebADataset(IMG_DIR, ATTR_PATH, split='val', transform=transform)
        
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    return loader, len(dataset)

def evaluate(model, loader, dataset_size):
    """
    Runs the model on the test set and calculates quantitative metrics.
    """
    print(f"\n Starting Quantitative Evaluation on {dataset_size} images...")
    start_time = time.time()
    
    total_loss = 0.0
    total_recon_loss = 0.0
    total_kld_loss = 0.0
    
    with torch.no_grad():
        for i, batch_images in enumerate(loader):
            batch_images = batch_images.to(DEVICE)
            
            # 1. Forward pass
            recon_images, mu, logvar = model(batch_images)
            
            # 2. Calculate Loss Components
            # Reconstruction Loss (MSE) - Summed over the batch
            # We use reduction='sum' to match the training logic
            BCE = F.mse_loss(recon_images, batch_images, reduction='sum')
            
            # KL Divergence - Summed over the batch
            KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            
            # Accumulate metrics
            total_recon_loss += BCE.item()
            total_kld_loss += KLD.item()
            total_loss += (BCE.item() + KLD.item())
            
            # Print progress every 100 batches
            if (i + 1) % 100 == 0:
                print(f"   Processed batch {i + 1}/{len(loader)}...")

    # 3. Calculate Average Loss per Image
    avg_loss = total_loss / dataset_size
    avg_recon = total_recon_loss / dataset_size
    avg_kld = total_kld_loss / dataset_size
    
    elapsed_time = time.time() - start_time

    # 4. Print Final Report
    print("\n" + "="*50)
    print(f" FINAL TEST RESULTS")
    print("="*50)
    print(f"  Time taken:          {elapsed_time:.2f} sec")
    print(f"  Images processed:    {dataset_size}")
    print("-" * 50)
    print(f" Total Average Loss:   {avg_loss:.4f}")
    print("-" * 50)
    print(f" Reconstruction Loss:  {avg_recon:.4f}  (Image fidelity/Quality)")
    print(f" KL Divergence:        {avg_kld:.4f}   (Latent space regularization)")
    print("="*50 + "\n")

if __name__ == "__main__":
    freeze_support()
    
    try:
        # Load Config
        cfg = load_config()
        print(f"  Using device: {DEVICE}")
        
        # Load Model
        vae = load_model(cfg)
        
        # Load Data
        test_loader, num_images = get_test_loader(batch_size=32)
        
        # Run Evaluation
        evaluate(vae, test_loader, num_images)
        
    except Exception as e:
        print(f"\n Critical Error: {e}")