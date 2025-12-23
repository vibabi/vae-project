import torch
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torchvision import transforms
from src.model import VAE
from src.dataset import CelebADataset 
import json
import os

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_PATH = "checkpoints/best_vae.pth"
CONFIG_PATH = "config/config.json"
RESULTS_DIR = "results/inference_test_set" 
IMG_DIR = "data/celeba/img_align_celeba/img_align_celeba"         
ATTR_PATH = "data/celeba/list_attr_celeba.csv"   

os.makedirs(RESULTS_DIR, exist_ok=True)

def load_config():
    """Loads hyperparameter configuration from JSON."""
    with open(CONFIG_PATH, "r") as f:
        return json.load(f)

def load_model(config):
    """Initializes the VAE and loads trained weights."""
    model = VAE(latent_dim=config['latent_dim']).to(DEVICE)
    
    if os.path.exists(CHECKPOINT_PATH):
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
        model.load_state_dict(checkpoint)
        print(f"Model loaded successfully from {CHECKPOINT_PATH}")
    else:
        raise FileNotFoundError(f"Checkpoint not found at {CHECKPOINT_PATH}")
    
    model.eval()
    return model

def get_test_loader(batch_size=32):
    """
    Loads the TEST split of the dataset.
    These are images the model has NEVER seen before.
    """
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])
    
    try:
        dataset = CelebADataset(IMG_DIR, ATTR_PATH, split='test', transform=transform)
        print(f"Loaded Test Dataset: {len(dataset)} images")
    except Exception as e:
        print(f"Warning: Could not load 'test' split ({e}). Falling back to 'val'.")
        dataset = CelebADataset(IMG_DIR, ATTR_PATH, split='val', transform=transform)

    # Shuffle=True ensures we see different faces every time we run the script
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return loader, dataset

# --- TEST 1: RECONSTRUCTION (UNSEEN DATA) ---
def test_reconstruction(model, loader):
    print("\nTest 1: Reconstruction on unseen data...")
    
    # Get a batch of real images
    real_images = next(iter(loader))
    real_images = real_images[:8].to(DEVICE) # Take top 8

    with torch.no_grad():
        # Pass through the model
        recon_images, _, _ = model(real_images)
        
        # Stack images: Top row = Original, Bottom row = VAE Reconstruction
        comparison = torch.cat([real_images, recon_images])
        
        save_path = f"{RESULTS_DIR}/1_test_reconstruction.png"
        save_image(comparison, save_path, nrow=8)
        print(f"   Saved result to: {save_path}")

# --- TEST 2: GENERATION (FROM NOISE) ---
def test_generation(model, num=16):
    print("\nTest 2: Generating new faces from random noise...")
    
    with torch.no_grad():
        # Sample random noise from Normal Distribution N(0, 1)
        z = torch.randn(num, 128).to(DEVICE)
        
        # Decode the noise into images
        generated = model.decode(z)
        
        save_path = f"{RESULTS_DIR}/2_random_generation.png"
        save_image(generated, save_path, nrow=4)
        print(f"   Saved result to: {save_path}")

# --- TEST 3: REAL IMAGE MORPHING ---
def test_real_morphing(model, loader, steps=10):
    print("\nTest 3: Morphing between two REAL test people...")
    
    # Get two real images
    batch = next(iter(loader))
    img1 = batch[0].unsqueeze(0).to(DEVICE) # Person A
    img2 = batch[1].unsqueeze(0).to(DEVICE) # Person B
    
    with torch.no_grad():
        # 1. Encode real images to Latent Space
        mu1, _ = model.encode(img1) 
        mu2, _ = model.encode(img2)
        
        # 2. Interpolate (Walk) from Person A to Person B
        interpolated_images = []
        
        # Add original Image A
        interpolated_images.append(img1)
        
        # Generate intermediate steps
        alphas = torch.linspace(0, 1, steps).to(DEVICE)
        for alpha in alphas:
            # Linear Interpolation: z = z1*(1-a) + z2*a
            z_step = mu1 * (1 - alpha) + mu2 * alpha
            
            # Decode the intermediate latent vector
            img_step = model.decode(z_step)
            interpolated_images.append(img_step)
            
        # Add target Image B
        interpolated_images.append(img2)
        
        # Concatenate all steps into one strip
        final_strip = torch.cat(interpolated_images, dim=0)
        
        save_path = f"{RESULTS_DIR}/3_real_morphing.png"
        save_image(final_strip, save_path, nrow=steps+2)
        print(f"   Saved result to: {save_path}")

if __name__ == "__main__":
    import random 
    
    print(f"Using device: {DEVICE}")

    # 1. Load Data
    test_loader, test_dataset = get_test_loader(batch_size=16) 
    print(f"Loaded Test Dataset: {len(test_dataset)} images")

    # 2. Load Model
    model = VAE().to(DEVICE)
    if os.path.exists(CHECKPOINT_PATH):
        model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
        print(f"Model loaded successfully from {CHECKPOINT_PATH}")
    else:
        raise FileNotFoundError(f"Checkpoint not found at {CHECKPOINT_PATH}")

    model.eval()

    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)

    # TEST 1: RECONSTRUCTION (5 Examples)
    print("\n Test 1: Reconstruction (Saving 5 batches)...")
    
    test_iter = iter(test_loader)
    
    for i in range(5): 
        try:
            real_images = next(test_iter).to(DEVICE)
            
            with torch.no_grad():
                recon_images, _, _ = model(real_images)
            
            comparison = torch.cat([real_images, recon_images])
            
            save_path = f"{RESULTS_DIR}/test1_reconstruction_{i+1}.png"
            save_image(comparison.cpu(), save_path, nrow=real_images.size(0), normalize=True)
            print(f"   -> Saved: {save_path}")
            
        except StopIteration:
            break

    # TEST 2: GENERATION (5 Examples)
    print("\n Test 2: Generating new faces (Saving 5 grids)...")
    
    for i in range(5):
        with torch.no_grad():
            z = torch.randn(32, 128).to(DEVICE) 
            generated_images = model.decode(z)
            
            save_path = f"{RESULTS_DIR}/test2_generated_{i+1}.png"
            save_image(generated_images.cpu(), save_path, nrow=8, normalize=True)
            print(f"   -> Saved: {save_path}")

    # TEST 3: MORPHING (5 Examples)
    print("\n Test 3: Morphing between random pairs (Saving 5 examples)...")
    
    for i in range(5):
        idx1 = random.randint(0, len(test_dataset) - 1)
        idx2 = random.randint(0, len(test_dataset) - 1)
        
        img1 = test_dataset[idx1].unsqueeze(0).to(DEVICE)
        img2 = test_dataset[idx2].unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            mu1, _ = model.encode(img1)
            mu2, _ = model.encode(img2)
            
            steps = 12
            z_interpolated = []
            for alpha in torch.linspace(0, 1, steps):
                z = mu1 * (1 - alpha) + mu2 * alpha
                z_interpolated.append(z)
            
            z_interpolated = torch.cat(z_interpolated, dim=0)
            
            interpolated_imgs = model.decode(z_interpolated)
            
            final_strip = torch.cat([img1, interpolated_imgs, img2], dim=0)
            
            save_path = f"{RESULTS_DIR}/test3_morphing_{i+1}.png"
            save_image(final_strip.cpu(), save_path, nrow=steps+2, normalize=True)
            print(f"   -> Saved: {save_path} (Index {idx1} -> {idx2})")

    print(f"\n All done! Check the folder: {RESULTS_DIR}")