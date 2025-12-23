import torch
import torch.optim as optim
import json
import os
import time
import matplotlib.pyplot as plt
import torchvision.utils as vutils
from src.dataset import get_dataloaders
from src.model import VAE, loss_function


def save_live_plot(history, filename='results/training_log.png'):
    """
    Plots a graph based on the accumulated step history and saves it to a file.
    This function is called at the end of every epoch to update the visual log.
    """
    steps = history['steps']
    
    if len(steps) < 2:
        return

    plt.style.use('ggplot')
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
    
    fig.suptitle(f'Training Progress (Updated: Step {steps[-1]})', fontsize=16)

    # 1. Total Loss
    ax1.plot(steps, history['total_loss'], label='Total Loss', color='#1f77b4', alpha=0.8)
    ax1.set_ylabel('ELBO Loss')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    # 2. Reconstruction Loss
    ax2.plot(steps, history['recon_loss'], label='Reconstruction (MSE)', color='#2ca02c', alpha=0.8)
    ax2.set_ylabel('MSE Loss')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)

    # 3. KL Divergence
    ax3.plot(steps, history['kld_loss'], label='KL Divergence', color='#9467bd', alpha=0.8)
    ax3.set_ylabel('KLD')
    ax3.set_xlabel('Global Training Steps')
    ax3.legend(loc='upper right')
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close(fig) 

def save_comparison_images(model, loader, epoch, device, result_dir='results'):
    """
    Saves a grid comparing Original vs Reconstructed images.
    """
    model.eval()
    with torch.no_grad():
        try:
            # Get one batch from validation loader
            data = next(iter(loader))
            data = data.to(device)
            
            # Reconstruct
            recon_batch, _, _ = model(data)
            
            # Create a grid: Top rows = Original, Bottom rows = Reconstructed
            n = min(data.size(0), 8)
            comparison = torch.cat([data[:n], recon_batch[:n]])
            
            # Save to file
            filename = os.path.join(result_dir, f'reconstruction_epoch_{epoch}.png')
            vutils.save_image(comparison.cpu(), filename, nrow=n, padding=2, normalize=False)
        except Exception as e:
            print(f"Warning: Could not save images. Error: {e}")


def train_one_epoch(model, dataloader, optimizer, device, beta, epoch_index, history, log_interval=100):
    """
    Runs training for one epoch.
    Updates the 'history' dictionary with loss values every 'log_interval' steps.
    """
    model.train()
    total_loss_sum = 0.0
    
    num_batches = len(dataloader)
    
    for i, data in enumerate(dataloader):
        # Calculate global step for continuous plotting across epochs
        global_step = (epoch_index - 1) * num_batches + (i + 1)
        
        data = data.to(device)
        
        # 1. Forward pass
        recon_batch, mu, logvar = model(data)
        
        # 2. Calculate Loss
        BCE = torch.nn.functional.mse_loss(recon_batch, data, reduction='sum')
        # KL Divergence
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        # Total Loss (ELBO)
        loss = BCE + (beta * KLD)
        
        # 3. Backward pass & Optimization
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient Clipping (prevents exploding gradients)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss_sum += loss.item()

        if (i + 1) % log_interval == 0:
            history['steps'].append(global_step)
            history['total_loss'].append(loss.item())
            history['recon_loss'].append(BCE.item())
            history['kld_loss'].append(KLD.item())
            
            print(f"   [Epoch {epoch_index}] Step [{i+1}/{num_batches}] "
                  f"| Loss: {loss.item():.1f} "
                  f"(Recon: {BCE.item():.1f}, KLD: {KLD.item():.1f})")
    
    return total_loss_sum / len(dataloader.dataset)


def validate(model, dataloader, device, beta):
    """
    Evaluates the model on the validation dataset.
    """
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for data in dataloader:
            data = data.to(device)
            recon, mu, logvar = model(data)
            loss = loss_function(recon, data, mu, logvar, beta=beta)
            val_loss += loss.item()
            
    return val_loss / len(dataloader.dataset)

def main():
    # 1. Load Configuration
    print("Loading configuration...")
    if not os.path.exists('config/config.json'):
        raise FileNotFoundError("config/config.json not found!")
        
    with open('config/config.json', 'r') as f:
        config = json.load(f)

    # Create directories for artifacts
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('results', exist_ok=True)

    device = torch.device(config['device'] if torch.cuda.is_available() else "cpu")
    print(f"   Running on device: {device}")
    print(f"   Batch Size: {config['batch_size']}")
    print(f"   Learning Rate: {config['learning_rate']}")
    print(f"   Beta: {config['beta']}")

    history = {
        'steps': [],
        'total_loss': [],
        'recon_loss': [],
        'kld_loss': []
    }

    # 2. Load Data
    print("\n   Loading DataLoaders...")
    train_loader, val_loader, _ = get_dataloaders(config)
    print(f"   Train images: {len(train_loader.dataset)}")
    print(f"   Val images:   {len(val_loader.dataset)}")

    # 3. Initialize Model
    print("\n   Initializing VAE Model...")
    model = VAE(latent_dim=config['latent_dim']).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])

    # 4. Start Training Loop
    print(f"\n   STARTING TRAINING for {config['epochs']} epochs...")
    print("=" * 60)
    
    start_time = time.time()
    best_loss = float('inf')

    for epoch in range(1, config['epochs'] + 1):
        epoch_start = time.time()
        
        t_loss = train_one_epoch(
            model, 
            train_loader, 
            optimizer, 
            device, 
            beta=config['beta'],
            epoch_index=epoch,
            history=history,       
            log_interval=100
        )
        
        print("    Validating...")
        val_loss = validate(model, val_loader, device, beta=config['beta'])
        
        is_best = val_loss < best_loss
        if is_best:
            best_loss = val_loss
            torch.save(model.state_dict(), f"checkpoints/best_vae.pth")

        save_live_plot(history)
        save_comparison_images(model, val_loader, epoch, device)

        duration = time.time() - epoch_start
        print("-" * 60)
        print(f"   EPOCH {epoch}/{config['epochs']} COMPLETE ({duration:.1f}s)")
        print(f"   Avg Train Loss: {t_loss:.4f} | Val Loss: {val_loss:.4f}")
        if is_best:
            print("    New Best Model Saved!")
        print("-" * 60)

    total_time = (time.time() - start_time) / 60
    print(f"\n Training finished in {total_time:.1f} minutes.")
    
    torch.save(model.state_dict(), f"checkpoints/final_vae.pth")

if __name__ == "__main__":
    main()