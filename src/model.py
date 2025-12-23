import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self, latent_dim=128):
        """
        Convolutional Variational Autoencoder (VAE) architecture.
        Input: (3, 64, 64) image.
        """
        super(VAE, self).__init__()
        self.latent_dim = latent_dim

        # ENCODER
        # Compresses the image: 64x64 -> 128 latent vector
        self.encoder = nn.Sequential(
            # Layer 1: 64x64 -> 32x32
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            
            # Layer 2: 32x32 -> 16x16
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            
            # Layer 3: 16x16 -> 8x8
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            
            # Layer 4: 8x8 -> 4x4
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2)
        )
        
        # Flatten size: 256 channels * 4 height * 4 width = 4096
        self.flatten_size = 256 * 4 * 4
        
        # Latent space vectors (Mean and Log-Variance)
        self.fc_mu = nn.Linear(self.flatten_size, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_size, latent_dim)
        
        # DECODER
        # Reconstructs the image: 128 latent vector -> 64x64
        self.decoder_input = nn.Linear(latent_dim, self.flatten_size)
        
        self.decoder = nn.Sequential(
            # Reshape happens in forward pass, here we start transposed convolutions
            # Layer 1: 4x4 -> 8x8
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            
            # Layer 2: 8x8 -> 16x16
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            
            # Layer 3: 16x16 -> 32x32
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            
            # Layer 4: 32x32 -> 64x64
            # Output: 3 channels (RGB), Sigmoid for range [0, 1]
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid() 
        )

    def reparameterize(self, mu, logvar):
        """
        Implements the Reparameterization Trick: z = mu + sigma * epsilon
        """
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu

    # --- ДОДАНІ МЕТОДИ (Encode / Decode) ---
    
    def encode(self, x):
        """Helper function to encode an image to latent variables"""
        x = self.encoder(x)
        x = x.view(x.size(0), -1) # Flatten
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    def decode(self, z):
        """Helper function to decode a latent vector to an image"""
        z = self.decoder_input(z)
        z = z.view(z.size(0), 256, 4, 4) # Unflatten
        reconstruction = self.decoder(z)
        return reconstruction

    def forward(self, x):
        # Тепер forward просто використовує наші нові методи
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decode(z)
        return reconstruction, mu, logvar

# LOSS FUNCTION (ELBO)
def loss_function(recon_x, x, mu, logvar, beta=1.0):
    """
    Computes the VAE Loss (Evidence Lower Bound).
    """
    # 1. Reconstruction Loss (MSE)
    BCE = F.mse_loss(recon_x, x, reduction='sum')

    # 2. KL Divergence
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    # Total Loss
    return BCE + (beta * KLD)