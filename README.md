# PyTorch Variational Autoencoder (VAE)

This repository contains a modular and highly configurable implementation of a **Variational Autoencoder (VAE)** built with PyTorch. It is designed to provide clear insights into the training process through automated visual logging.

## Key Requirements Met

* **Modular Design**: Clear separation between model architecture (`src/model.py`), data pipeline (`src/dataset.py`), and training execution (`train.py`).
* **Latent Space Control**: Latent dimension and Beta-VAE regularization ($\beta$) are fully adjustable via the JSON configuration file.
* **Visualization**: 
    * **Metric Logging**: Automatically generates and updates `results/training_log.png` after every epoch, showing Total Loss, Reconstruction Error (MSE), and KL Divergence.
    * **Reconstruction Grids**: Saves side-by-side comparison grids (`Original vs. Reconstructed`) to `results/` for qualitative assessment.
* **Training Stability**: Implements gradient clipping and tracks the best model based on validation loss.

##  Project Structure

```text
.
├── config/
│   └── config.json       # Hyperparameters (learning rate, latent_dim, beta)
├── src/
│   ├── dataset.py        # Data loading and preprocessing pipeline
│   └── model.py          # VAE architecture and Loss function logic
├── results/              # Auto-generated: Training plots and reconstruction grids
├── checkpoints/          # Auto-generated: Best and final model weights (.pth)
├── train.py              # Main training execution script
├── requirements.txt      # List of dependencies
└── .gitignore            # Rules to exclude artifacts from version control
```
#  Setup and Installation
## 1. Clone the Repository
```bash
git clone [https://github.com/vibabi/vae-project.git](https://github.com/vibabi/vae-project.git)
cd vae-project
```

## 2. Configure Virtual Environment (venv)
Using an isolated environment ensures that dependency versions do not conflict with your global Python setup.

For Windows:
```bash
python -m venv venv
.\venv\Scripts\activate
```

For macOS / Linux:
```bash
python3 -m venv venv
source venv/bin/activate
```

## 3. Install Dependencies
```bash
pip install -r requirements.txt
```

#  Usage
## Training the Model
To initiate the training process:
```bash
python train.py
```

While the model is training, you can monitor the following in the results/ directory:

training_log.png: A 3-panel plot visualizing the convergence of ELBO, Reconstruction Loss, and KLD.

reconstruction_epoch_N.png: Image grids where the top rows are original inputs and the bottom rows are the VAE's reconstructions.

#  Configuration
The model behavior can be modified in config/config.json:

latent_dim: Size of the compressed bottleneck layer.

beta: Weighting factor for the KL Divergence term (useful for Disentangled VAEs).

batch_size & learning_rate: Standard optimization hyperparameters.