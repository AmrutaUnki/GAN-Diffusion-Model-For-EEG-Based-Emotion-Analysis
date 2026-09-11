import torch
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
NUM_CLASSES = 8
IMG_CHANNELS = 3          # RGB spectrograms (Class_Wise_RGB_Spectrogram_DEAP)
IMG_SIZE = 64             # spectrogram images are resized to IMG_SIZE x IMG_SIZE
FEATURE_DIM = 256         # dimensionality of GAN-extracted / diffusion-refined feature vectors

# Root directory containing one sub-folder per class (8 folders expected).
# Update this path if you move the data. Class names are auto-detected from
# the sub-folder names found here (sorted alphabetically) -- see dataset.py.
DATA_ROOT = r"D:\GAN_Diffusion_CNN_Paper_Material\Class_Wise_RGB_Spectrogram_DEAP"

# ---------------------------------------------------------------------------
# GAN Model
# ---------------------------------------------------------------------------
GAN_ACTIVATION = "leaky_relu"      # LeakyReLU
GAN_LEAKY_SLOPE = 0.2
GAN_OPTIMIZER = "adam"
GAN_LR = 0.0002
GAN_LOSS = "bce"                   # Binary Cross Entropy
GAN_EPOCHS = 30
GAN_BATCH_SIZE = 16
GAN_LATENT_DIM = 100                # noise vector size for the generator

# ---------------------------------------------------------------------------
# Diffusion Model 
# ---------------------------------------------------------------------------
DIFF_ACTIVATION = "relu"           # ReLU
DIFF_OPTIMIZER = "adam"
DIFF_LR = 0.001
DIFF_LOSS = "mse"                  # Mean Squared Error
DIFF_EPOCHS = 30
DIFF_BATCH_SIZE = 16
DIFF_TIMESTEPS = 200                # number of forward-diffusion steps (T)
DIFF_BETA_START = 1e-4
DIFF_BETA_END = 0.02
DIFF_REFINE_STEPS = 20              # partial noise/denoise steps used at inference-time refinement

# ---------------------------------------------------------------------------
# CNN Classifier
# ---------------------------------------------------------------------------
CNN_ACTIVATION = "relu"            # ReLU (hidden layers)
CNN_ACTIVATION_OUT = "softmax"     # Softmax (output layer) -- see note above
CNN_OPTIMIZER = "adam"             # corrected from "SoftMax" (see module docstring)
CNN_LR = 0.001
CNN_LOSS = "categorical_crossentropy"
CNN_EPOCHS = 30
CNN_BATCH_SIZE = 16
