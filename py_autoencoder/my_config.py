from pathlib import Path
# Define constants for base directories and files
BASE_DIR = Path.home() / "Dropbox"
CODE_DIR = BASE_DIR / "Code"
AE_MC_DIR = BASE_DIR / "AE_MC"
AE_2023_06_DIR = CODE_DIR / "AE_2023_06" / "TrainedModels"

# Configuration settings
USE_GPU = True
LOCAL_SAVE = BASE_DIR / "TopResultlsOut"
RUN_LOCAL = True
IMAGE_FOLDER = None
ENCODER_PATH = None
DECODER_PATH = None
# CHECKPOINT_PATH = r"C:\Users\joeli\OneDrive\Documents\GitHub\AutoEncoder_Integrated\autoencoder_best.h5py"
# LOGDIR=r"C:\Users\joeli\OneDrive\Documents\GitHub\AutoEncoder_Integrated\tensorboard_log_dir"
CHECKPOINT_PATH = LOCAL_SAVE/"checkpoints"/"autoencoder_best.h5py"
LOGDIR = LOCAL_SAVE/"tensorboard_log_dir"
# Paths specific to local running
if RUN_LOCAL:
    IMAGE_FOLDER = AE_MC_DIR / "AE_InputModels" / "test"
    model_dir = AE_2023_06_DIR / "no_duplicates_75_2_mask"
    ENCODER_PATH = model_dir / "encoder.h5"
    DECODER_PATH = model_dir / "decoder.h5"
else:
    # Define remote or other non-local paths
    IMAGE_FOLDER = "models_4k"
    remote_dir = Path("/AE_2023_06/TrainedModels/no_duplicates_75_2_mask")
    ENCODER_PATH = remote_dir / "encoder.h5"
    DECODER_PATH = remote_dir / "decoder.h5"
    
CMAP_SPECULAR = 'viridis'

