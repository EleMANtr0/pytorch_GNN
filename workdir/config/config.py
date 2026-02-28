from pathlib import Path

import numpy as np
from torch.optim.lr_scheduler import MultiStepLR, ExponentialLR, CosineAnnealingLR, StepLR

from src.utils.loss import MSELoss, L1Loss, MSLELoss, RMSELoss, CosSimLoss, IoULoss, KLDivLoss


res_dir = Path("results")
models_dir = res_dir / "models"
data_path = Path('../data/processed')
models_dir.mkdir(parents=True, exist_ok=True)

loss_fn_dict = {
    "mse": MSELoss, 
    "mae":L1Loss, 
    "msle": MSLELoss, 
    "rmse": RMSELoss, 
    "cossim": CosSimLoss, 
    "iou": IoULoss, 
    "kldiv": KLDivLoss
    }

decay_dict = {
    "milestone": MultiStepLR,
    "exp": ExponentialLR,
    "cos": CosineAnnealingLR,
    "step": StepLR
}

wavenumbers = np.load('../data//processed/wavenumber_vals_v3.npy')
n_out = len(wavenumbers)
