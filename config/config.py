from pathlib import Path

import numpy as np
from torch.optim.lr_scheduler import MultiStepLR, ExponentialLR, CosineAnnealingLR, StepLR

from src.loss import MSELoss, L1Loss, MSLELoss, RMSELoss, CosSimLoss, IoULoss, KLDivLoss


cur_dir = Path(__file__).resolve().parent.parent / "results"
cur_dir.mkdir(exist_ok=True, parents=True)
tblogdir = cur_dir / "tblogs"
tblogdir.mkdir(parents=True, exist_ok=True)
logdir = cur_dir / "logs"
logdir.mkdir(parents=True, exist_ok=True)
models_dir = cur_dir / "models"
models_dir.mkdir(parents=True, exist_ok=True)

data_path = Path('data/processed')

loss_fn_dict = {
    "mse": MSELoss(), 
    "mae":L1Loss(), 
    "msle": MSLELoss(), 
    "rmse": RMSELoss(), 
    "cossim": CosSimLoss(), 
    "iou": IoULoss(), 
    "kldiv": KLDivLoss()
    }

# decay_dict = {
#     "milestone": MultiStepLR,
#     "exp": ExponentialLR,
#     "cos": CosineAnnealingLR,
#     "step": StepLR
# }
try:
    wavenumbers = np.load('data/processed/wavenumber_vals_v3.npy')
except FileNotFoundError:
    wavenumbers = np.load('../data/processed/wavenumber_vals_v3.npy')
n_out = len(wavenumbers)

base_args = {
    "model": "equivariant-transformer",
    "num_rbf": 50,
    "rbf_type": "gauss",
    "trainable_rbf": True,
    "activation": "silu",
    "cutoff_lower": 0.0,
    "cutoff_upper": 10.0,
    "max_z": 100,
    "max_num_neighbors": 128,
    "aggr": "add",
    "derivative": False,
    "atom_filter": -1,
    "prior_model": None,
    "output_model": "Scalar",
    "reduce_op": "add",
    "precision": 32,
    "check_errors": False,
    "distance_influence": "keys",
    "vector_cutoff": False,
    "hidden_size": 128,
    "dropout": 0.3,

    "num_layers": 4,
    "embedding_dimension": 64,
    "attn_activation": "silu",
    "num_heads": 8,
    "neighbor_embedding": True,
}
