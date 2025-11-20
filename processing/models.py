from torch.nn import Linear, ReLU, Dropout, Module, SiLU, Sequential
import torch
from schnetpack.representation import PaiNN
from schnetpack.nn.cutoff import CosineCutoff
from schnetpack.nn.radial import GaussianRBF
from src.data.dataset import pyg_batch_to_schnetpack
from src.models.painn import PaiNN_WL
import numpy as np
import torch_geometric.nn
from src.models.schnet import SchNet
import torch_scatter

model_wavenumbers = np.load('../data/processed/wavenumber_vals_v3.npy')

#%%
class PaiNN_raman0(Module):
    def __init__(self):
        super().__init__()
        self.painn = PaiNN(cutoff_fn=CosineCutoff(cutoff=10),n_interactions=3,n_atom_basis=256,radial_basis=GaussianRBF(n_rbf=30,cutoff=10))
        self.head = Sequential(
            SiLU(),
            Dropout(0.3),
            Linear(256, 128),
            SiLU(),
            Dropout(0.3),
            Linear(128, len(model_wavenumbers))
        )

    def forward(self, data):
        # data = dat.clone()
        # data.x = drop(data.x,7)
        batch = pyg_batch_to_schnetpack(data, cutoff=10.0)

        painn_output = self.painn(batch)

        atom_feats = painn_output["scalar_representation"]

        pooled_feats = torch_scatter.scatter_mean(atom_feats, data.batch, dim=0)

        output = self.head(pooled_feats)

        return output

    def __repr__(self):
        return "painn"

class PaiNN_raman1(Module):
    def __init__(self):
        super().__init__()
        self.painn = PaiNN(cutoff_fn=CosineCutoff(cutoff=10),n_interactions=4,n_atom_basis=384,radial_basis=GaussianRBF(n_rbf=30,cutoff=10))
        self.head = Sequential(
            SiLU(),
            Dropout(0.4),
            Linear(384, 128),
            SiLU(),
            Dropout(0.4),
            Linear(128, len(model_wavenumbers))
        )

    def forward(self, data):
        batch = pyg_batch_to_schnetpack(data, cutoff=10.0)

        painn_output = self.painn(batch)

        scalars = painn_output["scalar_representation"]

        pooled_feats = torch_scatter.scatter_mean(scalars, data.batch, dim=0)

        output = self.head(pooled_feats)

        return output

    def __repr__(self):
        return "painn1"

class PaiNN_raman2(Module):
    def __init__(self):
        super().__init__()
        self.painn = PaiNN(cutoff_fn=CosineCutoff(cutoff=10),n_interactions=3,n_atom_basis=256,radial_basis=GaussianRBF(n_rbf=30,cutoff=10))
        self.head = Sequential(
            SiLU(),
            Dropout(0.3),
            Linear(320, 128),
            SiLU(),
            Dropout(0.3),
            Linear(128, len(model_wavenumbers))
        )
        self.wl_embed = Sequential(
            Linear(1, 32),
            SiLU(),
            Dropout(0.3),
            Linear(32, 64)
        )

    def forward(self, data):
        # data = dat.clone()
        # data.x = drop(data.x,7)
        batch = pyg_batch_to_schnetpack(data, cutoff=10.0)
        painn_output = self.painn(batch)
        atom_feats = painn_output["scalar_representation"]
        pooled = torch_scatter.scatter_mean(atom_feats, data.batch, dim=0)
        if not hasattr(data,"wl"):
            wl = torch.tensor([5.14] * data.num_graphs, device=data.x.device)
        else:
            wl = data.wl.unsqueeze(1)
        wl_emb = self.wl_embed(wl)
        fused = torch.cat([pooled, wl_emb], dim=-1)
        out = self.head(fused)

        return out

    def __repr__(self):
        return "painn2"


class PaiNN_raman3(Module):
    def __init__(self):
        super().__init__()
        self.painn = PaiNN_WL(
            cutoff_fn=CosineCutoff(cutoff=10),
            n_interactions=3,
            n_atom_basis=256,
            radial_basis=GaussianRBF(n_rbf=30, cutoff=10),
            wl_embed_dim=64
        )
        self.head = Sequential(
            SiLU(),
            Dropout(0.3),
            Linear(256, 128),
            SiLU(),
            Dropout(0.3),
            Linear(128, len(model_wavenumbers))
        )

    def forward(self, data):
        batch = pyg_batch_to_schnetpack(data, cutoff=10.0)

        if hasattr(data, "wl"):
            batch['wl'] = data.wl
        else:
            batch['wl'] = torch.tensor([5.14] * data.num_graphs, device=data.x.device)

        batch['batch'] = data.batch

        painn_output = self.painn(batch)
        atom_feats = painn_output["scalar_representation"]
        pooled = torch_scatter.scatter_mean(atom_feats, data.batch, dim=0)
        out = self.head(pooled)

        return out

    def __repr__(self):
        return "painn3"

class Ensemble1(Module):
    def __init__(self,path1,path2):
        super().__init__()
        self.model1 = PaiNN_raman0()
        self.model1.load_state_dict(torch.load(path1))
        self.model2 = PaiNN_raman2()
        self.model2.load_state_dict(torch.load(path2))

    def forward(self,data):
        pred1 = self.model1(data)
        pred2 = self.model2(data)
        return (pred1 + pred2) / 2

    def __repr__(self):
        return "painn_ensemble1"

class Ensemble2(Module):
    def __init__(self,path1,path2,path3):
        super().__init__()
        self.model1 = PaiNN_raman0()
        self.model1.load_state_dict(torch.load(path1))
        self.model2 = PaiNN_raman0()
        self.model2.load_state_dict(torch.load(path2))
        self.model3 = PaiNN_raman0()
        self.model3.load_state_dict(torch.load(path3))

    def forward(self,data):
        pred1 = self.model1(data)
        pred2 = self.model2(data)
        pred3 = self.model3(data)
        return 0.5*pred1 + 0.1 * pred2 + 0.5*pred3

    def __repr__(self):
        return "painn_ensemble2"

Schnet = torch_geometric.nn.Sequential(
    'z, pos, batch',
    [
        (SchNet(hidden_channels=1024,out_features=256),'z, pos, batch->z'),
        ReLU(inplace=True),
        Dropout(0.5),
        Linear(256, 128),
        ReLU(inplace=True),
        Dropout(0.5),
        Linear(128, len(model_wavenumbers)),
    ]
)
