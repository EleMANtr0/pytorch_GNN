import torch
import torch.nn as nn
from config import n_out
from schnetpack.nn.cutoff import CosineCutoff
from schnetpack.nn.radial import GaussianRBF
from schnetpack.representation import PaiNN
from torch_geometric.nn import global_mean_pool

from src.data.dataset import pyg_batch_to_schnetpack
from .embeddings import WaveLenEmb, FoldsEmb


class PaiNN_raman0(nn.Module):
    def __init__(self):
        super().__init__()
        self.painn = PaiNN(cutoff_fn=CosineCutoff(cutoff=10),n_interactions=3,n_atom_basis=256,radial_basis=GaussianRBF(n_rbf=30,cutoff=10))
        self.head = nn.Sequential(
            nn.SiLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.SiLU(),
            nn.Dropout(0.3),
            nn.Linear(128, n_out)
        )

    def forward(self, x, *args, **kwargs):
        # x = dat.clone()
        # x.x = drop(x.x,7)
        batch = pyg_batch_to_schnetpack(x, cutoff=10.0)

        painn_output = self.painn(batch)

        atom_feats = painn_output["scalar_representation"]

        # pooled_feats = torch_scatter.scatter_mean(atom_feats, x.batch, dim=0)
        pooled_feats = global_mean_pool(atom_feats, x.batch)

        output = self.head(pooled_feats)

        return output

    def __str__(self):
        return "painn"
    
    def n_params(self):
        return sum([p.numel() for p in self.parameters()])

class PaiNN_raman2(nn.Module):
    def __init__(self):
        super().__init__()
        self.painn = PaiNN(cutoff_fn=CosineCutoff(cutoff=10),n_interactions=3,n_atom_basis=256,radial_basis=GaussianRBF(n_rbf=30,cutoff=10))
        self.head = nn.Sequential(
            nn.SiLU(),
            nn.Dropout(0.3),
            nn.Linear(320, 128),
            nn.SiLU(),
            nn.Dropout(0.3),
            nn.Linear(128, n_out)
        )
        self.wl_embed = WaveLenEmb(32,64,0.3)

    def forward(self, x, *args, **kwargs):
        # x = dat.clone()
        # x.x = drop(x.x,7)
        batch = pyg_batch_to_schnetpack(x, cutoff=10.0)
        painn_output = self.painn(batch)
        atom_feats = painn_output["scalar_representation"]
        # pooled = torch_scatter.scatter_mean(atom_feats, x.batch, dim=0)
        pooled = global_mean_pool(atom_feats, x.batch)

        if not hasattr(x,"wl"):
            wl = torch.tensor([5.14] * x.num_graphs, device=x.x.device)
        else:
            wl = x.wl.unsqueeze(1)
        wl_emb = self.wl_embed(wl)
        fused = torch.cat([pooled, wl_emb], dim=-1)
        out = self.head(fused)

        return out

    def __str__(self):
        return "painn2"
    
    def n_params(self):
        return sum([p.numel() for p in self.parameters()])


class PaiNN_raman3(nn.Module):
    def __init__(self,args={
        "dropout":0.3, "embedding_dimension":256, "hidden_size": 128
    }):
        super().__init__()
        self.args = args
        self.painn = PaiNN(cutoff_fn=CosineCutoff(cutoff=10),n_interactions=3,n_atom_basis=args["embedding_dimension"],radial_basis=GaussianRBF(n_rbf=30,cutoff=10))

        self.wl_emb = WaveLenEmb(args["embedding_dimension"],args["hidden_size"],64,args["dropout"])
        self.folds_emb = FoldsEmb(args["embedding_dimension"],args["hidden_size"],64,args["dropout"])
        self.orientation_head = nn.Sequential(
            nn.Linear(6, 64),
            nn.SiLU(),
            nn.Linear(64, 6)
        )

        self.head = nn.Sequential(
            nn.SiLU(),
            nn.LayerNorm(args["embedding_dimension"] + 128),
            nn.Dropout(args["dropout"]),
            nn.Linear(args["embedding_dimension"]+128, args["hidden_size"]),
            nn.LayerNorm(args["hidden_size"]),
            nn.SiLU(),
            nn.Dropout(args["dropout"]),
            nn.Linear(args["hidden_size"], n_out*6)
        )

    def forward(self, x, *args, **kwargs):
        # x = dat.clone()
        # x.x = drop(x.x,7)
        batch = pyg_batch_to_schnetpack(x, cutoff=8.0)
        painn_output = self.painn(batch)
        atom_feats = painn_output["scalar_representation"]
        # pooled = torch_scatter.scatter_mean(atom_feats, x.batch, dim=0)

        if not hasattr(x,"wl"):
            wl = torch.tensor([5.14] * x.num_graphs, device=x.x.device)
        else:
            wl = x.wl.view(-1,1)
        
        pooled = global_mean_pool(atom_feats, batch)
        wl_emb = self.wl_emb(wl)
        folds_emb = self.folds_emb(x.folds)
        combined = torch.cat([pooled, wl_emb, folds_emb], dim=-1)
        
        raw_tensors = self.head(combined)
        
        batch_size = raw_tensors.shape[0]

        cond_vec = x.cond_vec if hasattr(x, "cond_vec") else torch.zeros(batch_size, 7, device=raw_tensors.device)
        mask = cond_vec[:, 0].view(batch_size, 1).to(torch.float32)
        orientation_vec = cond_vec[:, 1:7].to(torch.float32)
        
        proj_weights = self.orientation_head(orientation_vec).unsqueeze(1)
        polarized_output = (raw_tensors * proj_weights).sum(dim=-1) ** 2

        R_xx = raw_tensors[:, :, 0]
        R_yy = raw_tensors[:, :, 1]
        R_zz = raw_tensors[:, :, 2]
        R_xy = raw_tensors[:, :, 3]
        R_yz = raw_tensors[:, :, 4]
        R_xz = raw_tensors[:, :, 5]

        a = (R_xx + R_yy + R_zz) / 3.0
        gamma_sq = 0.5 * ((R_xx - R_yy)**2 + (R_yy - R_zz)**2 + (R_zz - R_xx)**2 + 6.0 * (R_xy**2 + R_yz**2 + R_xz**2))
        
        unpolarized_output = 45.0 * (a**2) + 7.0 * gamma_sq

        output = polarized_output * mask + unpolarized_output * (1 - mask)
        return output

    def get_args(self):
        return self.args
    
    def __str__(self):
        return "painn3"
    
    def n_params(self):
        return sum([p.numel() for p in self.parameters()])
