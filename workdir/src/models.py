from dataclasses import dataclass
from typing import Literal

import torch
from schnetpack.representation import PaiNN
from schnetpack.nn.cutoff import CosineCutoff
from schnetpack.nn.radial import GaussianRBF
from torch_geometric.nn import global_mean_pool
import torch.nn as nn
from torch_scatter import scatter

from src.dataset import pyg_batch_to_schnetpack
from src.models_ps.painn import PaiNN_WL
from src.models_ps.schnet import SchNet
from src.models_ps.mdnet import TorchMD_ET
from src.models_ps.matformer import MatformerConfig, MatformerConv, RBFExpansion
from config import n_out


class FoldsEmb(nn.Module):
    def __init__(self, emb_dim, hidden, n_out, drop=0.3):
        super().__init__()
        self.emb = nn.Embedding(7, emb_dim)
        self.head = nn.Sequential(
            nn.Linear(emb_dim, hidden*2),
            nn.SiLU(),
            nn.LayerNorm(hidden*2),
            nn.Dropout(drop),
            nn.Linear(hidden*2, n_out)
        )

    def forward(self, x):
        x = self.emb(x).sum(-2)
        return self.head(x)

class WaveLenEmb(nn.Module):
    def __init__(self, emb_dim, hidden, n_out, drop=0.3):
        super().__init__()
        self.register_buffer("wl_embs",torch.Tensor([5.14, 5.32, 7.8, 7.85]))
        self.emb = nn.Sequential(
            nn.Embedding(4, emb_dim),
            nn.Linear(emb_dim, hidden*2),
            nn.SiLU(),
            nn.LayerNorm(hidden*2),
            nn.Dropout(drop),
            nn.Linear(hidden*2, n_out)
        )

    def forward(self, x):
        idx = (x == self.wl_embs).long().argmax(dim=-1)
        return self.emb(idx)

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

    def forward(self, data):
        # data = dat.clone()
        # data.x = drop(data.x,7)
        batch = pyg_batch_to_schnetpack(data, cutoff=10.0)

        painn_output = self.painn(batch)

        atom_feats = painn_output["scalar_representation"]

        # pooled_feats = torch_scatter.scatter_mean(atom_feats, data.batch, dim=0)
        pooled_feats = global_mean_pool(atom_feats, data.batch)

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

    def forward(self, data):
        # data = dat.clone()
        # data.x = drop(data.x,7)
        batch = pyg_batch_to_schnetpack(data, cutoff=10.0)
        painn_output = self.painn(batch)
        atom_feats = painn_output["scalar_representation"]
        # pooled = torch_scatter.scatter_mean(atom_feats, data.batch, dim=0)
        pooled = global_mean_pool(atom_feats, data.batch)

        if not hasattr(data,"wl"):
            wl = torch.tensor([5.14] * data.num_graphs, device=data.x.device)
        else:
            wl = data.wl.unsqueeze(1)
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

    def forward(self, data):
        # data = dat.clone()
        # data.x = drop(data.x,7)
        batch = pyg_batch_to_schnetpack(data, cutoff=8.0)
        painn_output = self.painn(batch)
        atom_feats = painn_output["scalar_representation"]
        # pooled = torch_scatter.scatter_mean(atom_feats, data.batch, dim=0)

        if not hasattr(data,"wl"):
            wl = torch.tensor([5.14] * data.num_graphs, device=data.x.device)
        else:
            wl = data.wl.view(-1,1)
        
        pooled = global_mean_pool(atom_feats, batch)
        wl_emb = self.wl_emb(wl)
        folds_emb = self.folds_emb(data.folds)
        combined = torch.cat([pooled, wl_emb, folds_emb], dim=-1)
        
        raw_tensors = self.head(combined)
        
        batch_size = raw_tensors.shape[0]

        cond_vec = data.cond_vec if hasattr(data, "cond_vec") else torch.zeros(batch_size, 7, device=raw_tensors.device)
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


class MDNet(nn.Module):
    def __init__(self, args=None):
        super().__init__()
        self.args = args
        if args is not None:
            for k, v in args.items():
                base_args[k] = v
        # temp = create_model(base_args)

        # self.body = temp.representation_model
        args = base_args
        self.body = TorchMD_ET(
            hidden_channels=args["embedding_dimension"],
            num_layers=args["num_layers"],
            num_rbf=args["num_rbf"],
            rbf_type=args["rbf_type"],
            trainable_rbf=args["trainable_rbf"],
            activation=args["activation"],
            attn_activation=args["attn_activation"],
            neighbor_embedding=args["neighbor_embedding"],
            num_heads=args["num_heads"],
            distance_influence=args["distance_influence"],
            cutoff_lower=args["cutoff_lower"],
            cutoff_upper=args["cutoff_upper"],
            max_z=args["max_z"],
            max_num_neighbors=args["max_num_neighbors"]
        )

        self.wl_emb = nn.Sequential(
            nn.Linear(1,64, bias=False),
            nn.ReLU(),
            nn.LayerNorm(64),
            nn.Linear(64,64)
        )

        self.head = nn.Sequential(
            # nn.LayerNorm(args["embedding_dimension"] + 64),
            nn.Dropout(args["dropout"]),
            nn.Linear(args["embedding_dimension"]+64, args["hidden_size"]),
            nn.LayerNorm(args["hidden_size"]),
            nn.ReLU(),
            nn.Dropout(args["dropout"]),
            nn.Linear(args["hidden_size"], n_out)
        )

    def forward(self,x):
        z = x.z
        pos = x.pos #+ torch.randn_like(x.pos) * 1e-5
        batch = x.batch
        wl = x.wl.view(-1,1)
        
        atom_feats, *_ = self.body(z, pos, batch)
        pooled_feats = global_mean_pool(atom_feats, batch)

        wl_out = self.wl_emb(wl)
        combined_feats = torch.cat([pooled_feats,wl_out],dim=1)
        
        out = self.head(combined_feats)
        return out
    
    def __str__(self):
        return "MDNet"
    
    def n_params(self):
        return sum([p.numel() for p in self.parameters()])
    
    def get_args(self):
        return self.args
    
class MDNet1(nn.Module):
    def __init__(self, args={}):
        super().__init__()
        self.args = args
        for k, v in args.items():
            base_args[k] = v
        
        args = base_args
        self.body = TorchMD_ET(
            hidden_channels=args["embedding_dimension"],
            num_layers=args["num_layers"],
            num_rbf=args["num_rbf"],
            rbf_type=args["rbf_type"],
            trainable_rbf=args["trainable_rbf"],
            activation=args["activation"],
            attn_activation=args["attn_activation"],
            neighbor_embedding=args["neighbor_embedding"],
            num_heads=args["num_heads"],
            distance_influence=args["distance_influence"],
            cutoff_lower=args["cutoff_lower"],
            cutoff_upper=args["cutoff_upper"],
            max_z=args["max_z"],
            max_num_neighbors=args["max_num_neighbors"]
        )

        self.wl_emb = WaveLenEmb(args["embedding_dimension"],args["hidden_size"],64,args["dropout"])
        self.folds_emb = FoldsEmb(args["embedding_dimension"],args["hidden_size"],64,args["dropout"])
        self.orientation_head = nn.Sequential(
            nn.Linear(6, args["hidden_size"]),
            nn.SiLU(),
            nn.LayerNorm(args["hidden_size"]),
            nn.Linear(args["hidden_size"], 6)
        )

        self.head = nn.Sequential(
            nn.Dropout(args["dropout"]),
            nn.Linear(args["embedding_dimension"]+128, args["hidden_size"]),
            nn.LayerNorm(args["hidden_size"]),
            nn.SiLU(),
            nn.Dropout(args["dropout"]),
            nn.Linear(args["hidden_size"], n_out * 6)
        )

    def forward(self, x):
        z = x.z
        pos = x.pos 
        batch = x.batch
        wl = x.wl.view(-1,1)
        folds = x.folds
        
        atom_feats, *_ = self.body(z, pos, batch)
        pooled = global_mean_pool(atom_feats, batch)
        wl_emb = self.wl_emb(wl)
        folds_emb = self.folds_emb(folds)
        combined = torch.cat([pooled, wl_emb, folds_emb], dim=-1)
        
        raw_tensors = self.head(combined)
        batch_size = raw_tensors.shape[0]
        raw_tensors = raw_tensors.view(batch_size, n_out, 6)

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
    
    def __str__(self):
        return "MDNet1"
    
    def n_params(self):
        return sum([p.numel() for p in self.parameters()])
    
    def get_args(self):
        return self.args

class MatFormer(nn.Module):
    def __init__(self, args):
        super().__init__()
        
        config = MatformerConfig()
        for k, v in args.items():
            if hasattr(config, k):
                setattr(config, k, v)
        self.args = {k:v for k,v in args.items() if k in {"hidden_size", "num_heads", "num_layers"}}
        self.atom_embedding = nn.Linear(8, config.hidden_size)
        
        self.rbf = nn.Sequential(
            RBFExpansion(vmin=0, vmax=5.0, bins=config.hidden_size),
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.Softplus(),
            nn.Linear(config.hidden_size, config.hidden_size),
        )

        self.att_layers = nn.ModuleList([
            MatformerConv(in_channels=config.hidden_size, out_channels=config.hidden_size, heads=config.num_heads, edge_dim=config.hidden_size)
            for _ in range(config.num_layers)
        ])

        self.fc = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size), 
            nn.SiLU()
        )
        
        self.wl_emb = nn.Sequential(
            nn.Linear(1, 64),
            nn.LayerNorm(64),
            nn.SiLU(),
            nn.Linear(64, 64)
        )
        
        self.fc_out = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size + 64, config.hidden_size),
            nn.LayerNorm(config.hidden_size),
            nn.SiLU(),
            nn.Linear(config.hidden_size, n_out)
        )

    def forward(self, data):
        node_features = self.atom_embedding(data.x)
        
        edge_features = self.rbf(data.dist)
        
        for layer in self.att_layers:
            node_features = layer(node_features, data.edge_index, edge_features)

        features = scatter(node_features, data.batch, dim=0, reduce="mean")
        features = self.fc(features)
        
        wl_out = self.wl_emb(data.wl.view(-1, 1))
        combined = torch.cat([features, wl_out], dim=1)

        out = self.fc_out(combined)
        return out
    
    def __str__(self):
        return "matformer"
    
    def n_params(self):
        return sum([p.numel() for p in self.parameters()])
    
    def get_args(self):
        return self.args


models_dict = {
    "MDNet": MDNet,
    "MDNet1": MDNet1,
    "painn": PaiNN_raman0,
    "painn2": PaiNN_raman2,
    "painn3": PaiNN_raman3,
    "matformer": MatFormer
}
