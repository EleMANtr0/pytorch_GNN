import torch
import torch.nn as nn
from torch_geometric.nn import global_mean_pool

from config import base_args, n_out
from src.models_raw.mdnet import TorchMD_ET

from .embeddings import FoldsEmb, WaveLenEmb


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
            max_num_neighbors=args["max_num_neighbors"],
        )

        self.wl_emb = nn.Sequential(
            nn.Linear(1, 64, bias=False), nn.ReLU(), nn.LayerNorm(64), nn.Linear(64, 64)
        )

        self.head = nn.Sequential(
            # nn.LayerNorm(args["embedding_dimension"] + 64),
            nn.Dropout(args["dropout"]),
            nn.Linear(args["embedding_dimension"] + 64, args["hidden_size"]),
            nn.LayerNorm(args["hidden_size"]),
            nn.ReLU(),
            nn.Dropout(args["dropout"]),
            nn.Linear(args["hidden_size"], n_out),
        )

    def forward(self, x, *args, **kwargs):
        z = x.z
        pos = x.pos
        batch = x.batch
        wl = x.wl.view(-1, 1)

        atom_feats, *_ = self.body(z, pos, batch)
        pooled_feats = global_mean_pool(atom_feats, batch)

        wl_out = self.wl_emb(wl)
        combined_feats = torch.cat([pooled_feats, wl_out], dim=1)

        raman = self.head(combined_feats)
        output = {"raman": (raman, None)}
        return output

    def __str__(self):
        return "MDNet"

    def n_params(self):
        return sum([p.numel() for p in self.parameters()])

    def get_args(self):
        return self.args


class MDNet1(nn.Module):
    def __init__(self, args=None):
        super().__init__()
        self.args = args
        if self.args is not None:
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
            max_num_neighbors=args["max_num_neighbors"],
        )

        hidden_emb = 64
        self.wl_emb = WaveLenEmb(
            args["embedding_dimension"],
            args["hidden_size"],
            hidden_emb,
            args["dropout"],
        )
        self.folds_emb = FoldsEmb(
            args["embedding_dimension"],
            args["hidden_size"],
            hidden_emb,
            args["dropout"],
        )
        self.orientation_head = nn.Sequential(
            nn.Linear(7, args["hidden_size"]),
            nn.SiLU(),
            nn.LayerNorm(args["hidden_size"]),
            nn.Linear(args["hidden_size"], hidden_emb),
        )

        self.head = nn.Sequential(
            nn.Dropout(args["dropout"]),
            nn.Linear(
                args["embedding_dimension"] + hidden_emb * 3, args["hidden_size"]
            ),
            nn.LayerNorm(args["hidden_size"]),
            nn.SiLU(),
            nn.Dropout(args["dropout"]),
            nn.Linear(args["hidden_size"], n_out * 6),
        )

    def forward(self, x, *args, **kwargs):
        z = x.z
        pos = x.pos
        batch = x.batch
        wl = x.wl.view(-1, 1)
        batch_size = wl.shape[0]
        if hasattr(x, "folds"):
            folds = x.folds
        else:
            folds = torch.zeros(batch_size, 4, device=wl.device).long()
            folds[:, -1] = 1

        atom_feats, *_ = self.body(z, pos, batch)
        pooled = global_mean_pool(atom_feats, batch)
        wl_emb = self.wl_emb(wl)
        folds_emb = self.folds_emb(folds)

        cond_vec = (
            x.cond_vec
            if hasattr(x, "cond_vec")
            else torch.zeros(batch_size, 7, device=wl.device)
        )
        cond_vec[:, 0] = 1 - cond_vec[:, 0]
        orient_emb = self.orientation_head(cond_vec)
        combined = torch.cat([pooled, wl_emb, folds_emb, orient_emb], dim=-1)

        raw_tensors = self.head(combined).view(batch_size, n_out, 6)

        R_xx = raw_tensors[:, :, 0]
        R_yy = raw_tensors[:, :, 1]
        R_zz = raw_tensors[:, :, 2]
        R_xy = raw_tensors[:, :, 3]
        R_yz = raw_tensors[:, :, 4]
        R_xz = raw_tensors[:, :, 5]

        a = (R_xx + R_yy + R_zz) / 3.0
        gamma_sq = 0.5 * (
            (R_xx - R_yy) ** 2
            + (R_yy - R_zz) ** 2
            + (R_zz - R_xx) ** 2
            + 6.0 * (R_xy**2 + R_yz**2 + R_xz**2)
        )

        raman = 45.0 * (a**2) + 7.0 * gamma_sq
        output = {"raman": (raman, None)}
        return output

    def __str__(self):
        return "MDNet1"

    def n_params(self):
        return sum([p.numel() for p in self.parameters()])

    def get_args(self):
        return self.args


class MDNet2(nn.Module):
    def __init__(self, args=None):
        super().__init__()
        self.args = args
        if self.args is not None:
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
            max_num_neighbors=args["max_num_neighbors"],
        )

        hidden_emb = 64
        self.wl_emb = WaveLenEmb(
            args["embedding_dimension"],
            args["hidden_size"],
            hidden_emb,
            args["dropout"],
        )
        self.folds_emb = FoldsEmb(
            args["embedding_dimension"],
            args["hidden_size"],
            hidden_emb,
            args["dropout"],
        )
        self.orientation_head = nn.Sequential(
            nn.Linear(7, args["hidden_size"]),
            nn.SiLU(),
            nn.LayerNorm(args["hidden_size"]),
            nn.Linear(args["hidden_size"], hidden_emb),
        )

        # self.interm = nn.Sequential(
        #     nn.Dropout(args["dropout"]),
        #     nn.Linear(args["embedding_dimension"], args["hidden_size"]),
        #     nn.LayerNorm(args["hidden_size"]),
        #     nn.SiLU(),
        # )

        self.raman_head = nn.Sequential(
            # nn.Dropout(args["dropout"]),
            nn.Linear(args["embedding_dimension"] + hidden_emb * 3, args["hidden_size"]),
            nn.LayerNorm(args["hidden_size"]),
            nn.SiLU(),
            nn.Dropout(args["dropout"]),
            nn.Linear(args["hidden_size"], n_out * 6),
        )

        self.ir_head = nn.Sequential(
            # nn.Dropout(args["dropout"]),
            nn.Linear(args["embedding_dimension"], args["hidden_size"]),
            nn.LayerNorm(args["hidden_size"]),
            nn.SiLU(),
            nn.Dropout(args["dropout"]),
            nn.Linear(args["hidden_size"], n_out * 3),
        )

        self.raman_scale = nn.Linear(n_out, 1)
        self.ir_scale = nn.Linear(n_out, 1)

    def forward(self, x, ir_flag=False):
        z = x.z
        pos = x.pos
        batch = x.batch
        wl = x.wl.view(-1, 1)
        batch_size = wl.shape[0]
        if hasattr(x, "folds"):
            folds = x.folds
        else:
            folds = torch.zeros(batch_size, 4, device=wl.device).long()
            folds[:, -1] = 1

        atom_feats, *_ = self.body(z, pos, batch)
        pooled = global_mean_pool(atom_feats, batch)
        # interm = self.interm(pooled)

        wl_emb = self.wl_emb(wl)
        folds_emb = self.folds_emb(folds)
        cond_vec = (
            x.cond_vec
            if hasattr(x, "cond_vec")
            else torch.zeros(batch_size, 7, device=wl.device)
        )
        cond_vec[:, 0] = 1 - cond_vec[:, 0]
        orient_emb = self.orientation_head(cond_vec)
        combined = torch.cat([pooled, wl_emb, folds_emb, orient_emb], dim=-1)

        output = {}
        raman, ram_fact = self.raman(combined)
        output["raman"] = (raman, ram_fact)

        if ir_flag:
            ir, ir_fact = self.ir(pooled[x.has_ir])
            output["ir"] = (ir, ir_fact)

        return output

    def raman(self, x):
        raw_tensors = self.raman_head(x).view(x.shape[0], n_out, 6)

        R_xx = raw_tensors[:, :, 0]
        R_yy = raw_tensors[:, :, 1]
        R_zz = raw_tensors[:, :, 2]
        R_xy = raw_tensors[:, :, 3]
        R_yz = raw_tensors[:, :, 4]
        R_xz = raw_tensors[:, :, 5]

        a = (R_xx + R_yy + R_zz) / 3.0
        gamma_sq = 0.5 * (
            (R_xx - R_yy) ** 2
            + (R_yy - R_zz) ** 2
            + (R_zz - R_xx) ** 2
            + 6.0 * (R_xy**2 + R_yz**2 + R_xz**2)
        )

        output = 45.0 * (a**2) + 7.0 * gamma_sq

        # scale = 100 * self.raman_scale(raw_tensors[:, :, 6])
        scale = None

        return output, scale

    def ir(self, x):
        raw_tensors = self.ir_head(x).view(x.shape[0], n_out, 3)

        mu_x = raw_tensors[:, :, 0]
        mu_y = raw_tensors[:, :, 1]
        mu_z = raw_tensors[:, :, 2]

        output = mu_x**2 + mu_y**2 + mu_z**2

        # scale = 100 * self.ir_scale(raw_tensors[:, :, 3])
        scale = None

        return output, scale

    def __str__(self):
        return "MDNet2"

    def n_params(self):
        return sum([p.numel() for p in self.parameters()])

    def get_args(self):
        return self.args


class MDNetSide(nn.Module):
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
            max_num_neighbors=args["max_num_neighbors"],
        )

        hidden_emb = 64
        self.wl_emb = WaveLenEmb(
            args["embedding_dimension"],
            args["hidden_size"],
            hidden_emb,
            args["dropout"],
        )
        # self.folds_emb = FoldsEmb(args["embedding_dimension"],args["hidden_size"],hidden_emb,args["dropout"])
        # self.orientation_head = nn.Sequential(
        #     nn.Linear(7, args["hidden_size"]),
        #     nn.SiLU(),
        #     nn.LayerNorm(args["hidden_size"]),
        #     nn.Linear(args["hidden_size"], hidden_emb)
        # )

        self.head = nn.Sequential(
            nn.Dropout(args["dropout"]),
            nn.Linear(args["embedding_dimension"] + hidden_emb, args["hidden_size"]),
            nn.LayerNorm(args["hidden_size"]),
            nn.SiLU(),
            nn.Dropout(args["dropout"]),
            nn.Linear(args["hidden_size"], n_out * 6),
        )

    def forward(self,  x, *args, **kwargs):
        z = x.z
        pos = x.pos
        batch = x.batch
        wl = x.wl.view(-1, 1)
        # folds = x.folds
        batch_size = wl.shape[0]
        # cond_vec = x.cond_vec.float() if hasattr(x, "cond_vec") else torch.zeros(batch_size, 7, device=x.device).float()
        if not hasattr(x, "wl"):
            wl = torch.tensor([5.14] * x.num_graphs, device=x.x.device)
        else:
            wl = x.wl.view(-1, 1)

        atom_feats, *_ = self.body(z, pos, batch)
        pooled = global_mean_pool(atom_feats, batch)
        wl_emb = self.wl_emb(wl)
        # folds_emb = self.folds_emb(folds)
        # orient_emb = self.orientation_head(cond_vec)

        combined = torch.cat([pooled, wl_emb], dim=-1)

        raw_tensors = self.head(combined).view(batch_size, n_out, 6) ** 2

        R_xx = raw_tensors[:, :, 0]
        R_yy = raw_tensors[:, :, 1]
        R_zz = raw_tensors[:, :, 2]
        R_xy = raw_tensors[:, :, 3]
        R_yz = raw_tensors[:, :, 4]
        R_xz = raw_tensors[:, :, 5]

        a = (R_xx + R_yy + R_zz) / 3.0
        gamma_sq = 0.5 * (
            (R_xx - R_yy) ** 2
            + (R_yy - R_zz) ** 2
            + (R_zz - R_xx) ** 2
            + 6.0 * (R_xy**2 + R_yz**2 + R_xz**2)
        )

        raman = 45.0 * (a**2) + 7.0 * gamma_sq

        output = {"raman": (raman, None)}
        return output

    def __str__(self):
        return "MDNetSide"

    def n_params(self):
        return sum([p.numel() for p in self.parameters()])

    def get_args(self):
        return self.args
