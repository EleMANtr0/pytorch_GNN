import torch
import torch.nn as nn

from .embeddings import WaveLenEmb
from config import n_out, base_args
from src.models_raw.matformer import Matformer as MF


class Matformer(nn.Module):
    def __init__(self, args=None):
        super().__init__()
        self.args = args
        if self.args is not None:
            for k, v in args.items():
                base_args[k] = v
        hidden_emb = 64
        args = base_args
        self.body = MF(
            node_features=args["embedding_dimension"],
            edge_features=args.get("edge_features", 128),
            conv_layers=args["num_layers"],
            node_layer_head=args["num_heads"],
            max_z=args["max_z"],
            cutoff=args["cutoff_upper"],
            use_angle_lattice=args.get("use_angle_lattice", True)
        )
        self.wl_emb = WaveLenEmb(
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
        self.raman_head = nn.Sequential(
            nn.Linear(args["embedding_dimension"] + hidden_emb * 2 + args["hidden_size"], args["hidden_size"]),
            nn.LayerNorm(args["hidden_size"]),
            nn.SiLU(),
            nn.Dropout(args["dropout"]),
            nn.Linear(args["hidden_size"], n_out * 7),
        )
        self.ir_features = nn.Sequential(
            nn.Linear(args["embedding_dimension"], args["hidden_size"]),
            nn.LayerNorm(args["hidden_size"]),
            nn.SiLU(),
            nn.Dropout(args["dropout"])
        )
        self.ir_head = nn.Linear(args["hidden_size"], n_out * 3)

    def forward(self, x, ir_flag=False):
        batch_size = x.wl.shape[0]
        pooled_scalar = self.body(x)
        wl_emb = self.wl_emb(x.wl.view(-1, 1))
        cond_vec = x.cond_vec.clone() if hasattr(x, "cond_vec") else torch.zeros(batch_size, 7, device=x.wl.device)
        cond_vec[:, 0] = 1.0 - cond_vec[:, 0]
        orient_emb = self.orientation_head(cond_vec)
        ir_features = self.ir_features(pooled_scalar)
        combined = torch.cat([pooled_scalar, wl_emb, orient_emb, ir_features.detach()], dim=-1)
        output = {}
        if ir_flag:
            output["ir"] = self.ir(ir_features[x.has_ir])
        output["raman"] = self.raman(combined)
        return output

    def raman(self, x):
        raw_tensors = self.raman_head(x).view(x.shape[0], n_out, 7)

        R_xx, R_yy, R_zz = raw_tensors[:, :, 0], raw_tensors[:, :, 1], raw_tensors[:, :, 2]
        R_xy, R_yz, R_xz = raw_tensors[:, :, 3], raw_tensors[:, :, 4], raw_tensors[:, :, 5]
        bg = raw_tensors[:, :, 6]

        a = (R_xx + R_yy + R_zz) / 3.0
        gamma_sq = 0.5 * (
            (R_xx - R_yy) ** 2 + (R_yy - R_zz) ** 2 + (R_zz - R_xx) ** 2 +
            6.0 * (R_xy**2 + R_yz**2 + R_xz**2)
        )

        output = 45.0 * (a**2) + 7.0 * gamma_sq
        background = torch.sigmoid(bg) - 0.5
        output = (output + background)
        output = output / output.max(dim=-1, keepdim=True)[0]
        return output

    def ir(self, x):
        raw_tensors = self.ir_head(x).view(x.shape[0], n_out, 3)
        mu_x, mu_y, mu_z = raw_tensors[:, :, 0], raw_tensors[:, :, 1], raw_tensors[:, :, 2]
        output = mu_x**2 + mu_y**2 + mu_z**2
        output = output / output.max(dim=-1, keepdim=True)[0]
        return output

    def __str__(self):
        return "Matformer"

    def n_params(self):
        return sum([p.numel() for p in self.parameters()])

    def get_args(self):
        return self.args