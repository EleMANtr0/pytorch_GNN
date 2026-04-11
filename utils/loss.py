import torch
import torch.nn as nn
import torch.nn.functional as F


class L1Loss(nn.L1Loss):
    def __str__(self):
        return "mae"

class MSELoss(nn.MSELoss):
    def __str__(self):
        return "mse"

class MSLELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, y_pred, y_true):
        cos_sim = F.cosine_similarity(y_pred, y_true, dim=-1, eps=self.eps)
        loss = 1 - cos_sim.mean()
        return loss

    def __str__(self):
        return "msle"

class RMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse_ls = nn.MSELoss()

    def forward(self,y_pred,y_true):
        return torch.sqrt(self.mse_ls(y_pred,y_true))

    def __str__(self):
        return "rmse"

class CosSimLoss(nn.Module):
    def __init__(self,eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self,y_pred, y_true):
        y_pred_norm = y_pred / (y_pred.norm(dim=-1, keepdim=True) + self.eps)
        y_true_norm = y_true / (y_true.norm(dim=-1, keepdim=True) + self.eps)

        cos_sim = (y_pred_norm * y_true_norm).sum(dim=-1)
        loss = 1 - cos_sim.mean()
        return loss

    def __str__(self):
        return "cossim"

class IoULoss(nn.Module):
    def __init__(self, eps=0):
        super().__init__()
        self.eps = eps

    def forward(self, y_pred, y_true):
        y_pred = y_pred.view(y_pred.size(0), -1)
        y_true = y_true.view(y_true.size(0), -1)

        intersection = torch.min(y_pred,y_true).sum(dim=1)
        union = torch.max(y_pred,y_true).sum(dim=1)

        loss = 1 - (intersection + self.eps)/(union + self.eps)

        return loss.mean()

    def __str__(self):
        return "iou"

class KLDivLoss(nn.Module):
    def __init__(self,eps=1e-8):
        super().__init__()
        self.eps = eps
    def forward(self,y_pred, y_true):
        log_q = F.log_softmax(y_pred, dim=1)
        p = F.softmax(y_true, dim=1) if y_true.dim() == 2 else y_true

        # KL(P || Q) = sum(P * (logP - logQ))
        log_p = torch.log(p + self.eps)
        kl = torch.sum(p * (log_p - log_q), dim=1)
        return kl.mean()

    def __str__(self):
        return "kldiv"

class CombLoss(nn.Module):
    def __init__(self,*CF):
        super().__init__()
        self.loss = CF

    def forward(self, y_pred, y_true):
        res = 0
        for i in self.loss:
            res += i[0] * i[1](y_pred, y_true)
        return res
    
    def __str__(self):
        name = ""
        for i in self.loss:
            name += str(i[0]) + str(i[1]) + "_"
        name = name[:-1]
        return name


class MultiTaskLoss(nn.Module):
    def __init__(self, loss_fn: CombLoss):
        super().__init__()
        self.loss_fn = loss_fn
        self.log_vars = nn.Parameter(torch.zeros(2))

    def forward(self, pred: tuple, target: tuple):
        precision_main = torch.exp(-self.log_vars[0])
        precision_aux = torch.exp(-self.log_vars[1])
        total_loss = precision_main * self.loss_fn(pred[0], target[0]) + precision_aux * self.loss_fn(pred[1], target[1]) + self.log_vars[0] + self.log_vars[1]
        return total_loss
    
    def __str__(self):
        return f"multi_{self.loss_fn}"


loss_fn_dict = {
    "mse": MSELoss(), 
    "mae":L1Loss(), 
    "msle": MSLELoss(), 
    "rmse": RMSELoss(), 
    "cossim": CosSimLoss(), 
    "iou": IoULoss(), 
    "kldiv": KLDivLoss(),
    "multi": MultiTaskLoss
    }
