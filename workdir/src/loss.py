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

    def forward(self,y_pred,y_true):
        res = 0
        for i in self.loss:
            res += i[0] * i[1](y_pred,y_true)
        return res
    def __str__(self):
        name = ""
        for i in self.loss:
            name += str(i[0]) + str(i[1]) + "_"
        name = name[:-1]
        return name


class AdaptiveCombLoss(nn.Module):
    def __init__(self, *CF, mode='dynamic'):
        super().__init__()
        self.mode = mode
        self.loss_functions = CF
        if mode == 'dynamic':
            self.alpha = 0.5
            self.history = [0.0] * len(self.loss_functions)
            self.current_weights = [1.0 / len(self.loss_functions)] * len(self.loss_functions)
        elif mode == 'learnable':
            initial_weights = [weight for weight, _ in self.loss_functions]
            self.weights = nn.Parameter(torch.tensor(initial_weights, dtype=torch.float32))

    def forward(self, y_pred, y_true):
        if self.mode == 'dynamic':
            losses = []
            for _, loss_fn in self.loss_functions:
                loss_value = loss_fn(y_pred, y_true)
                losses.append(loss_value.detach().item())

            for i, val in enumerate(losses):
                if self.history[i] == 0:
                    self.history[i] = val
                else:
                    self.history[i] = self.alpha * val + (1 - self.alpha) * self.history[i]

            weights = [1.0 / (h + 1e-8) for h in self.history]
            s = sum(weights)
            if s > 0:
                weights = [w / s for w in weights]
            else:
                n = len(weights)
                weights = [1.0 / n] * n
            self.current_weights = weights

            total_loss = 0
            for i, (_, loss_fn) in enumerate(self.loss_functions):
                total_loss += weights[i] * loss_fn(y_pred, y_true)
            return total_loss

        elif self.mode == 'learnable':
            normalized_weights = F.softmax(self.weights, dim=0)
            total_loss = 0
            for i, (_, loss_fn) in enumerate(self.loss_functions):
                total_loss += normalized_weights[i] * loss_fn(y_pred, y_true)
            return total_loss

    def get_weights(self):
        if self.mode == 'dynamic':
            return self.current_weights
        elif self.mode == 'learnable':
            return F.softmax(self.weights, dim=0).tolist()

    def __str__(self):
        name = ""
        if self.mode == "dynamic":
            name = "adaptive_"
        else:
            name = "learnable_"
        for _, loss_fn in self.loss_functions:
            name += f"{repr(loss_fn)}_"
        return name[:-1]
