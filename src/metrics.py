from torch_geometric.loader import DataLoader
import numpy as np
# import matplotlib.pyplot as plt
from utils.loss import KLDivLoss
import torch

kl_loss = KLDivLoss()

class Process:
    def __init__(self,model,device,dataset):
        model = model.to(device)
        self.preds = []
        self.truths = []
        for i, data in enumerate(DataLoader(dataset, batch_size=1)):
            data = data.to(device)
            self.preds.append(model(data).detach().cpu().numpy().flatten())
            self.truths.append(data.y.detach().cpu().numpy().flatten())

    @classmethod
    def init514(cls,model,device,dataset):
        self = cls.__new__(cls)
        model = model.to(device)
        self.preds = []
        self.truths = []
        for i, data in enumerate(DataLoader(dataset, batch_size=1)):
            if round(100*data.wl.item()) == 514:
                data = data.to(device)
                self.preds.append(model(data).detach().cpu().numpy().flatten())
                self.truths.append(data.y.detach().cpu().numpy().flatten())
        return self

    def get(self,i):
        return self.preds[i], self.truths[i]

    def get_all(self):
        return self.preds, self.truths

def kldiv(model,device,dataloader,wl=None, is_schnet=False):
    model = model.to(device)
    loss = 0
    model.eval()
    bad = {}
    if is_schnet:
        with torch.inference_mode():
            count = 0
            for i, data in enumerate(dataloader):
                data.to(device)
                pred = model(data.z, data.pos, data.batch)
                raman, scale = pred["raman"]
                curr_loss = kl_loss(raman, data.y)
                loss += curr_loss
                count += 1
                bad[i] = curr_loss.item()
            loss /= count
        return loss.item(), {k:v for k,v in sorted(bad.items(),key = lambda x: x[1],reverse=True)}
    if wl is not None:
        wl = set(list(np.array([wl]).flatten()))
        with torch.inference_mode():
            count = 0
            for i, data in enumerate(dataloader):
                if round(100 * data.wl.item()) in wl:
                    data.to(device)
                    pred = model(data)
                    raman, scale = pred["raman"]
                    curr_loss = kl_loss(raman, data.y)
                    loss += curr_loss
                    count += 1
                    bad[i] = curr_loss.item()
            loss /= count
    else:
        with torch.inference_mode():
            count = 0
            for i, data in enumerate(dataloader):
                data.to(device)
                pred = model(data)
                curr_loss = kl_loss(pred, data.y)
                loss += curr_loss
                count += 1
                bad[i] = curr_loss.item()
            loss /= count
    return loss.item(), {k:v for k,v in sorted(bad.items(),key = lambda x: x[1],reverse=True)}