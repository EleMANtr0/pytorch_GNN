from torch_geometric.loader import DataLoader
import numpy as np
from scipy.signal import find_peaks, peak_widths
import matplotlib.pyplot as plt
from src.loss import KLDivLoss
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

def scipy_peaks(pred: np.ndarray,true: np.ndarray, err = 2,prom=0.15):
    tp, tn, fp, fn = 0, 0, 0, 0
    ppeaks, _ = find_peaks(pred, prominence=prom, width=(1,20), height=(0.3,10))
    tpeaks, _ = find_peaks(true, prominence=prom, width=(1,20), height=(0.1,10))
    ppeaks_copy = list(ppeaks.copy())
    if len(tpeaks) >= 2:
        min_dist = np.min(np.diff(np.sort(tpeaks)))
        if min_dist < 2*err:
            err = max(1, min_dist // 2)
    # print("err: ", err)
    # print("pred: ", ppeaks)
    # print("true: ",tpeaks)
    for true_peak in tpeaks:
        found = False
        for pred_peak in ppeaks_copy:
            if abs(true_peak-pred_peak) <= err:
                tp += 1
                found = True
                ppeaks_copy.remove(pred_peak)
                break
        if not found:
            fn += 1
    fp = len(ppeaks_copy)
    tn = len(pred) - (tp + fp + fn)
    # res = (tp,tn,fp,fn)
    # acc_dict = {"tp":0,"tn":0,"fp":0,"fn":0}
    # for n, k in enumerate(acc_dict.keys()):
    #     acc_dict[k] += res[n]
    # return acc_dict
    return tp, tn, fp, fn

def find(pred,true,i=0):
    for prom in [0.15]:
        peaks, props = find_peaks(pred, prominence=prom, width=(1,20), height=(0.3,10))
        widths, *_ = peak_widths(pred,peaks)
        print("pred: ",peaks)
        print("pred width: ", widths)
        plt.plot(pred, label="predicted")
        plt.title(f"prom={prom}")
        plt.plot(peaks, pred[peaks], "x")
        peaks, props = find_peaks(true, prominence=prom, width=(1,20), height=(0.3,10))
        widths, *_ = peak_widths(true,peaks)
        print("true: ",peaks)
        print("true width: ", widths)
        plt.plot(true, label="true")
        plt.plot(peaks, true[peaks], "x")
        plt.legend()
        plt.show()

def calc_single(true,pred,err=0,i=0):
    acc_dict = {"tp":0,"tn":0,"fp":0,"fn":0}
    res = scipy_peaks(pred, true, err)
    for n, k in enumerate(acc_dict.keys()):
        acc_dict[k] += res[n]
    return acc_dict#, res[4], res[5]

def calc_all(truths,preds,err=0):
    acc_dict = {"tp":0,"tn":0,"fp":0,"fn":0}
    for i in range(len(preds)):
        res = scipy_peaks(preds[i], truths[i], err)
        for n, k in enumerate(acc_dict.keys()):
            acc_dict[k] += res[n]
    return acc_dict
    return (acc_dict["tp"] + acc_dict["tn"]) / sum(acc_dict.values())

def kldiv(model,device,dataset,wl=None, is_schnet=False):
    model = model.to(device)
    loss = 0
    model.eval()
    bad = {}
    dataloader = DataLoader(dataset,batch_size=1)
    if is_schnet:
        with torch.inference_mode():
            count = 0
            for i, data in enumerate(dataloader):
                data.to(device)
                pred = model(data.z, data.pos, data.batch)
                curr_loss = kl_loss(pred, data.y)
                loss += curr_loss
                count += 1
                bad[i] = curr_loss.item()
            loss /= count
        return loss.item(), {k:v for k,v in sorted(bad.items(),key = lambda x: x[1],reverse=True)}
    if wl:
        wl = set(list(np.array([wl]).flatten()))
        with torch.inference_mode():
            count = 0
            for i, data in enumerate(dataloader):
                if round(100*data.wl.item()) in wl:
                    data.to(device)
                    pred = model(data)
                    curr_loss = kl_loss(pred, data.y)
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