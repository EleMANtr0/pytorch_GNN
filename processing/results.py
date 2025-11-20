from pathlib import Path
import json
import numpy as np
import json
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import gc
from PIL import Image
import matplotlib.pyplot as plt
from itertools import combinations
import re

names = dict(
    model_mse = "predict/painn_mse",
    model_mae = "predict/painn_mae",
    model_msle = "predict/painn_msle",
    model_rmse = "predict/painn_rmse",
    model_iou = "predict/painn_iou",
    model_kl = "predict/painn_kl",
    model_cos = "predict/painn_cossim",
    model_comb1 = "predict/painn_1mse_1mae_0.2rmse_1cossim_0.5kl",
    model_comb2 = "predict/painn_1mse_1mae_0.3msle_0.1rmse_0.4cossim_0.3kl",
    model_comb3 = "predict/painn_1mse_1mae_1msle_1rmse_1cossim_1kl",
    model_learn = "predict/painn_learnable_mse_mae_msle_rmse_cossim_kl_iou",
    model_learn2 = "predict/painn_learnable_mse_mae_msle_cossim_kl",
    model_2mse = "predict/painn2_mse",
    model_2comb1 = "predict/painn2_0.5mse_0.2rmse_1cossim_0.5kl",
    model_2comb2 = "predict/painn2_0.5mse_0.5mae_0.2rmse_0.2msle_1cossim_0.5kl_1iou",
    model_2comb3 = "predict/painn2_3mse_1mae_0.7cossim_15kl_0.3iou",
    model_2comb4 = "predict/painn2_1mse_1mae_1msle_1rmse_1cossim_1kl_0.5iou",
    model_2comb5 = "predict/painn2_1mse_1mae_1msle_1rmse_1cossim_1kl",
    model_2comb6 = "predict/painn2_1mse_1cossim",
    model_2adapt1 = "predict/painn2_adaptive_mse_mae_rmse_msle_cossim_kl_iou",
    model_2adapt2 = "predict/painn2_adaptive_mse_mae_cossim_kl_iou_514_532",
    model_2learn = "predict/painn2_learnable_mse_mae_msle_rmse_cossim_kl_iou",
    model_ensemble = "predict/painn_ensemble1"
)

class bad_spectra:
    def __init__(self,json_path,divide = True,rename = None):
        badi_p = Path(json_path)
        self.bad_idx = json.loads(badi_p.read_text())
        for name, d in self.bad_idx.items():
            self.bad_idx[name] = {int(k): v for k, v in d.items()}
        self.bad_idx514 = {}
        temp = {}
        temp1 = {}
        if rename is not None:
            for name,v in self.bad_idx.items():
                if name in rename.keys():
                    temp1[rename[name]] = v
                else:
                    temp1[name] = v
            self.bad_idx = temp1
        if divide:
            for k,v in self.bad_idx.items():
                if k.startswith("model_2") and not k.endswith("_wl514"):
                    temp[k] = v
                else:
                    self.bad_idx514[k] = v

            self.bad_idx = temp.copy()
        res = {}
        for name in self.bad_idx.keys():
            temp = {idx:loss for idx, loss in sorted(self.bad_idx[name].items(), key=lambda x: x[1], reverse=True)}
            res[name] = temp
        self.bad = res.copy()
        res = {}
        for name in self.bad_idx514.keys():
            temp = {idx:loss for idx, loss in sorted(self.bad_idx514[name].items(), key=lambda x: x[1], reverse=True)}
            res[name] = temp
        self.bad514 = res.copy()

    def split(self,treshold,single = False):
        new = defaultdict(dict)
        if single:
            for key,dic in self.bad_idx514.items():
                for k,v in dic.items():
                    if v > treshold:
                        new[key][k] = v
            self.bad_idx514 = new.copy()
        else:
            for key,dic in self.bad_idx.items():
                for k,v in dic.items():
                    if v > treshold:
                        new[key][k] = v
            self.bad_idx = new.copy()

    def see_loss(self,name,idx):
        idx = list(np.array([idx]).flatten())
        res = defaultdict(dict)
        for i in idx:
            res[name][i] = self.bad[name][i]
        return res

    def get_set(self,names=None,single=False):
        bad_idx = {}
        if names:
            for i in names:
                if single:
                    bad_idx[i] = set(self.bad_idx514[i].keys())
                else:
                    bad_idx[i] = set(self.bad_idx[i].keys())
            return bad_idx
        for k,v in self.bad_idx.items():
            bad_idx[k] = set(v)
        return bad_idx

    def get_np(self,names=None,single=False):
        bad_idx = {}
        if names:
            for i in names:
                if single:
                    bad_idx[i] = np.array(self.bad_idx514[i].keys())
                else:
                    bad_idx[i] = np.array(self.bad_idx[i].keys())
            return bad_idx
        for k,v in self.bad_idx.items():
            bad_idx[k] = np.array(v)
        return bad_idx

    def get(self,names=None,single=False):
        if names:
            if single:
                bad_idx = {k:v for k,v in self.bad514.items() if k in names}
            else:
                bad_idx = {k:v for k,v in self.bad.items() if k in names}
            return bad_idx
        if single:
            return self.bad514
        return self.bad

    def inter(self,names: list[str],single=False):
        for i in range(len(names)):
            if not names[i].startswith("model_"):
                names[i] = "model_" + names[i]
        bad_set = self.get_set(names,single)
        res = bad_set[names[0]]
        for i in names:
            res = res & bad_set[i]
        if len(names) == 2:
            return {"intersection":np.array(sorted(list(res),key=int,reverse=False))}
        return np.array(res)

    def symdiff(self,names: list[str],single=False):
        for i in range(len(names)):
            if not names[i].startswith("model_"):
                names[i] = "model_" + names[i]
        bad_set = self.get_set(names,single)
        res = bad_set[names[0]]
        for i in names:
            res = res ^ bad_set[i]
        if len(names) == 2:
            return {"symmetrical difference":np.array(sorted(list(res),key=int,reverse=False))}
        return np.array(res)

    def diff(self,names: list[str],single=False):
        for i in range(len(names)):
            if not names[i].startswith("model_"):
                names[i] = "model_" + names[i]
        bad_set = self.get_set(names,single)
        res = {names[0]:np.array(sorted(list(bad_set[names[0]]-bad_set[names[1]]),key=int,reverse=False)),names[1]:np.array(sorted(list(bad_set[names[1]]-bad_set[names[0]]),key=int,reverse=False))}
        return res

    def all_op(self,op,single=False):
        res = {}
        if single:
            keys = self.bad_idx514.keys()
        else:
            keys = self.bad_idx.keys()
        key_pairs = combinations(keys,2)
        if op == "diff":
            op = self.diff
        elif op == "symdiff":
            op = self.symdiff
        elif op == "inter":
            op = self.inter
        else:
            raise ValueError("no valid operation assigned")
        for i in key_pairs:
            res[f"{i[0][6:]}_{i[1][6:]}"] = op(list(i),single)
        self.res = res
        return res

    def getall(self,name,verbose = False):
        res = {}
        pattern = rf'(^|_)*{re.escape(name)}(_|$)*'
        for k,v in self.res.items():
            # print("_" + name in k or name + "_" in k,k)
            # if "_" + name in k or name + "_" in k:
            if re.search(pattern,k):
                res[k] = v
                if verbose:
                    print(f"{k}:\n")
                    for k,v in v.items():
                        print(f"{k}:{v}\n")
                    print("\n")
        return res

    def visualize(self,name: str,i,ax=None,save_dir=None,img_title=None):
        if not name.startswith("model_"):
            name = "model_" + name
        dir = None
        dirs = None
        path_list = []
        match = None
        combs = ["comb","adapt","learn"]
        if any(i in name for i in combs) and not name.endswith("_wl514"):
            dirs = ["514","532","780","785"]
        else:
            dir = "514"
        if name.endswith("_wl514"):
            name = name[:-6]
        path = Path(names[name] + "/val")
        path.mkdir(parents=True,exist_ok=True)
        if dir:
            path_list.append((path / dir))
        elif dirs:
            for d in dirs:
                path_list.append((path / d))
        else:
            print("path list is empty")
            print(path_list)
            return

        pattern = re.compile(rf"(?:(?<=\D)|^){i}$")
        for directory in path_list:
            for file_path in directory.rglob("*"):
                if file_path.is_file() and str(i) in file_path.name and pattern.search(file_path.stem):
                    match = file_path.resolve()

        if match is None:
            print("not found: ", i)
            return
        try:
            if name in self.new_names:
                title = self.new_names[name]
            else:
                title = name
        except Exception:
            title = name
        try:
            title += " kl_div: " + f"{self.bad[name][i]:.4f}"
        except KeyError:
            title += " kl_div: " + f"{self.bad514[name][i]:.4f}"
        except Exception as e:
            print(e)
            return
        print(match)
        img = Image.open(match)
        img_array = np.array(img)
        img.close()
        if ax:
            ax.imshow(img_array)
            ax.axis("off")
            ax.set_title(title)
        else:
            plt.figure()
            plt.title(title)
            plt.imshow(img_array)
            plt.axis("off")
            plt.show()
        if not save_dir:
            if ax is None:
                plt.close()
                gc.collect()
        else:
            if ax is None:
                save_dir = Path(save_dir)
                save_dir.mkdir(parents=True,exist_ok=True)
                plt.savefig(save_dir/f"{str(img_title)}.png")
                plt.close()
                gc.collect()


    def rename(self,new_names):
        self.new_names = new_names
        return new_names

    def compare(self,names,i):
        for k,n in enumerate(names):
            self.visualize(n,i)

    def compare_all(self,idx,x,y,figsize,save_dir = None,title=None):
        fig, ax = plt.subplots(x,y,figsize=figsize)
        names_lst = list(self.bad.keys()) + (list(self.bad514.keys()))
        directory = Path("predict/raw/val")
        pattern = re.compile(rf"(?:(?<=\D)|^){idx}$")
        match = None
        for file_path in directory.rglob("*"):
            if file_path.is_file() and str(idx) in file_path.name and pattern.search(file_path.stem):
                match = file_path.resolve()
        if match is not None:
            img = Image.open(match)
            img_array = np.array(img)
            img.close()
            ax[0,0].imshow(img_array)
            ax[0,0].axis("off")
            ax[0,0].set_title("real spectra")
            print("----------------------------------------------------------------------")
        t = 0
        for i in range(x):
            for j in range(y):
                if match:
                    if i == 0 and j == 0:
                        continue
                try:
                    self.visualize(names_lst[t],idx,ax[i,j])
                except IndexError:
                    break
                t+=1
        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True,exist_ok=True)
            fig.savefig(save_dir/str(title))
        else:
            fig.show()
        plt.close(fig)
        gc.collect()
        return fig

    def worst(self,n):
        res = {}
        for name in self.bad.keys():
            res[name] = {k:v for i, (k, v) in enumerate(self.bad[name].items()) if i < n}
        return res

    def best(self,n):
        res = {}
        for name in self.bad.keys():
            length = len(list(self.bad[name].keys()))
            res[name] = {k:v for i, (k, v) in enumerate(self.bad[name].items()) if i >= length - n}
        return res