import time
import os
from datetime import datetime
import copy
from pathlib import Path
import json
from tqdm import tqdm
import argparse
import logging
from dataclasses import dataclass
import sys
sys.path.append("..")

import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

from src.models import MDNet
from src.data.dataset import Crystals
from src.utils.loss import CombLoss
# from src.utils.util import drop  # for dropping features just to test
from config import loss_fn_dict, models_dir, data_path, logdir, tblogdir

logger = logging.Logger("train")
    

def log(msg):
    print(msg)
    logger.info(msg)

def tb_log(sm:SummaryWriter, epoch, loss, name):
    sm.add_scalar(f"{name}/loss", loss, epoch)

class DummyScheduler:
    def __init__(self, **kwargs):
        pass
    def step(self):
        pass

@dataclass
class TrainContext:
    model: nn.Module
    name: str
    epochs: int
    epoch: int
    lr: float
    lr_decay: float
    lr_decay_steps: int
    weight_decay: float
    eval_iters: int
    train_dataset: Crystals
    val_dataset: Crystals
    batch_size: int
    sleep: float
    tblogdir: str | Path
    logdir: str | Path
    model_dir: Path
    scheduler: torch.optim.lr_scheduler.LRScheduler = DummyScheduler()

@dataclass
class TrainRes:
    save_model_dir: str
    best_model: nn.Module
    best_val_loss: float

def val(model, dataloader):
    metric_fn = loss_fn_dict["kldiv"]
    with torch.inference_mode():
        model.eval()
        running_loss = 0
        for data in dataloader:
            data = data.to(device)
            pred = model(data)
            running_loss += metric_fn(pred, data.y).item() * len(data.x)
        return running_loss / len(dataloader.sampler)

def train(context: TrainContext, result:TrainRes):
        optimizer = torch.optim.AdamW(context.model.parameters(), lr=context.lr, weight_decay=context.weight_decay)
        train_loader = DataLoader(context.train_dataset,batch_size=context.batch_size,shuffle=True)
        val_loader = DataLoader(context.val_dataset,batch_size=context.batch_size,shuffle=True)
        scheduler = context.scheduler
        model_name = context.name
        sum_writer = SummaryWriter(log_dir=context.tblogdir)
        try:
            for epoch in tqdm(range(context.epoch, context.epoch + context.epochs+1)):
                model.train()
                running_loss = 0
                for data in train_loader:
                    data = data.to(device)
                    pred = model(data)
                    loss = loss_fn(pred, data.y)
                    running_loss += loss.item() * data.x.shape[0]
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()
                scheduler.step()

                if epoch % context.eval_iters == 0:
                    train_kl = val(model, train_loader).item()
                    val_kl = val(model, val_loader).item()
                    tb_log(sum_writer, epoch, train_kl, "train")
                    tb_log(sum_writer, epoch, val_kl, "val")
                    log(f"validation loss: {val_kl}")
                    log(f"train loss: {train_kl}")
                if val_kl < result.best_val_loss:
                    result.best_val_loss = val_kl
                    result.best_model.load_state_dict(model.state_dict)
                    result.save_model_dir = context.model_dir / f'{model_name}_best_kl{val_kl:.4f}.pt'
                    torch.save(model.state_dict(),model_save_path)
                    log(f"validation {loss_name} loss: {val_kl.item()}")
                    log(f"model saved to {model_save_path}")
                time.sleep(context.sleep)
        except KeyboardInterrupt:
            log(f"interrupted on epoch {epoch}")
            log(f"validation {loss_name} loss: {val_kl.item()}")
            log(f"train {loss_name} loss: {train_kl.item()}")
        finally:
            return epoch,val_kl,model_save_path,best_model_wts

if __name__ == "__main__":
    now = datetime.now()
    dt_string = now.strftime("%m%d_%H:%M")
    logging.basicConfig(filename=logdir/f"train_log{dt_string}.log", format="%(message)s")
    logger.setLevel(logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", nargs="?")
    parser.add_argument("--epoch", nargs="?", default=0, type=int)
    parser.add_argument("--weight_decay", nargs="?", default=0.0, type=float)
    parser.add_argument("--lr", nargs="?", default=1e-4, type=float)
    parser.add_argument("--eval_steps", nargs="?", default=10, type=int)
    parser.add_argument("--batch_size", nargs="?", default=64, type=int)
    parser.add_argument("--loss_fn", nargs="*", default=["mse","cossim"], type=list[str])
    parser.add_argument("--lr_decay", nargs="?", default=1.0, type=float)
    parser.add_argument("--lr_decay_steps", nargs="?", default=1000, type=int)
    parser.add_argument("--lr_decayer", nargs="?", default=None, type=list[str])

    parser.add_argument("--model_name", nargs="?", default="")
    parser.add_argument("--dataset", nargs="?", default="v6.pt")

    parser.add_argument("--emb_dim", nargs="?", default=64, type=int)
    parser.add_argument("--num_heads", nargs="?", default=8, type=int)
    parser.add_argument("--neighbor_emb", nargs="?", default=True, type=bool)

    parser.add_argument("--sleep", nargs="?", default=0.0, type=float)

    if not torch.cuda.is_available():
        raise RuntimeError("Cuda not available. Interrupting")
    device = torch.device('cuda')

    args = parser.parse_args()
    model_args = {
        "embedding_dimension": args.emb_dim,
        "attn_activation": "silu",
        "num_heads": args.num_heads,
        "neighbor_embedding": args.neighbor_emb
    }

    losses = args.loss_fn
    if len(losses) == 1:
        loss_fn = loss_fn_dict[losses[0]]
    else:
        loss_fn = CombLoss(*[(1, loss_fn_dict[loss]) for loss in losses])


    model = MDNet(model_args).to(device)
    loss_name = repr(loss_fn)
    model_name = str(model) if args.model_name == "" else args.model_name

    dataset_path = data_path / args.dataset
    dataset = Crystals(dataset_path)
    save_model_dir = Path(f'models/{model_name}_{loss_name}')
    save_model_dir.mkdir(parents=True,exist_ok=True)
    n_params = model.n_params()
    log(f"{model_name} {n_params/1e6:.2f}M")

    train_dataset, val_dataset, test_dataset = dataset.get_splits(deterministic=True)

    display_epochs = args.eval_steps
    best_model = copy.deepcopy(model.state_dict())
    best_val_loss = 0.009
    model_save_path = None

    epoch,val_kl,model_save_path,best_model_wts = train(model,0,1000,lr=1e-4)
