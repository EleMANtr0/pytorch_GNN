import json
from time import time
from datetime import datetime
from tqdm import tqdm
import argparse
import logging
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np

from src.models import MDNet2, models_dict
from src.data.dataset import Crystals
from src.loss import CombLoss, L1Loss
from config import loss_fn_dict, models_dir, data_path, logdir, tblogdir
from utils import VersionFromName, DatasetContext, DatasetExtractor, validate_device, load_model, load_data

# uv run workdir/train.py --epochs 100 --lr 1e-4 --eval_epochs 2 --batch_size 256 --val_batch_size 256 --inter True --n_embd 128 --num_heads 4 --drop 0.4 --hidden 128
class Logger:
    def __init__(self, interactive=True):
        self.interactive = interactive
        self.logger = logging.getLogger("train")
        self.logger.setLevel(logging.INFO)
        dt_string = datetime.now().strftime("%H:%M")
        logging.basicConfig(filename=logdir/f"train_log{dt_string}.log", format="%(message)s")

    def add_logs(self, sum_wr: SummaryWriter, epoch, loss, name):
        sum_wr.add_scalar(f"{name}/loss", loss, epoch)

    def log(self, msg):
        if self.interactive:
            print(msg)
        self.logger.info(msg)

@dataclass
class DynamicStats:
    best_loss: float = np.inf
    best_path: str = None
    train_loss: dict = None
    val_loss: dict = None

@dataclass
class Trainer:
    model: nn.Module 
    device: torch.device
    batch_size: int
    lr: float
    weight_decay: float
    lr_factor: float = None
    lr_patience: float = None
    eta_min: float = None
    train_dataset: Crystals
    epochs: int
    logger: Logger
    loss_fn: CombLoss
    loss_name: str = None
    eval_epochs: int = 10
    epoch: int = 0
    val_dataset: Crystals = None
    val_batch_size: int = 256
    num_workers: int = 4
    interactive: bool = True
    ir_priority: float = 1.0
    scale_priority: float = 1.0

    def __post_init__(self):
        self.model = self.model.to(self.device)
        n_params = f"{model.n_params()/1e6:.2f}"
        model_name = str(self.model)
        self.logger.log(f"{model_name} {n_params}M")
        self.model_name = model_name + f"_{n_params}M"
        tb_logdir = tblogdir / self.model_name
        tb_logdir.mkdir(exist_ok=True,parents=True)
        self.sum_wr = SummaryWriter(log_dir=tb_logdir)
        self.best_stats = DynamicStats(best_loss=0.009)

        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, 
                            num_workers=self.num_workers, shuffle=True, 
                            pin_memory=True, persistent_workers=True)
        if self.val_dataset is not None:
            self.val_loader = DataLoader(self.val_dataset, batch_size=self.val_batch_size, 
                                num_workers=self.num_workers, shuffle=True, 
                                pin_memory=True, persistent_workers=True)
        self.train_val_loader = DataLoader(self.train_dataset, batch_size=self.val_batch_size, 
                                num_workers=self.num_workers, shuffle=True, 
                                pin_memory=True, persistent_workers=True)
        
        self.criterion = self.loss_fn
        self.scale_criterion = L1Loss()
        self.val_criterion = loss_fn_dict["kldiv"]
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, 
                                    weight_decay=self.weight_decay)
        # self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, factor=self.lr_factor, 
        #                                                             patience=self.lr_patience)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR( self.optimizer, T_max=self.epochs, 
                                                                    eta_min=self.eta_min)
        if self.loss_name is not None:
            self.models_dir = models_dir / f'{model_name}_{self.loss_name}'
        else:
            self.models_dir = models_dir / f'{model_name}'
        self.models_dir.mkdir(parents=True,exist_ok=True)
        
        conf = self.models_dir / f"{n_params}M.config"
        conf.write_text(json.dumps(self.model.get_args(), indent=1))
        self.ir_flag = hasattr(self.model, "ir_head")
        self.logger.log("using ir spectra")

    def train(self):
        if self.interactive:
            pbar = tqdm(np.arange(self.epoch+1,self.epoch+self.epochs+1), mininterval=1.0)
        else:
            pbar = np.arange(self.epoch+1,self.epoch+self.epochs+1)
        try:

            for epoch in pbar:
                self.model.train()
                self.epoch = epoch
                running_loss = 0
                for batch in self.train_loader:
                    batch = batch.to(self.device, non_blocking=True)
                    pred = self.model(batch, ir_flag=self.ir_flag)
                    raman, ram_scale = pred["raman"]
                    if self.ir_flag:
                        ir, ir_scale = pred["ir"]
                        if ir.shape[0] != 0:
                            ir_loss = self.criterion(ir, batch.ir_y[batch.has_ir]) * self.ir_priority
                            if self.scale_priority > 0:
                                ir_loss += self.scale_criterion(ir_scale.view(-1), batch.ir_fact[batch.has_ir]) * self.ir_priority * self.scale_priority
                    
                    loss = self.criterion(raman, batch.y)
                
                    running_loss += loss.item() * batch.y.shape[0]

                    if self.scale_priority > 0 and ram_scale is not None:
                            loss += self.scale_criterion(ram_scale.view(-1), batch.ram_fact) * self.scale_priority
                    
                    if self.ir_flag:
                        if ir.shape[0] != 0:
                            loss += ir_loss
                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()
                running_loss /= len(self.train_loader.sampler)
                
                torch.save(self.model.state_dict(), self.models_dir / f"{self.model_name}_latest.pt")
                if self.interactive:
                    pbar.set_description(f"lr: {self.scheduler.get_last_lr()[0]}")
                if self.epoch % self.eval_epochs == 0:
                    self.full_validate()
                    self.scheduler.step(self.val_loss)
                
        except KeyboardInterrupt:
            self.logger.log(f"\ninterrupted on epoch {self.epoch}")
            self.logger.log(f"last train loss: {self.train_loss:.5f}")
    
    def full_validate(self):

        logger.log(f"epoch {self.epoch}")
        self.train_loss = self.validate_model(self.train_val_loader)
        self.logger.add_logs(self.sum_wr, epoch=self.epoch, loss=self.train_loss, name="train")
        self.logger.log(f"train loss: {self.train_loss:.5f}")

        if self.val_dataset is not None:
            self.val_loss = self.validate_model(self.val_loader)
            self.logger.add_logs(self.sum_wr, epoch=self.epoch, loss=self.val_loss, name="val")
            self.logger.log(f"validation loss: {self.val_loss:.5f}")

            self.update_best()

    def update_best(self):
        best_loss = self.best_stats.best_loss
        if self.val_loss < best_loss or np.allclose(self.val_loss,best_loss) and self.epoch > 0:

            self.best_stats.best_path = f"{self.model_name}_{self.val_loss:.4f}_best.pt"
            torch.save(self.model.state_dict(), self.models_dir / self.best_stats.best_path)
            self.logger.log(f"saving model - {self.best_stats.best_path}")

            self.best_stats.best_loss = self.val_loss
            self.best_stats.train_loss = self.train_loss
            self.best_stats.val_loss = self.val_loss

    def validate_model(self, loader):
        self.model.eval()
        loss = 0
        with torch.inference_mode():
            for batch in loader:
                batch = batch.to(self.device, non_blocking=True)
                pred = self.model(batch)
                raman, ram_scale = pred["raman"]
                loss += self.val_criterion(raman, batch.y).item() * batch.y.shape[0]
            loss /= len(loader.sampler)
        return loss


if __name__ == "__main__":
    start = time()

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", nargs="?", type=int)
    parser.add_argument("--epoch", nargs="?", default=0, type=int)
    parser.add_argument("--weight_decay", nargs="?", default=0.001, type=float)
    parser.add_argument("--lr", nargs="?", default=1e-4, type=float)
    parser.add_argument("--eval_epochs", nargs="?", default=10, type=int)
    parser.add_argument("--batch_size", nargs="?", default=128, type=int)
    parser.add_argument("--val_batch_size", nargs="?", default=256, type=int)
    parser.add_argument("--loss_fn", nargs="*", default=["mse","cossim"], type=str)
    parser.add_argument("--lr_factor", nargs="?", default=0.3, type=float)
    parser.add_argument("--lr_patience", nargs="?", default=5, type=int)
    parser.add_argument("--eta_min", nargs="?", default=1e-6, type=float)
    # parser.add_argument("--lr_decayer", nargs="?", default=None, type=list[str])
    parser.add_argument("--device", nargs="?")

    parser.add_argument("--model_name", nargs="?")
    parser.add_argument("--model_version", nargs="?")
    parser.add_argument("--dataset", nargs="?", default="v8.pt")
    parser.add_argument("--inter", nargs="?", default="True")

    parser.add_argument("--n_embd", nargs="?", default=64, type=int)
    parser.add_argument("--num_heads", nargs="?", default=4, type=int)
    parser.add_argument("--neighbor_emb", nargs="?", default=True, type=bool)
    parser.add_argument("--hidden", nargs="?", default=128, type=int)
    parser.add_argument("--drop", nargs="?", default=0.3, type=float)
    parser.add_argument("--num_layers", nargs="?", default=4, type=int)

    parser.add_argument("--scale_priority", nargs="?", default=0.0, type=float)
    parser.add_argument("--ir_priority", nargs="?", default=1.0, type=float)
    

    args = parser.parse_args()
    inter = args.inter == "True"
    logger = Logger(interactive=inter)
    device = validate_device(args.device)
    logger.log(f"using {device}")
    model_args = {
        "embedding_dimension": args.n_embd,
        "attn_activation": "silu",
        "num_heads": args.num_heads,
        "neighbor_embedding": args.neighbor_emb,
        "hidden_size": args.hidden,
        "dropout": args.drop,
        "num_layers": args.num_layers
    }

    model_name = args.model_name
    losses = args.loss_fn
    if len(losses) == 1:
        loss_fn = loss_fn_dict["".join(losses[0])]
    else:
        loss_fn = CombLoss(*[(1, loss_fn_dict[loss]) for loss in losses])
    loss_name = str(loss_fn)

    if args.model_version is not None:
        model = models_dict[args.model_version](model_args).to(device)
    elif model_name is not None:
        version = VersionFromName(model_name,loss_name=loss_name)
        model = load_model(model_version=version,model_name=model_name, args=model_args)
    else:
        model = MDNet2(model_args).to(device)
    dataset_path = data_path / args.dataset

    extractor = DatasetExtractor(DatasetContext(
        dataset=Crystals(dataset_path, [514, 532, 780, 785]),
        test_size=0.3,
        inference=False,
        seed=0
    ))
    train_dataset, val_dataset, test_dataset = load_data(extractor)

    trainer = Trainer(
        model=model, 
        device=device, 
        batch_size=args.batch_size, 
        lr=args.lr, 
        weight_decay=args.weight_decay,
        lr_factor=args.lr_factor, 
        lr_patience=args.lr_patience,
        eta_min=args.eta_min,
        train_dataset=train_dataset, 
        epochs=args.epochs,
        logger=logger,
        loss_fn=loss_fn,
        loss_name=loss_name,
        eval_epochs=args.eval_epochs,
        epoch=args.epoch, 
        val_dataset=val_dataset,
        val_batch_size=args.val_batch_size,
        num_workers=4,
        interactive=inter,
        ir_priority=args.ir_priority,
        scale_priority=args.scale_priority
        )
    end = time()
    logger.log(f"elapsed time - {end-start:.1f}s. starting training...")
    trainer.train()

    stats = trainer.best_stats
    logger.log(f"best model is {stats.best_path}")

    val_loss = stats.val_loss
    train_loss = stats.train_loss

    logger.log(f"\t\ttrain\t\t|\tvalidation\n"
        f"kl_loss\t\t{train_loss:.4f}\t|\t{val_loss:.4f}\n")
