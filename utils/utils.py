from dataclasses import dataclass
from typing import Protocol, Any
from pathlib import Path
import json

import torch
import torch.nn as nn

from config import models_dir
from src.models import models_dict
from src.data.dataset import Crystals


@dataclass
class DatasetContext:
    dataset: Crystals
    test_size: int
    inference: bool = False
    seed: int = 0

class Converter(Protocol):
    def convert(self, preds: list[str] | None) -> None:
        ...

class Version(Protocol):
    def get_name(self) -> str:
        ...

class Extractor(Protocol):
    def extract(self) -> Any:
        ...

class Inference:
    def __init__(self, model: nn.ModuleDict):
        self.model = model

    def predict(self, x:torch.Tensor):
        self.model.eval()
        return self.preds

class VersionFromName:
    def __init__(self, model_name, loss_name):
        self.model_version = "".join(model_name.split("_")[0])
        self.num_params = model_name.split("_")[1]
        self.loss_name = loss_name

    def get_name(self):
        return self.model_version

    def get_loss(self):
        return self.loss_name
    
    def get_params(self):
        return self.num_params


def load_model(model_version: VersionFromName, model_name: str, device: torch.device = "cuda", args=None):
    model_n = model_version.get_name()
    loss_name = model_version.get_loss()
    num_params = model_version.get_params()
    model_path = models_dir / f"{model_n}_{loss_name}/{model_name}"
    if args is None:
        args = json.loads((model_path.parent / f"{num_params}.config").read_text())
    model = models_dict[model_n](args)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    return model

class DatasetExtractor:
    def __init__(self, context: DatasetContext):
        self.context = context

    def extract(self):
        train_data, val_data, test_data = self.context.dataset.split(self.context.test_size, self.context.seed)
        if self.context.inference:
            return test_data
        return train_data, val_data, test_data
    
def load_data(extractor: Extractor):
    return extractor.extract()

def validate_device(device):
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            print("cuda is not available")
            device = torch.device("cpu")
    else:
        device = torch.device(device)
    return device

def drop(tens: torch.Tensor, idx):
    mask = torch.ones(tens.size(0),tens.size(1),dtype=torch.bool,device=tens.device)
    mask[:,idx] = False
    return tens[mask].view(tens.size(0),-1)
