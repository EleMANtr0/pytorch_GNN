import torch
def drop(tens: torch.Tensor,idx):
    mask = torch.ones(tens.size(0),tens.size(1),dtype=torch.bool,device=tens.device)
    mask[:,idx] = False
    return tens[mask].view(tens.size(0),-1)