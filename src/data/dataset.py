from collections import defaultdict
from random import shuffle

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch_geometric.data import InMemoryDataset
from torch_cluster import radius_graph


class Crystals(InMemoryDataset):
    def __init__(
        self,
        root: str,
        wl_list=None,
        transform = None,
        pre_transform = None,
        pre_filter = None,
        has_raman = True
    ):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(root, weights_only=False)
        
        if not has_raman:
            valid_indices = torch.where(self._data.has_ir.view(-1).bool())[0].tolist()
            valid_graphs = [self.get(i) for i in valid_indices]
            self.data, self.slices = self.collate(valid_graphs)

        if wl_list is None:
            self.wl_list = [514, 532, 780, 785]
        else:
            self.wl_list = list(np.array([wl_list]).flatten())

    def split(self, test_size=0.3, seed=0):
        all_dict = defaultdict(list)
        if hasattr(self, "wl"):
            for i in range(len(self.wl)):
                if (wl := round(100 * self.wl[i].item())) in self.wl_list:
                    all_dict[wl].append(self[i])
            traind = {}
            val_testd = {}
            vald = {}
            testd = {}
            # print({k: len(v) for k, v in all_dict.items()})
            for i in self.wl_list:
                traind[i], val_testd[i] = train_test_split(
                    all_dict[i], test_size=test_size, random_state=seed
                )
                vald[i], testd[i] = train_test_split(
                    val_testd[i], test_size=0.5, random_state=seed
                )
            train = []
            val = []
            test = []
            for i in self.wl_list:
                train.extend(traind[i])
                val.extend(vald[i])
                test.extend(testd[i])
            shuffle(train)
            # shuffle(val)
            # shuffle (train)
            return train, val, test
        train, val_test = train_test_split(self, test_size=0.3, random_state=seed)
        val, test = train_test_split(val_test, test_size=0.5, random_state=seed)
        return train, val, test

    @property
    def processed_file_names(self) -> str:
        return ["data/processed/data_v5.pt"]


def pyg_batch_to_schnetpack(data, cutoff=10.0):
    batch = {
        "_atomic_numbers": data.z.long(),
        "_positions": data.pos,
    }
    if hasattr(data, "batch"):
        batch["n_atoms"] = torch.bincount(data.batch)
    else:
        batch["n_atoms"] = torch.tensor([data.z.size(0)])
    if hasattr(data, "x") and data.x is not None:
        batch["_atom_features"] = data.x.float()

    edge_index = radius_graph(
        data.pos, r=cutoff, batch=data.batch if hasattr(data, "batch") else None
    )

    idx_i, idx_j = edge_index
    pos_i = data.pos[idx_i]
    pos_j = data.pos[idx_j]

    offsets = pos_j - pos_i
    distances = torch.norm(offsets, dim=1)
    mask = distances > 1e-8
    batch["_idx_i"] = idx_i[mask]
    batch["_idx_j"] = idx_j[mask]
    batch["_Rij"] = offsets[mask]

    return batch