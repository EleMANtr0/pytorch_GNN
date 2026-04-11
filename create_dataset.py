import codecs
from collections import Counter
import warnings

import numpy as np
import pandas as pd
import torch
import torch_geometric as pyg
from tqdm import tqdm

from src.data import load_crystal_structures, load_raman_spectra


warnings.filterwarnings("ignore")

def filter_graphs(data_lst, wl_lst):
    valid = []
    validw = []
    removed = 0
    for i, d in enumerate(data_lst):
        if d.edge_index is not None and d.edge_index.size(1) != 0:
            valid.append(data_lst[i])
            validw.append(wl_lst[i])
            continue
        else:
            removed += 1
        if hasattr(d, "dist") and d.dist is not None:
            valid.append(data_lst[i])
            validw.append(wl_lst[i])
            continue
        else:
            removed += 1
    return valid, validw, removed


cif_mineral_names = []
cif_graphs = []
big_cif_file_path = "data/raw/cifdata_10.txt"
temp_file_path = "data/raw/tempcif.txt"
cur_mineral_name = ""
cur_cif_lines = []
i = 0

with open(big_cif_file_path, "r", errors="ignore") as f:
    for line in tqdm(f, mininterval=1.0, leave=False):
        if line.startswith("_chemical_formula_sum ''"):
            continue
        if line.startswith("_chemical_name_mineral"):
            cur_mineral_name = line.split("'")[1]
        if line.startswith("_amcsd_formula_title "):
            cur_mineral_name = line.split("'")[1]
        if line.startswith("END"):
            i += 1
            if i <= 0:
                pass
            else:
                try:
                    with codecs.open(temp_file_path, "w", "utf-8") as out:
                        out.write("".join(cur_cif_lines))
                    _, G = load_crystal_structures.load_single_crystal_structure(
                        temp_file_path,
                        min_distance_for_edge=1e-8,
                        max_distance_for_edge=8.0,
                        verbose=False,
                    )
                    cif_graphs.append(G)
                    cif_mineral_names.append(cur_mineral_name)
                except Exception:
                    pass
            cur_cif_lines = []
        else:
            cur_cif_lines.append(line)
max_folds = max([len(g.folds) for g in cif_graphs])

model_wavenumber_values = np.load("data/processed/wavenumber_vals_v3.npy")
ir_wavenumbers = np.linspace(370, 4000, 266)
ir_dict_id, ir_dict_name = load_raman_spectra.load_ir_data("data/raw/ir/", ir_wavenumbers, zero_pad=False)
print(f"found {len(ir_dict_id.values())} samples of ir spectra")
new_graphs_pool = {}

for i in [514, 532, 780, 785]:
    has_ir = 0
    no_ir = 0
    print("current wavelength: ", i)
    (
        raman_file_paths,
        raman_ids,
        raman_mineral_names,
        raman_spectra,
        raman_wavelengths,
        cond_vecs,
        max_intensity
    ) = load_raman_spectra.load_raman_data(
        model_wavenumber_values, raman_data_directory_path="data/raw/raman/", wavelength=i, zero_pad=False
    )
    print(
        len(raman_mineral_names),
        "Raman spectra loaded, each of length",
        len(raman_spectra[0]),
    )

    all_minerals = sorted(list(set(cif_mineral_names + raman_mineral_names)))
    cif_counter = Counter(cif_mineral_names)
    raman_counter = Counter(raman_mineral_names)

    minerals_for_dataset = [
        m for m in all_minerals if cif_counter[m] >= 1 and raman_counter[m] >= 1
    ]
    for skip in ["Diamond", "Sulphur", "Silicon"]:
        if skip in minerals_for_dataset:
            minerals_for_dataset.remove(skip)
        else:
            print("no", skip.lower(), "at", i)

    print(len(minerals_for_dataset), "/", len(all_minerals))

    temp_data_list = []
    temp_wl_list = []
    wl_val = round(i / 100.0, 3)

    for mineral in minerals_for_dataset:
        cur_mineral_raman_indices = [
            idx for idx, x in enumerate(raman_mineral_names) if x == mineral
        ]
        cur_mineral_graph_indices = [
            idx for idx, x in enumerate(cif_mineral_names) if x == mineral
        ]

        num_cifs = len(cur_mineral_graph_indices)
        for local_idx, i_raman in enumerate(cur_mineral_raman_indices):
            i_graph = cur_mineral_graph_indices[local_idx % num_cifs]
            
            cur_graph = pyg.utils.convert.from_networkx(cif_graphs[i_graph])
            cur_graph["y"] = raman_spectra[i_raman]
            cur_graph["mineral"] = mineral
            cur_graph["cond_vec"] = cond_vecs[i_raman]
            cur_graph["ram_fact"] = torch.tensor([max_intensity[i_raman]], dtype=torch.float32)
            cur_graph["cell_params"] = torch.tensor(cif_graphs[i_graph].cell_params, dtype=torch.float32).view(1, -1)
            cur_graph["folds"] = torch.tensor(
                [0] * (max_folds - len(cif_graphs[i_graph].folds))
                + cif_graphs[i_graph].folds,
                dtype=torch.long,
            ).view(1, -1)

            base_r_id = raman_ids[i_raman]
            if base_r_id and base_r_id in ir_dict_id:
                cur_graph["ir_y"] = ir_dict_id[base_r_id][0]
                cur_graph["ir_fact"] = torch.tensor([ir_dict_id[base_r_id][1]], dtype=torch.float32)
                cur_graph["has_ir"] = torch.tensor([1], dtype=torch.bool)
                has_ir += 1
            elif mineral in ir_dict_name:
                cur_graph["ir_y"] = ir_dict_name[mineral][0]
                cur_graph["ir_fact"] = torch.tensor([ir_dict_name[mineral][1]], dtype=torch.float32)
                cur_graph["has_ir"] = torch.tensor([1], dtype=torch.bool)
                has_ir += 1
            else:
                cur_graph["ir_y"] = np.zeros(266)
                cur_graph["ir_fact"] = torch.tensor([0.0], dtype=torch.float32)
                cur_graph["has_ir"] = torch.tensor([0], dtype=torch.bool)
                no_ir += 1

            temp_data_list.append(cur_graph)
            temp_wl_list.append(wl_val)

    valid_graphs, valid_wls, removed = filter_graphs(temp_data_list, temp_wl_list) 
    print(f"assigned {has_ir} ir spectras. {no_ir} are missing")
    print(f"left: {len(valid_graphs)} wl: {len(valid_wls)}, removed: {removed}")

    for g, w in zip(valid_graphs, valid_wls):
        key = (g["mineral"], w)
        if key not in new_graphs_pool:
            new_graphs_pool[key] = []
        new_graphs_pool[key].append(g)

data_list = []
wavelengths = []

for (mineral, wl), graphs in new_graphs_pool.items():
    for graph in graphs:
        graph.wl = torch.tensor([wl], dtype=torch.float32)
        data_list.append(graph)
        wavelengths.append(wl)

print((pd.DataFrame(wavelengths)).value_counts())

data, slices, _ = pyg.data.collate.collate(
    data_list[0].__class__,
    data_list=data_list,
    increment=False,
    add_batch=True,
)

data.x = data.x.type(torch.FloatTensor)
data.wl = torch.tensor(wavelengths, dtype=torch.float32)
data.pos = data.pos.type(torch.FloatTensor)
data.dist = data.dist.type(torch.FloatTensor)
data.y = torch.stack([torch.Tensor(yi) for yi in data.y]).type(torch.FloatTensor)
data.cond_vec = torch.stack([torch.Tensor(cv) for cv in data.cond_vec]).type(
    torch.FloatTensor
)
data.ram_fact = torch.stack([torch.Tensor(rf) for rf in data.ram_fact]).type(
    torch.FloatTensor
)
data.cell_params = torch.stack([torch.Tensor(cp) for cp in data.cell_params]).type(
    torch.FloatTensor
)
data.folds = torch.stack([torch.Tensor(f) for f in data.folds]).type(torch.LongTensor)
data.ir_y = torch.stack([torch.Tensor(iy) for iy in data.ir_y]).type(torch.FloatTensor)
data.ir_fact = torch.stack([torch.Tensor(irf) for irf in data.ir_fact]).type(
    torch.FloatTensor
)
data.has_ir = torch.stack([torch.Tensor(hi) for hi in data.has_ir]).type(
    torch.BoolTensor
)

torch.save((data, slices), "data/processed/v8.pt")
