from matplotlib import pyplot as plt
import numpy as np
from pathlib import Path
import os
from torch_geometric.loader import DataLoader
from config import wavenumbers


def plot_one_dataset(x, title, save_dir=None, rtitle=None):
    if save_dir or save_dir=="":
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{title}{rtitle}.png"
        save_path = save_dir / filename
        fig, ax = plt.subplots()
        ax.plot(wavenumbers, x, color="#ff7f0e")
        ax.set_title(title)
        ax.set_ylabel('Intensity')
        ax.set_xlabel('Raman shift (cm^-1)')
        fig.tight_layout()
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.plot(wavenumbers, x, color="#ff7f0e")
        plt.title(title)
        plt.ylabel('Intensity')
        plt.xlabel('Raman shift (cm^-1)')
        plt.tight_layout()
        plt.show()
        plt.close()

def plot_spectra(pts_pred, pts_true=None, legend=True, title='', rtitle = '', save_dir = None, verbose = True):
    plt.figure()
    plots = []
    xs_pred = [pt[0] for pt in pts_pred]
    ys_pred = [pt[1] for pt in pts_pred]

    plots.append(plt.plot(xs_pred, ys_pred,"--", label='Predicted spectrum')[0])
    
    if pts_true is not None:
        xs_true = [pt[0] for pt in pts_true]
        ys_true = [pt[1] for pt in pts_true]
        plots.append(plt.plot(xs_true, ys_true, label='True spectrum')[0])

    if legend:
        plt.legend(handles=plots)
    if title:
        plt.title(title)
    plt.ylabel('Intensity')
    plt.xlabel('Raman shift (cm^-1)')
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        filename = f'{title}{rtitle}.png'
        save_path = os.path.join(save_dir, filename)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    if verbose:
        plt.show()
    plt.close()

def save_all(model, device, dataset, save_dir = None, is_schnet=False,verbose=True):
    model.eval()
    for i, data in enumerate(DataLoader(dataset, batch_size=1)):
        data = data.to(device)
        if is_schnet:
            pred = model(data.z, data.pos, data.batch).detach().cpu().numpy().flatten()
        else:
            pred = model(data).detach().cpu().numpy().flatten()
        true = data.y.detach().cpu().numpy().flatten()

        P = [(x, p) for (x, p) in zip(wavenumbers, pred)]
        Y = [(x, y) for (x, y) in zip(wavenumbers, true)]
        if isinstance(save_dir, Path):
            plot_spectra(P, Y, title=f"{data.mineral[0]}", rtitle=str(i),
                         save_dir=save_dir,verbose=verbose)
        elif isinstance(save_dir, dict):
            plot_spectra(P, Y, title=f"{data.mineral[0]}", rtitle=str(i),
                         save_dir=save_dir[round(data.wl.item() * 100)],verbose=verbose)
