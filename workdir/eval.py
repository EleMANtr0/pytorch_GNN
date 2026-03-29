import json
import argparse
from pathlib import Path

from torch_geometric.loader import DataLoader


from src.dataset import Crystals
from src.loss import CombLoss
# from src.utils.util import drop  # for dropping features just to test
from config import loss_fn_dict, cur_dir, data_path
from utils import VersionFromName, DatasetContext, DatasetExtractor, validate_device, load_model, load_data
from src.plotting import save_all
from src.metrics import kldiv


parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", nargs="?", default=128, type=int)
parser.add_argument("--loss_fn", nargs="*", default=["mse","cossim"], type=str)
parser.add_argument("--device", nargs="?")
parser.add_argument("--model_name", nargs="?")
parser.add_argument("--dataset", nargs="?", default="v7.pt")
# parser.add_argument("--n_embd", nargs="?", default=64, type=int)
# parser.add_argument("--num_heads", nargs="?", default=4, type=int)
# parser.add_argument("--neighbor_emb", nargs="?", default=True, type=bool)
# parser.add_argument("--hidden", nargs="?", default=128, type=int)
# parser.add_argument("--drop", nargs="?", default=0.3, type=float)
# parser.add_argument("--num_layers", nargs="?", default=4, type=int)

args = parser.parse_args()

# model_args = {
#     "embedding_dimension": args.n_embd,
#     "attn_activation": "silu",
#     "num_heads": args.num_heads,
#     "neighbor_embedding": args.neighbor_emb,
#     "hidden_size": args.hidden,
#     "dropout": args.drop,
#     "num_layers": args.num_layers
# }

device = validate_device(args.device)
model_name = args.model_name
losses = args.loss_fn
if len(losses) == 1:
    loss_fn = loss_fn_dict["".join(losses[0])]
else:
    loss_fn = CombLoss(*[(1, loss_fn_dict[loss]) for loss in losses])
loss_name = str(loss_fn)

version = VersionFromName(model_name,loss_name=loss_name)
model = load_model(model_version=version,model_name=model_name)
dataset_path = data_path / args.dataset

context = DatasetContext(
    dataset=Crystals(dataset_path),
    test_size=0.3,
    inference=False,
    seed=0
)
extractor = DatasetExtractor(context)
train_dataset, val_dataset, test_dataset = load_data(extractor)
val_dataloader = DataLoader(val_dataset, batch_size=128)

predict_dir_val = cur_dir / f"predict/{args.model_name}_{loss_name}/val"
predict_dir_val.mkdir(parents=True,exist_ok=True)
predict_dir_test = cur_dir / f"predict/{model_name}_{loss_name}/test"
predict_dir_test.mkdir(parents=True,exist_ok=True)
predict_dirs_test = {}
predict_dirs_val = {}
for i in context.dataset.wl_list:
    path = Path(str(i))
    predict_dirs_val[i] = predict_dir_val/str(i)
    predict_dirs_test[i] = predict_dir_test/str(i)
    predict_dirs_val[i].mkdir(parents=True,exist_ok=True)
    predict_dirs_test[i].mkdir(parents=True,exist_ok=True)

kl_div = kldiv(model, device, val_dataloader, 514)[0]
Path("cur_dir/predict/val/kldiv_514.txt").write_text(kl_div)
print(kl_div)


save_all(model,device,val_dataset,predict_dirs_val,verbose=False)
