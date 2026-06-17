from .models_md import MDNet, MDNet1, MDNet2, MDNet3, MDNetSide
from .models_painn import PaiNN_raman0, PaiNN_raman2, PaiNN_raman3
# from .models_matformer import Matformer


models_dict = {
    "MDNet": MDNet,
    "MDNet1": MDNet1,
    "MDNet2": MDNet2,
    "MDNet3": MDNet3,
    "MDNetSide": MDNetSide,
    "painn": PaiNN_raman0,
    "painn2": PaiNN_raman2,
    "painn3": PaiNN_raman3,
    # "Matformer": Matformer
}
