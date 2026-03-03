from typing import Callable, Dict, Optional, List
import torch
import torch.nn as nn
import torch.nn.functional as F
import schnetpack.properties as properties
import schnetpack.nn as snn
from schnetpack.representation.painn import PaiNNInteraction, PaiNNMixing



class PaiNNMixing_WL(nn.Module):
    """PaiNN mixing block with FiLM conditioning from wavelength."""

    def __init__(self, n_atom_basis: int, activation: Callable, epsilon: float = 1e-8):
        # FIX: ensure we call the right superclass
        super().__init__()
        self.n_atom_basis = n_atom_basis

        self.intraatomic_context_net = nn.Sequential(
            snn.Dense(2 * n_atom_basis, n_atom_basis, activation=activation),
            snn.Dense(n_atom_basis, 3 * n_atom_basis, activation=None),
        )
        self.mu_channel_mix = snn.Dense(
            n_atom_basis, 2 * n_atom_basis, activation=None, bias=False
        )
        self.epsilon = epsilon

    def forward(self, q: torch.Tensor, mu: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor):
        """Compute intraatomic mixing with FiLM conditioning."""
        mu_mix = self.mu_channel_mix(mu)
        mu_V, mu_W = torch.split(mu_mix, self.n_atom_basis, dim=-1)
        mu_Vn = torch.sqrt(torch.sum(mu_V**2, dim=-2, keepdim=True) + self.epsilon)

        ctx = torch.cat([q, mu_Vn], dim=-1)
        x = self.intraatomic_context_net(ctx)

        dq_intra, dmu_intra, dqmu_intra = torch.split(x, self.n_atom_basis, dim=-1)
        dmu_intra = dmu_intra * mu_W
        dqmu_intra = dqmu_intra * torch.sum(mu_V * mu_W, dim=1, keepdim=True)

        q = q + dq_intra + dqmu_intra
        q = gamma * q + beta
        mu = mu + dmu_intra
        return q, mu


class PaiNN_WL(nn.Module):
    """
    PaiNN with wavelength FiLM conditioning.
    Wavelength generates scale/shift parameters for mixing blocks.
    """

    def __init__(
        self,
        n_atom_basis: int,
        n_interactions: int,
        radial_basis: nn.Module,
        cutoff_fn: Optional[Callable] = None,
        activation: Optional[Callable] = F.silu,
        shared_interactions: bool = False,
        shared_filters: bool = False,
        epsilon: float = 1e-8,
        nuclear_embedding: Optional[nn.Module] = None,
        electronic_embeddings: Optional[List] = None,
        wl_embed_dim: int = 64,
    ):
        super(PaiNN_WL, self).__init__()

        self.n_atom_basis = n_atom_basis
        self.n_interactions = n_interactions
        self.cutoff_fn = cutoff_fn
        self.cutoff = cutoff_fn.cutoff
        self.radial_basis = radial_basis

        if nuclear_embedding is None:
            nuclear_embedding = nn.Embedding(100, n_atom_basis)
        self.embedding = nuclear_embedding
        if electronic_embeddings is None:
            electronic_embeddings = []
        electronic_embeddings = nn.ModuleList(electronic_embeddings)
        self.electronic_embeddings = electronic_embeddings

        self.wl_encoder = nn.Sequential(
            nn.Linear(1, wl_embed_dim),
            nn.SiLU(),
            nn.Linear(wl_embed_dim, wl_embed_dim),
            nn.SiLU()
        )

        self.film_generators = nn.ModuleList([
            nn.Sequential(
                nn.Linear(wl_embed_dim, 2 * n_atom_basis),
                nn.SiLU(),
                nn.Linear(2 * n_atom_basis, 2 * n_atom_basis)
            )
            for _ in range(n_interactions)
        ])

        self.share_filters = shared_filters
        if shared_filters:
            self.filter_net = snn.Dense(
                self.radial_basis.n_rbf, 3 * n_atom_basis, activation=None
            )
        else:
            self.filter_net = snn.Dense(
                self.radial_basis.n_rbf,
                self.n_interactions * n_atom_basis * 3,
                activation=None,
            )

        self.interactions = snn.replicate_module(
            lambda: PaiNNInteraction(
                n_atom_basis=self.n_atom_basis, activation=activation
            ),
            self.n_interactions,
            shared_interactions,
        )
        self.mixing = snn.replicate_module(
            lambda: PaiNNMixing_WL(
                n_atom_basis=self.n_atom_basis, activation=activation, epsilon=epsilon
            ),
            self.n_interactions,
            shared_interactions,
        )

    def forward(self, inputs: Dict[str, torch.Tensor]):
        """
        Compute atomic representations/embeddings with wavelength FiLM conditioning.

        Args:
            inputs: SchNetPack dictionary of input tensors. Should include 'wl' and 'batch' keys.

        Returns:
            dict with 'scalar_representation' and 'vector_representation'
        """
        atomic_numbers = inputs[properties.Z]
        r_ij = inputs[properties.Rij]
        idx_i = inputs[properties.idx_i]
        idx_j = inputs[properties.idx_j]
        n_atoms = atomic_numbers.shape[0]

        d_ij = torch.norm(r_ij, dim=1, keepdim=True)
        dir_ij = r_ij / d_ij
        phi_ij = self.radial_basis(d_ij)
        fcut = self.cutoff_fn(d_ij)

        filters = self.filter_net(phi_ij) * fcut[..., None]
        if self.share_filters:
            filter_list = [filters] * self.n_interactions
        else:
            filter_list = torch.split(filters, 3 * self.n_atom_basis, dim=-1)

        q = self.embedding(atomic_numbers)
        for embedding in self.electronic_embeddings:
            q = q + embedding(q, inputs)

        if 'wl' in inputs:
            wl = inputs['wl']
            if wl.dim() == 0:
                wl = wl.unsqueeze(0)
            if wl.dim() == 1:
                wl = wl.unsqueeze(-1)
            wl_features = self.wl_encoder(wl)

            film_params_list = []
            for film_gen in self.film_generators:
                film_params = film_gen(wl_features)
                gamma, beta = torch.chunk(film_params, 2, dim=-1)
                gamma = gamma + 1.0
                film_params_list.append((gamma, beta))
        else:
            film_params_list = [(torch.ones(1, self.n_atom_basis, device=q.device),
                                torch.zeros(1, self.n_atom_basis, device=q.device))
                               for _ in range(self.n_interactions)]

        q = q.unsqueeze(1)

        qs = q.shape
        mu = torch.zeros((qs[0], 3, qs[2]), device=q.device)

        for i, (interaction, mixing) in enumerate(zip(self.interactions, self.mixing)):
            q, mu = interaction(q, mu, filter_list[i], dir_ij, idx_i, idx_j, n_atoms)

            gamma, beta = film_params_list[i]
            if 'batch' in inputs:
                batch_idx = inputs['batch']
                gamma_expanded = gamma[batch_idx].unsqueeze(1)
                beta_expanded = beta[batch_idx].unsqueeze(1)
            else:
                gamma_expanded = gamma.unsqueeze(0)
                beta_expanded = beta.unsqueeze(0)

            q, mu = mixing(q, mu, gamma_expanded, beta_expanded)

        q = q.squeeze(1)

        inputs["scalar_representation"] = q
        inputs["vector_representation"] = mu

        return inputs
