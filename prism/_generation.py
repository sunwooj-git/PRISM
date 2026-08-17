"""
Conditional normalizing flow + gene decoder: generates donor-specific
synthetic bone-marrow-like cells conditioned on a donor's own bone
marrow-like cells' program scores and cell type.

Classes below are lifted verbatim (renaming aside -- none of this code
referenced "HemaCycle" or "spillover" in the first place) from
bm_generation_v6.py's ConditionalFlow/AffineCouplingLayer/PermutationLayer/
CelltypeGeneDecoderNB, since those are the exact architectures
flow_celltype_model.pt / gene_decoder.pt's weights were trained into.

load_generation_models here is a from-scratch replacement for
bm_generation_v6.load_generation_models: that function's gene-decoder
loading branch always instantiates the generic `GeneDecoder` class
regardless of gene_decoder_config.json's "decoder_type" -- for this
release, decoder_type is "celltype_nb" (CelltypeGeneDecoderNB), which
that function does not handle at all. Verified directly against the
shipped gene_decoder_config.json before writing this.
"""
import json
import math
import os
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class AffineCouplingLayer(nn.Module):
    def __init__(self, d_embed: int, d_condition: int, hidden: int = 128, n_layers: int = 2):
        super().__init__()
        self.d_split = d_embed // 2
        self.d_rest = d_embed - self.d_split
        inp_dim = self.d_split + d_condition
        layers = []
        for i in range(n_layers):
            layers.extend([nn.Linear(inp_dim if i == 0 else hidden, hidden), nn.GELU()])
        self.net_s = nn.Sequential(*layers, nn.Linear(hidden, self.d_rest), nn.Tanh())
        self.net_t = nn.Sequential(
            *[nn.Linear(inp_dim if i == 0 else hidden, hidden) if i % 2 == 0 else nn.GELU()
              for i in range(n_layers * 2)],
            nn.Linear(hidden, self.d_rest),
        )

    def forward(self, x, condition):
        x1, x2 = x[:, :self.d_split], x[:, self.d_split:]
        ctx = torch.cat([x1, condition], dim=-1)
        s, t = self.net_s(ctx), self.net_t(ctx)
        y2 = x2 * torch.exp(s) + t
        return torch.cat([x1, y2], dim=-1), s.sum(dim=-1)

    def inverse(self, y, condition):
        y1, y2 = y[:, :self.d_split], y[:, self.d_split:]
        ctx = torch.cat([y1, condition], dim=-1)
        s, t = self.net_s(ctx), self.net_t(ctx)
        x2 = (y2 - t) * torch.exp(-s)
        return torch.cat([y1, x2], dim=-1)


class PermutationLayer(nn.Module):
    def __init__(self, d_embed: int, seed: int = 0):
        super().__init__()
        perm = torch.randperm(d_embed, generator=torch.Generator().manual_seed(seed))
        self.register_buffer("perm", perm)
        self.register_buffer("inv_perm", torch.argsort(perm))

    def forward(self, x, condition=None):
        return x[:, self.perm], torch.zeros(x.shape[0], device=x.device)

    def inverse(self, y, condition=None):
        return y[:, self.inv_perm]


class ConditionalFlow(nn.Module):
    def __init__(self, d_embed=32, d_condition=5, n_flow_layers=8, hidden=128, n_mlp_layers=2):
        super().__init__()
        self.d_embed = d_embed
        self.d_condition = d_condition
        layers = []
        for i in range(n_flow_layers):
            layers.append(AffineCouplingLayer(d_embed, d_condition, hidden, n_mlp_layers))
            layers.append(PermutationLayer(d_embed, seed=i))
        self.flow_layers = nn.ModuleList(layers)

    def inverse(self, z, condition):
        x = z
        for layer in reversed(list(self.flow_layers)):
            x = layer.inverse(x, condition)
        return x

    @torch.no_grad()
    def generate(self, condition: torch.Tensor, n_samples: int = 1):
        self.eval()
        device = next(self.parameters()).device
        condition = condition.to(device)
        if condition.dim() == 1:
            condition = condition.unsqueeze(0).expand(n_samples, -1)
        elif condition.shape[0] == 1 and n_samples > 1:
            condition = condition.expand(n_samples, -1)
        z = torch.randn(n_samples, self.d_embed, device=device)
        return self.inverse(z, condition)


class CelltypeGeneDecoderNB(nn.Module):
    """Cell-type-aware negative-binomial decoder for raw gene counts."""
    def __init__(self, d_embed=32, n_types=8, n_genes=13000, hidden=256, n_layers=3, dropout=0.1):
        super().__init__()
        self.d_embed, self.n_types, self.n_genes = d_embed, n_types, n_genes
        d_in = d_embed + n_types
        layers = []
        for i in range(n_layers):
            inp = d_in if i == 0 else hidden
            layers.extend([nn.Linear(inp, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout)])
        self.shared = nn.Sequential(*layers)
        self.mu_logits_head = nn.Linear(hidden, n_genes)
        self.theta_head = nn.Linear(hidden, n_genes)

    def forward(self, z_embed, celltype_onehot, library_size):
        x = torch.cat([z_embed, celltype_onehot], dim=-1)
        h = self.shared(x)
        mu = F.softmax(self.mu_logits_head(h), dim=-1) * library_size + 1e-6
        theta = F.softplus(self.theta_head(h)).clamp(max=1e4) + 1e-6
        return mu, theta

    def predict_mean(self, z_embed, celltype_onehot, library_size=None):
        if library_size is None:
            library_size = torch.full((z_embed.shape[0], 1), 5000.0, device=z_embed.device)
        mu, _ = self.forward(z_embed, celltype_onehot, library_size)
        return mu

    def sample(self, z_embed, celltype_onehot, library_size=None):
        """
        Draws actual integer counts from the trained NB distribution (using
        both the mean mu and the dispersion theta the decoder was trained
        to predict via nb_loss), rather than returning the deterministic
        per-gene mean -- real scRNA-seq counts have this kind of
        overdispersed sampling noise, and predict_mean() alone produces
        near-identical values across cells of the same type/library size.
        """
        if library_size is None:
            library_size = torch.full((z_embed.shape[0], 1), 5000.0, device=z_embed.device)
        mu, theta = self.forward(z_embed, celltype_onehot, library_size)
        # torch.distributions.NegativeBinomial's mean is total_count*probs/(1-probs)
        # (verified directly -- NOT total_count*(1-probs)/probs). Solving
        # theta*p/(1-p) = mu for p gives p = mu/(theta+mu). Clamped away from
        # the [0,1) boundary: theta can be small enough relative to mu that
        # this rounds to exactly 1.0 in float32, which the distribution
        # rejects outright.
        probs = (mu / (theta + mu)).clamp(min=1e-6, max=1.0 - 1e-6)
        dist = torch.distributions.NegativeBinomial(total_count=theta, probs=probs)
        return dist.sample()


@dataclass
class GenerationBundle:
    flow: ConditionalFlow
    gene_decoder: CelltypeGeneDecoderNB
    unique_types: list
    type_to_idx: Dict[str, int]
    d_programs: int
    z_mean: np.ndarray
    z_std: np.ndarray
    gene_names: np.ndarray
    library_size_per_ct: Dict[str, float]
    lib_scaling_per_ct: Dict[str, float]
    median_library_size: float
    device: str


def load_generation_models(artifacts_dir: str, device: str = "cpu") -> GenerationBundle:
    with open(os.path.join(artifacts_dir, "generative_config.json")) as f:
        gen_config = json.load(f)
    if gen_config["model_type"] != "flow_celltype":
        raise NotImplementedError(f"Unsupported model_type={gen_config['model_type']!r}")

    flow = ConditionalFlow(
        d_embed=gen_config["d_embed"],
        d_condition=gen_config["d_condition"],
        n_flow_layers=gen_config["n_flow_layers"],
        hidden=gen_config.get("hidden", 64),
        n_mlp_layers=gen_config.get("n_mlp_layers", 2),
    ).to(device)
    flow.load_state_dict(torch.load(
        os.path.join(artifacts_dir, "flow_celltype_model.pt"), map_location=device))
    flow.eval()

    with open(os.path.join(artifacts_dir, "gene_decoder_config.json")) as f:
        gc_ = json.load(f)
    if gc_["decoder_type"] != "celltype_nb":
        raise NotImplementedError(f"Unsupported decoder_type={gc_['decoder_type']!r}")

    gene_dec = CelltypeGeneDecoderNB(
        d_embed=gc_["d_embed"], n_types=gc_["n_types"], n_genes=gc_["n_genes"],
        hidden=gc_.get("hidden", 512), n_layers=gc_.get("n_layers", 3),
    ).to(device)
    gene_dec.load_state_dict(torch.load(
        os.path.join(artifacts_dir, "gene_decoder.pt"), map_location=device))
    gene_dec.eval()

    return GenerationBundle(
        flow=flow,
        gene_decoder=gene_dec,
        unique_types=gen_config["unique_types"],
        type_to_idx=gen_config["type_to_idx"],
        d_programs=gen_config["d_programs"],
        z_mean=np.asarray(gen_config["z_mean"], dtype=np.float32),
        z_std=np.asarray(gen_config["z_std"], dtype=np.float32),
        gene_names=np.asarray(gc_["gene_names"]),
        library_size_per_ct=gc_["library_size_per_ct"],
        lib_scaling_per_ct=gc_["lib_scaling_per_ct"],
        median_library_size=gc_["median_library_size"],
        device=device,
    )


def generate_per_celltype(
    gen: GenerationBundle,
    predicted_proportions: Dict[str, float],
    program_conditioning_per_ct: Dict[str, np.ndarray],
    blood_lib_per_ct: Dict[str, float],
    n_cells: int,
    seed: int = 0,
    sample_counts: bool = True,
) -> Dict[str, np.ndarray]:
    """
    Generate n_cells total synthetic bone-marrow-like cells, split across
    cell types by predicted_proportions, each conditioned on that cell
    type's program-score vector (program_conditioning_per_ct).

    Per-cell-type library size: donor's own observed blood library size for
    that cell type (blood_lib_per_ct), scaled by the trained blood->BM
    factor (gen.lib_scaling_per_ct) -- falls back to the trained
    population median for that cell type, then the global median, if the
    donor has no blood cells of that type to measure a library size from.
    """
    torch.manual_seed(seed)
    device = gen.device

    cells_per_type: Dict[str, int] = {}
    remaining = n_cells
    for ct, frac in sorted(predicted_proportions.items(), key=lambda kv: -kv[1]):
        if ct not in gen.type_to_idx:
            continue
        n_ct = int(round(frac * n_cells))
        cells_per_type[ct] = n_ct
        remaining -= n_ct
    if remaining != 0 and cells_per_type:
        largest = max(cells_per_type, key=cells_per_type.get)
        cells_per_type[largest] += remaining

    n_types = len(gen.unique_types)
    z_mean_t = torch.from_numpy(gen.z_mean).to(device)
    z_std_t = torch.from_numpy(gen.z_std).to(device)

    all_Z, all_X, all_types = [], [], []
    with torch.no_grad():
        for ct, n_ct in cells_per_type.items():
            if n_ct <= 0:
                continue

            P_cond = np.asarray(program_conditioning_per_ct[ct], dtype=np.float32)
            ct_onehot_flow = np.zeros(n_types, dtype=np.float32)
            ct_onehot_flow[gen.type_to_idx[ct]] = 1.0
            condition = np.concatenate([
                np.broadcast_to(P_cond.reshape(1, -1), (n_ct, len(P_cond))),
                np.broadcast_to(ct_onehot_flow.reshape(1, -1), (n_ct, n_types)),
            ], axis=1)
            condition_t = torch.from_numpy(condition).to(device)

            Z_ct = gen.flow.generate(condition_t, n_samples=n_ct)
            Z_ct = Z_ct * z_std_t + z_mean_t

            ct_onehot_dec = torch.zeros(n_ct, len(gen.unique_types), device=device)
            ct_onehot_dec[:, gen.type_to_idx[ct]] = 1.0

            scaling = gen.lib_scaling_per_ct.get(ct)
            blood_lib = blood_lib_per_ct.get(ct)
            if blood_lib is not None and scaling is not None:
                lib_value = blood_lib * scaling
            else:
                lib_value = gen.library_size_per_ct.get(ct, gen.median_library_size)
            lib = torch.full((n_ct, 1), float(lib_value), device=device)

            if sample_counts:
                X_ct = gen.gene_decoder.sample(Z_ct, ct_onehot_dec, lib)
            else:
                X_ct = gen.gene_decoder.predict_mean(Z_ct, ct_onehot_dec, lib)

            all_Z.append(Z_ct.cpu().numpy())
            all_X.append(X_ct.cpu().numpy())
            all_types.extend([ct] * n_ct)

    return {
        "Z": np.concatenate(all_Z, axis=0),
        "X": np.concatenate(all_X, axis=0),
        "cell_type": np.array(all_types),
        "gene_names": gen.gene_names,
        "cells_per_type": cells_per_type,
    }
