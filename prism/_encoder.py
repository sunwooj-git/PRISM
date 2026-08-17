"""
Tissue-classifying encoder (renamed from HemaCycleNet) and its inference-time
loading path. Rebuilt from PRISM_core.py's HemaCycleNet/load_consensus_model,
trimmed to what's needed to run a trained model forward -- no training code.

Note on scope: this class's raw input is NOT gene expression. It is the
32-dim scVI joint latent embedding produced by prism._scvi_embed (see that
module's docstring for why). The encoder itself is an MLP + a tissue
(blood vs BM) classification head trained on top of that embedding.
"""
import json
import os
from dataclasses import dataclass
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPEncoder(nn.Module):
    def __init__(self, d_in: int, d_latent: int, hidden: int, n_layers: int, dropout: float):
        super().__init__()
        layers = []
        d = d_in
        for _ in range(n_layers - 1):
            layers += [nn.Linear(d, hidden), nn.ReLU(inplace=True), nn.Dropout(dropout)]
            d = hidden
        layers += [nn.Linear(d, d_latent)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class AttentionPool(nn.Module):
    """Unused at inference; kept only so consensus_model.pt's state_dict loads exactly."""
    def __init__(self, d_latent: int, hidden: int = 64):
        super().__init__()
        self.scorer = nn.Sequential(nn.Linear(d_latent, hidden), nn.Tanh(), nn.Linear(hidden, 1))

    def forward(self, z, temp: float = 1.0):
        logits = self.scorer(z).squeeze(-1)
        alpha = F.softmax(logits / max(temp, 1e-6), dim=0)
        pooled = (alpha[:, None] * z).sum(dim=0)
        return pooled, alpha, logits


class PRISMEncoderNet(nn.Module):
    """
    Blood-vs-bone-marrow tissue classifier over the scVI joint latent space.
    d_latent is the encoder's own OUTPUT embedding (used downstream by the
    NMF programs and the generative flow); it is unrelated to d_in, the
    scVI latent dimensionality fed in as input (both happen to be 32).

    donor_head/leiden_head/attn_pool are training-only auxiliary heads with
    no role in inference; they're kept so state_dict shapes match exactly
    when loading consensus_model.pt.
    """
    def __init__(
        self,
        d_in: int,
        d_latent: int,
        hidden: int,
        n_layers: int,
        dropout: float,
        n_donors: int,
        use_attention_train: bool,
        n_leiden: int = 0,
    ):
        super().__init__()
        self.encoder = MLPEncoder(d_in, d_latent, hidden, n_layers, dropout)
        self.tissue_head = nn.Linear(d_latent, 2)  # blood vs BM

        self.donor_head = nn.Sequential(
            nn.Linear(d_latent, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, n_donors),
        )

        self.n_leiden = n_leiden
        if n_leiden > 0:
            self.leiden_head = nn.Sequential(
                nn.Linear(d_latent, 128),
                nn.ReLU(inplace=True),
                nn.Dropout(0.2),
                nn.Linear(128, n_leiden),
            )
        else:
            self.leiden_head = None

        self.use_attention_train = use_attention_train
        self.attn_pool = AttentionPool(d_latent) if use_attention_train else None
        self.sample_tissue_head = nn.Linear(d_latent, 2) if use_attention_train else None

    def forward_cell(self, x):
        z = self.encoder(x)
        tissue_logits = self.tissue_head(z)
        return z, tissue_logits


def marrowness_logit_from_tissue_logits(tissue_logits: np.ndarray) -> np.ndarray:
    """marrowness logit = logit_BM - logit_blood."""
    return (tissue_logits[:, 1] - tissue_logits[:, 0]).astype(np.float32)


def apply_z_calibration(m: np.ndarray, calib: Dict[str, float]) -> np.ndarray:
    m = np.asarray(m, dtype=np.float64)
    return ((m - float(calib["mu"])) / float(calib["sd"])).astype(np.float32)


def z_to_percentile(m_z: np.ndarray, ref_blood_z: np.ndarray) -> np.ndarray:
    """Percentile of m_z relative to the trained blood cohort's z distribution."""
    ref_sorted = np.sort(np.asarray(ref_blood_z, dtype=np.float64))
    m_z = np.asarray(m_z, dtype=np.float64)
    pctiles = np.searchsorted(ref_sorted, m_z, side="right") / len(ref_sorted) * 100.0
    return pctiles.astype(np.float32)


@torch.no_grad()
def encode_cells(model: PRISMEncoderNet, X: np.ndarray, device: str, bs: int = 8192):
    """X: [N, d_in] scVI-latent input (see prism._scvi_embed). Returns (tissue_logits, Z)."""
    model.eval()
    logits_all, z_all = [], []
    for i in range(0, len(X), bs):
        x = torch.from_numpy(X[i:i + bs]).float().to(device)
        z, tlog = model.forward_cell(x)
        logits_all.append(tlog.cpu().numpy())
        z_all.append(z.cpu().numpy())
    return np.concatenate(logits_all, 0), np.concatenate(z_all, 0)


@dataclass
class EncoderBundle:
    model: PRISMEncoderNet
    arch: Dict
    summary: Dict
    mcal: Dict[str, float]              # marrowness z-calibration (train-blood mu/sd)
    blood_z_ref: np.ndarray             # trained blood cohort's calibrated marrow_z (for percentile lookup)
    device: str


def load_encoder(artifacts_dir: str, device: str = "cpu") -> EncoderBundle:
    with open(os.path.join(artifacts_dir, "consensus_arch.json")) as f:
        arch = json.load(f)

    model = PRISMEncoderNet(
        d_in=arch["d_in"],
        d_latent=arch["d_latent"],
        hidden=arch["hidden"],
        n_layers=arch["n_layers"],
        dropout=arch["dropout"],
        n_donors=arch["n_donors"],
        use_attention_train=arch["use_attention_train"],
        n_leiden=arch.get("n_leiden", 0),
    ).to(device)
    state = torch.load(os.path.join(artifacts_dir, "consensus_model.pt"), map_location=device)
    model.load_state_dict(state)
    model.eval()

    with open(os.path.join(artifacts_dir, "consensus_summary.json")) as f:
        summary = json.load(f)
    mcal = {"mu": summary["calibration"]["mu"], "sd": summary["calibration"]["sd"]}

    # Per-cell trained-blood calibrated marrowness z-scores, used only for
    # empirical percentile lookup (z_to_percentile) -- extracted ahead of
    # time from consensus_blood_reference.npz's marrow_z_blood field so the
    # package doesn't need that file's much larger Z/P/obs_names arrays.
    blood_z_ref = np.load(os.path.join(artifacts_dir, "blood_marrowz_ref.npy")).astype(np.float64)

    return EncoderBundle(
        model=model, arch=arch, summary=summary, mcal=mcal,
        blood_z_ref=blood_z_ref, device=device,
    )
