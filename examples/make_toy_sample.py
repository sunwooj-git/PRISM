"""
Generates examples/toy_sample.h5ad: a small, synthetic single-donor blood
AnnData for the quickstart example.

v3: generate from a real reference donor's per-cell-type statistics
(--reference), when one is available, instead of hand-picked marker genes.
IMPORTANT: the reference must be real BLOOD data, matching what PRISM
actually accepts as input -- an earlier version of this file was built
from a real bone marrow donor by mistake, which produces a fundamentally
different data distribution (~50-90% of cells reading as bone-marrow-like)
than real blood ever does. True bone marrow-like signal is supposed to be
RARE within blood -- that's the entire premise of what PRISM detects, not
an artifact to tune away. For each cell type in the reference (with enough
cells to be reliable), this fits an independent per-gene negative-binomial
model directly from that type's real cells -- mean expression fraction,
and a dispersion (theta = mean^2/(var-mean)) estimated from the real
per-gene variance, not a single shared noise level applied uniformly to
every gene. A synthetic cell of that type draws its library size from the
type's real total_counts values, then samples every gene independently as
NB(mean = library_size * mean_fraction[gene], dispersion = theta[gene]).
Genes that are noisy/bursty in the real data get correspondingly noisy
synthetic counts; stable genes stay stable. Cell-type proportions in the
output match the reference donor's own real proportions (restricted to
well-represented types), so a toy run and a real run of the same donor
should produce comparable Output 1 compositions.

Verified against the correct held-out benchmark (the same real donor's own
blood data, run through the exact same pipeline): 61/1500 (4.1%) of toy
cells came back bone-marrow-like vs. 34/6948 (0.5%) for the real donor's
own blood -- both now correctly in the "rare tail event" regime, a
qualitative fix from the bone-marrow-templated version's 48-86%, though a
real ~8x gap in the exact rate remains (plausibly because independent
per-gene sampling still can't reproduce a real cell's *correlated*
cross-gene noise -- a real cell's technical/biological noise moves many
genes together, not independently -- meaning synthetic cells are still
somewhat more "prototypical" of their assigned type than genuinely noisy
real ones). Composition and program-score alignment were both close on
this corrected benchmark.

IMPORTANT: this only ever reads a reference file locally to compute
aggregate summaries (a per-gene mean/dispersion per cell type + a same-
distribution sample of library sizes) -- it never copies individual real
cell records into the output, and no real per-donor file is bundled with
this repo. Point --reference at your own local copy of real data; the
committed toy_sample.h5ad here was generated this way from a real donor
but the source file itself is not part of the repo.

Without --reference, falls back to a hand-built latent-factor model: each
of the 11 cell types gets a "module" of ~150 co-expressed genes (a few real
markers at strong loading, plus many more at weaker/varying loading, all
driven by one shared per-cell activity value so they move together) --
this has real multi-gene covariance too, just fabricated rather than drawn
from a real donor. Useful when you don't have reference data on hand.
"""
import argparse
import os
from typing import Dict, Optional

import anndata as ad
import numpy as np
import pandas as pd
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS_DIR = os.path.join(HERE, "..", "weights")

MIN_REFERENCE_CELLS_PER_TYPE = 10  # below this, a pseudo-bulk profile is too noisy to trust

N_EXTRA_GENES_PER_MODULE = 150  # fallback (no --reference) generator only
CORE_MARKER_LOADING = 5.0
OWN_MODULE_ACTIVITY_MEAN = 4.0
OWN_MODULE_ACTIVITY_SD = 0.5
CROSSTALK_ACTIVITY_SD = 0.3

MARKER_GENES = {
    "b_cell": ["MS4A1", "CD79A", "CD79B", "PAX5", "CD19"],
    "t_cell": ["CD3D", "CD3E", "CD8A", "CD4", "IL7R"],
    "nk_cell": ["NKG7", "GNLY", "KLRD1", "PRF1", "GZMB"],
    "monocyte": ["CD14", "FCN1", "VCAN", "S100A8"],
    "macrophage": ["CD68", "MRC1", "MSR1", "CSF1R"],
    "neutrophil": ["FCGR3B", "CSF3R", "S100A8", "S100A9"],
    "dendritic": ["FCER1A", "CD1C", "CLEC10A", "IRF8"],
    "plasma_cell": ["MZB1", "JCHAIN", "IGKC", "XBP1", "PRDM1"],
    "erythroid": ["HBB", "HBA1", "HBA2", "GYPA", "GATA1"],
    "megakaryocyte": ["PF4", "PPBP", "GP9", "ITGA2B"],
    "progenitor": ["CD34", "KIT", "FLT3", "MEIS1", "HOXA9"],
}


MIN_THETA = 0.3    # dispersion floor (very overdispersed / bursty genes)
MAX_THETA = 1000.0  # dispersion ceiling (near-Poisson -- no real overdispersion signal)


def _sample_from_profiles(
    n_cells: int,
    gene_names: np.ndarray,
    cell_types: list,
    proportions: np.ndarray,
    profiles: Dict[str, np.ndarray],       # ct -> [n_genes] mean-expression fractions (sum to 1)
    thetas: Dict[str, np.ndarray],         # ct -> [n_genes] per-gene NB dispersion, from real data
    lib_size_pool: Dict[str, np.ndarray],  # ct -> array of real total_counts to bootstrap from
    rng: np.random.Generator,
) -> tuple:
    """
    Independent per-gene NB(mean, dispersion) draws, with dispersion estimated
    from the reference's real per-gene variance -- genes that are noisy/bursty
    in real data get correspondingly noisy synthetic counts, and stable genes
    stay stable, rather than every gene getting the same shared noise level.
    Standard approach in single-cell simulators (e.g. splatter, scDesign):
    per-cell library size sets each gene's target mean, dispersion carries
    over from the reference at roughly that same scale.
    """
    n_genes = len(gene_names)
    target_type = rng.choice(cell_types, size=n_cells, p=proportions)

    X = np.zeros((n_cells, n_genes), dtype=np.float32)
    for ct in cell_types:
        idx = np.where(target_type == ct)[0]
        if len(idx) == 0:
            continue
        lib = rng.choice(lib_size_pool[ct], size=len(idx)).astype(np.float64)  # [n_ct]

        mu = lib[:, None] * profiles[ct][None, :]             # [n_ct, n_genes]
        theta = thetas[ct][None, :]                           # [1, n_genes], broadcasts
        p = theta / (theta + mu)                               # numpy's negative_binomial: mean = n*(1-p)/p
        p = np.clip(p, 1e-10, 1.0 - 1e-10)
        n_param = np.broadcast_to(theta, mu.shape)

        X[idx] = rng.negative_binomial(n_param, p).astype(np.float32)

    return X, target_type


def _build_reference_profiles(reference_h5ad: str, celltype_key: str, gene_names: np.ndarray):
    """Aggregate-only: per-gene mean/dispersion and a library-size pool per well-represented cell type."""
    ref = ad.read_h5ad(reference_h5ad)
    if not np.array_equal(ref.var_names.values, gene_names):
        raise ValueError(
            f"{reference_h5ad}'s var_names don't match the trained gene panel exactly -- "
            "reindex the reference to weights/scvi_model.pt's var_names before using it here."
        )

    X = ref.layers["counts"] if "counts" in ref.layers else ref.X
    if hasattr(X, "toarray"):
        X = X.toarray()
    X = np.asarray(X, dtype=np.float64)

    counts = ref.obs[celltype_key].value_counts()
    included = counts[counts >= MIN_REFERENCE_CELLS_PER_TYPE].index.tolist()
    excluded = counts[counts < MIN_REFERENCE_CELLS_PER_TYPE]
    if len(excluded) > 0:
        print(
            f"[make_toy_sample] Excluding cell types with <{MIN_REFERENCE_CELLS_PER_TYPE} "
            f"reference cells (too few for a reliable profile): {excluded.to_dict()}"
        )

    profiles, thetas, lib_pools = {}, {}, {}
    labels = ref.obs[celltype_key].values
    for ct in included:
        mask = labels == ct
        Xct = X[mask]
        mean_g = Xct.mean(axis=0)
        var_g = Xct.var(axis=0)

        # NB: var = mean + mean^2/theta  =>  theta = mean^2 / (var - mean).
        # Where var <= mean (no overdispersion signal, or too few cells to
        # tell), fall back to MAX_THETA (near-Poisson) rather than a
        # division by ~0 or a negative theta.
        excess = var_g - mean_g
        theta_g = np.full_like(mean_g, MAX_THETA)
        overdispersed = excess > 1e-6
        theta_g[overdispersed] = (mean_g[overdispersed] ** 2) / excess[overdispersed]
        theta_g = np.clip(theta_g, MIN_THETA, MAX_THETA)

        profiles[ct] = mean_g / mean_g.sum()
        thetas[ct] = theta_g
        lib_pools[ct] = Xct.sum(axis=1)

    proportions = counts[included].values.astype(np.float64)
    proportions /= proportions.sum()

    return included, proportions, profiles, thetas, lib_pools


def _build_modules(gene_names: np.ndarray, rng: np.random.Generator):
    n_genes = len(gene_names)
    gene_index = {g: i for i, g in enumerate(gene_names)}
    cell_types = sorted(MARKER_GENES.keys())
    n_modules = len(cell_types)

    loadings = np.zeros((n_genes, n_modules), dtype=np.float32)
    used = set()
    for m, ct in enumerate(cell_types):
        core_idx = [gene_index[g] for g in MARKER_GENES[ct] if g in gene_index]
        loadings[core_idx, m] = CORE_MARKER_LOADING
        used.update(core_idx)

        available = np.array([i for i in range(n_genes) if i not in used])
        extra_idx = rng.choice(available, size=min(N_EXTRA_GENES_PER_MODULE, len(available)), replace=False)
        loadings[extra_idx, m] = rng.lognormal(mean=0.5, sigma=0.5, size=len(extra_idx)).astype(np.float32)
        used.update(extra_idx.tolist())

    return loadings, cell_types


def _generate_fallback(n_cells, gene_names, rng):
    n_genes = len(gene_names)
    loadings, cell_types = _build_modules(gene_names, rng)
    n_modules = len(cell_types)
    type_to_idx = {ct: i for i, ct in enumerate(cell_types)}
    target_type = rng.choice(cell_types, size=n_cells)

    gene_baseline_log = rng.normal(loc=0.0, scale=1.0, size=n_genes).astype(np.float32)
    activity = rng.normal(loc=0.0, scale=CROSSTALK_ACTIVITY_SD, size=(n_cells, n_modules)).astype(np.float32)
    own_module = np.array([type_to_idx[t] for t in target_type])
    activity[np.arange(n_cells), own_module] = rng.normal(
        loc=OWN_MODULE_ACTIVITY_MEAN, scale=OWN_MODULE_ACTIVITY_SD, size=n_cells
    ).astype(np.float32)

    log_rate = (gene_baseline_log[None, :] + activity @ loadings.T).astype(np.float64)
    log_rate -= log_rate.max(axis=1, keepdims=True)
    probs = np.exp(log_rate)
    probs /= probs.sum(axis=1, keepdims=True)

    library_sizes = rng.lognormal(mean=np.log(2500), sigma=0.4, size=n_cells)
    X = np.zeros((n_cells, n_genes), dtype=np.float32)
    for i in range(n_cells):
        X[i] = rng.multinomial(int(library_sizes[i]), probs[i]).astype(np.float32)
    return X, target_type


def main(
    n_cells: int = 3000,
    seed: int = 0,
    out_path: Optional[str] = None,
    reference_h5ad: Optional[str] = None,
    celltype_key: str = "celltype_coarse",
):
    rng = np.random.default_rng(seed)

    ckpt = torch.load(os.path.join(ARTIFACTS_DIR, "scvi_model.pt"), map_location="cpu", weights_only=False)
    gene_names = np.asarray(ckpt["var_names"])

    if reference_h5ad is not None:
        cell_types, proportions, profiles, thetas, lib_pools = _build_reference_profiles(
            reference_h5ad, celltype_key, gene_names
        )
        X, target_type = _sample_from_profiles(
            n_cells, gene_names, cell_types, proportions, profiles, thetas, lib_pools, rng
        )
    else:
        X, target_type = _generate_fallback(n_cells, gene_names, rng)

    obs = pd.DataFrame({
        "person_id": ["toy_donor_001"] * n_cells,
        "celltype_coarse": target_type,
    }, index=[f"toy_cell_{i:05d}" for i in range(n_cells)])

    # scRNA-seq counts are overwhelmingly zero per cell (~10^4 genes, a few
    # thousand total counts) -- store sparse so the shipped demo file stays
    # small (dense float32 here would be ~60MB for this shape).
    import scipy.sparse as sp
    adata = ad.AnnData(X=sp.csr_matrix(X), obs=obs, var=pd.DataFrame(index=gene_names))

    out_path = out_path or os.path.join(HERE, "toy_sample.h5ad")
    adata.write_h5ad(out_path, compression="gzip")
    print(f"Wrote {out_path}: {adata.shape[0]} cells x {adata.shape[1]} genes")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=str, default=None, help="Path to a real reference .h5ad (see module docstring)")
    parser.add_argument("--n-cells", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--celltype-key", type=str, default="celltype_coarse")
    args = parser.parse_args()
    main(n_cells=args.n_cells, seed=args.seed, reference_h5ad=args.reference, celltype_key=args.celltype_key)
