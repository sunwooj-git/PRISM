"""
k-NN retrieval against the real reference bone marrow cohort, for predicting
a donor's bone-marrow-like cell-type composition -- the same mechanism
paper/bm_retrieval_v4.py's BMReferenceDB.query_blood_cells() and
paper/bm_generation_v6.py's generate_bm_from_blood_v3() use (k=20, brute-
force Euclidean in encoder embedding space), reimplemented standalone here
since the training-era BMReferenceDB carries substantial functionality
(donor/cluster composition, deconvolution) this package doesn't need --
only the k-NN + cell-type tally.

Why this exists, not just the donor's own CellTypist labels: retrieving
each of a donor's own bone-marrow-like cells' nearest real reference BM
neighbors and tallying THEIR real cell-type labels smooths over the sparse,
noisy cell-type calls a small number of a single donor's own cells give --
a type the donor's own CellTypist output never produced can still get
nonzero predicted proportion if cells embedding near it are common in the
reference. Confirmed this is the actual mechanism the original pipeline
used for its "predicted proportions" (not the donor's own label counts).
"""
from typing import Dict

import numpy as np


def knn_predict_composition(
    Z_query: np.ndarray,
    Z_reference: np.ndarray,
    celltype_reference: np.ndarray,
    k: int = 20,
) -> Dict[str, float]:
    """
    For each row of Z_query, find its k nearest neighbors (Euclidean) in
    Z_reference, pool all retrieved neighbors across all query cells, and
    return the fraction of pooled neighbors belonging to each cell type.
    """
    Z_query = np.asarray(Z_query, dtype=np.float32)
    Z_reference = np.asarray(Z_reference, dtype=np.float32)
    k = min(k, len(Z_reference))

    # Squared Euclidean via ||a-b||^2 = ||a||^2 + ||b||^2 - 2*a.b -- matches
    # BMReferenceDB._find_knn_euclidean's exact approach.
    q_sq = np.sum(Z_query ** 2, axis=1, keepdims=True)
    r_sq = np.sum(Z_reference ** 2, axis=1, keepdims=True).T
    dist_sq = q_sq + r_sq - 2.0 * (Z_query @ Z_reference.T)
    dist_sq = np.maximum(dist_sq, 0.0)

    top_k_idx = np.argpartition(dist_sq, k - 1, axis=1)[:, :k]

    all_nn = celltype_reference[top_k_idx.ravel()]
    labels, counts = np.unique(all_nn, return_counts=True)
    total = counts.sum()
    return {str(lbl): float(c) / total for lbl, c in zip(labels, counts)}
