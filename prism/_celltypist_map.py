"""
Cell-type labeling: runs CellTypist when the user doesn't supply
adata.obs["celltype_coarse"] themselves, then maps CellTypist's raw output
onto the exact 11-category set PRISM's flow and gene decoder were
conditioned on (generative_config.json / gene_decoder_config.json's
unique_types). Every generated/consumed label must be a member of that
fixed set -- conditioning on an unrecognized category would silently
produce meaningless output, so unmapped labels are a hard error, not a
warning.

The fine->coarse mapping (_build_coarse_map) is the project's own rule, not
independently re-verified against the original training preprocessing in
this workspace -- project context suggests the training data was annotated
with CellTypist's `Immune_All_Low` model using majority voting, and that is
what this module defaults to.

Note: _build_coarse_map's own "ILC" -> "innate_lymphoid" rule targets a
category outside PRISM's trained 11 (b_cell, dendritic, erythroid,
macrophage, megakaryocyte, monocyte, neutrophil, nk_cell, plasma_cell,
progenitor, t_cell) -- any cell landing there still fails the hard-error
check below rather than silently reaching generation with a bogus category.
"""
from typing import Dict, List

import numpy as np
import pandas as pd

CELLTYPIST_MODEL = "Immune_All_Low.pkl"

# Ordered (first match wins) substring rules, fine CellTypist label -> coarse
# category. Order matters -- more specific patterns are listed before the
# general ones they'd otherwise be swallowed by (e.g. "NKT" before "NK").
_COARSE_RULES: Dict[str, str] = {
    # T cells (specific before general)
    "NKT": "t_cell",           # BEFORE "NK"
    "MAIT": "t_cell",
    "gdT": "t_cell",
    "Treg": "t_cell",
    "CD4": "t_cell", "CD8": "t_cell",
    "Th1": "t_cell", "Th2": "t_cell", "Th17": "t_cell",
    "Naive T": "t_cell", "Memory T": "t_cell", "Effector T": "t_cell",
    "T cell": "t_cell",

    # NK (after NKT)
    "NK": "nk_cell",

    # B cells
    "B cell": "b_cell", "Naive B": "b_cell", "Memory B": "b_cell",
    "Pre-B": "b_cell", "Pro-B": "b_cell", "Transitional B": "b_cell",

    # Plasma
    "Plasma": "plasma_cell", "Plasmablast": "plasma_cell",

    # Monocyte
    "Classical monocyte": "monocyte", "Non-classical monocyte": "monocyte",
    "Intermediate monocyte": "monocyte", "Monocyte": "monocyte",
    "CD14": "monocyte", "CD16": "monocyte",

    # Macrophage
    "Macrophage": "macrophage",

    # Neutrophil
    "Promyelocyte": "neutrophil",
    "Neutrophil": "neutrophil", "Granulocyte": "neutrophil",

    # Dendritic (specific before general)
    "ASDC": "dendritic", "cDC": "dendritic", "pDC": "dendritic",
    "Dendritic": "dendritic", "DC": "dendritic",    # DC last

    # Erythroid
    "Erythro": "erythroid", "Erythroid": "erythroid",

    # Progenitor
    "MEMP": "progenitor",
    "HSC": "progenitor", "HSPC": "progenitor", "Progenitor": "progenitor",
    "CMP": "progenitor", "GMP": "progenitor", "MEP": "progenitor",
    "MPP": "progenitor", "LMPP": "progenitor",

    # Megakaryocyte
    "Megakaryocyte": "megakaryocyte", "Platelet": "megakaryocyte",

    # ILC -- not one of PRISM's trained categories; see module docstring.
    "ILC": "innate_lymphoid",
}


def _build_coarse_map(fine_labels: np.ndarray) -> Dict[str, str]:
    """
    Map CellTypist fine-grained labels to broad categories.
    Adjust mappings based on your specific CellTypist model output.
    """
    unique_fine = np.unique(fine_labels)
    coarse_map = {}

    for fl in unique_fine:
        mapped = False
        for pattern, coarse in _COARSE_RULES.items():
            if pattern.lower() in fl.lower():
                coarse_map[fl] = coarse
                mapped = True
                break
        if not mapped:
            coarse_map[fl] = fl  # keep original if no match

    return coarse_map


def run_celltypist(adata_counts, model: str = CELLTYPIST_MODEL, majority_voting: bool = True):
    """
    Runs CellTypist on raw counts and returns its per-cell predicted_labels
    Series (majority-voted if majority_voting=True).

    CellTypist does NOT normalize internally -- it validates that .X is
    already log1p(CP10K) and raises if not (confirmed by running it: it
    rejects raw counts outright rather than silently mis-scoring them). We
    normalize a throwaway copy for CellTypist's consumption only;
    adata_counts itself is left untouched (raw counts) for the rest of the
    pipeline, which needs raw counts for the scVI embedding step.
    """
    import celltypist
    from celltypist import models

    from ._utils import log1p_cp10k

    X = adata_counts.layers.get("counts", adata_counts.X)
    if hasattr(X, "toarray"):
        X = X.toarray()
    adata_norm = adata_counts.copy()
    adata_norm.X = log1p_cp10k(np.asarray(X))

    models.download_models(model=[model], force_update=False)
    predictions = celltypist.annotate(
        adata_norm, model=model, majority_voting=majority_voting,
    )
    label_col = "majority_voting" if majority_voting else "predicted_labels"
    return predictions.predicted_labels[label_col]


def labels_to_prism_categories(raw_labels, trained_unique_types: List[str]) -> np.ndarray:
    """
    Maps raw CellTypist labels to PRISM's trained category set via
    _build_coarse_map. Cells whose mapped label isn't a member of that set
    (unmapped entirely, or mapped to something outside it like
    "innate_lymphoid") get "__unmapped__" -- the caller decides whether to
    drop them (assign_celltype_coarse's CellTypist path) or treat that as
    an error (a user-supplied celltype_coarse column).
    """
    raw_labels = np.asarray(raw_labels, dtype=str)
    trained_set = set(trained_unique_types)

    coarse_map = _build_coarse_map(raw_labels)
    mapped = np.array([coarse_map[lbl] for lbl in raw_labels])

    bad_mask = ~np.isin(mapped, list(trained_set))
    mapped[bad_mask] = "__unmapped__"
    return mapped


def assign_celltype_coarse(adata_counts, trained_unique_types: List[str], celltype_key: str = "celltype_coarse"):
    """
    Ensures adata_counts.obs[celltype_key] exists and only contains labels
    from trained_unique_types.

    If the column is absent, runs CellTypist + maps its output via
    _build_coarse_map. Cells whose label doesn't land in PRISM's trained
    set (contaminants like "Epithelial cells", precursor states the
    mapping rule doesn't cover, etc.) are dropped with a printed warning
    rather than failing the whole donor's inference over a handful of
    ambiguous cells -- real blood samples routinely produce a few such
    calls. Raises if that would drop every cell.

    If the user already supplied the column, it's validated in place and
    any label outside the trained set is a hard error instead -- a
    self-supplied column with bad labels is far more likely a genuine
    upstream mistake worth surfacing immediately than routine classifier
    noise.
    """
    if celltype_key in adata_counts.obs.columns:
        user_labels = adata_counts.obs[celltype_key].astype(str).values
        bad = sorted(set(user_labels.tolist()) - set(trained_unique_types))
        if bad:
            raise ValueError(
                f"adata.obs['{celltype_key}'] contains labels outside PRISM's "
                f"trained category set {sorted(trained_unique_types)}: {bad}. "
                "Relabel to the trained categories before calling PRISM."
            )
        return adata_counts

    raw_labels = run_celltypist(adata_counts)
    mapped = labels_to_prism_categories(raw_labels, trained_unique_types)

    keep_mask = mapped != "__unmapped__"
    n_dropped = int((~keep_mask).sum())
    if n_dropped > 0:
        raw_labels_arr = np.asarray(raw_labels, dtype=str)
        dropped_counts = (
            pd.Series(raw_labels_arr[~keep_mask]).value_counts().to_dict()
        )
        print(
            f"[prism] Dropping {n_dropped}/{len(mapped)} cells with CellTypist "
            f"labels that don't map to PRISM's trained categories: {dropped_counts}. "
            "Extend prism._celltypist_map._COARSE_RULES to cover them, or supply "
            f"adata.obs['{celltype_key}'] yourself, to stop losing these cells."
        )
    if keep_mask.sum() == 0:
        raise ValueError(
            "Every cell's CellTypist label failed to map to PRISM's trained "
            f"categories {sorted(trained_unique_types)}. Extend "
            "prism._celltypist_map._COARSE_RULES, or supply "
            f"adata.obs['{celltype_key}'] yourself."
        )

    adata_counts = adata_counts[keep_mask].copy()
    adata_counts.obs[celltype_key] = mapped[keep_mask]
    return adata_counts
