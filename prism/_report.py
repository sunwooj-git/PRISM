"""Summary report for generated cells: composition, read-count stats, UMAP overlay."""
import os
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd

from ._utils import log1p_cp10k

# Fixed color per PRISM cell-type category, shared across every plot this
# module produces (reference-only panel, generated-over-reference overlay,
# and any future one) so the same cell type always reads as the same color
# regardless of which categories happen to be present in a given panel.
CELLTYPE_COLORS: Dict[str, str] = {
    "b_cell": "#4C72B0",
    "dendritic": "#DD8452",
    "erythroid": "#55A868",
    "macrophage": "#C44E52",
    "megakaryocyte": "#8172B2",
    "monocyte": "#937860",
    "neutrophil": "#DA8BC3",
    "nk_cell": "#8C8C8C",
    "plasma_cell": "#CCB974",
    "progenitor": "#64B5CD",
    "t_cell": "#1F77B4",
}


@dataclass
class GenerationReport:
    cell_type_composition: pd.Series          # fraction of generated cells per cell type
    read_count_stats: pd.DataFrame             # per-cell-type mean/median/std total counts
    umap_figure_path: Optional[str]


@dataclass
class ReferenceUMAP:
    """
    A UMAP fit ONCE on the reference bone marrow cells alone and reused for
    every subsequent report -- generated cells are projected into this same
    fixed space via .transform(), not refit from scratch each time.

    Why this matters: fitting a fresh joint (reference + generated) UMAP on
    every call, as an earlier version of this module did, gives each run an
    arbitrarily different rotation/reflection of the same underlying
    structure -- UMAP only preserves local neighbor relationships, not any
    canonical global orientation, so two different generated-cell point
    sets joined to the identical reference converge to two different-
    looking layouts even with the same random_state. That made two runs'
    figures visually incomparable despite both being individually correct.
    Fitting once and transforming afterward keeps the reference layout (and
    therefore cross-run comparisons) stable.
    """
    reducer: object
    coords: np.ndarray          # [n_ref, 2], the reference cells' fixed layout
    celltype: Optional[np.ndarray]
    xlim: tuple
    ylim: tuple


def summarize_generated(X_gen: np.ndarray, cell_type: np.ndarray) -> pd.DataFrame:
    lib = X_gen.sum(axis=1)
    df = pd.DataFrame({"cell_type": cell_type, "total_counts": lib})
    stats = df.groupby("cell_type")["total_counts"].agg(["mean", "median", "std", "count"])
    return stats


def cell_type_composition(cell_type: np.ndarray) -> pd.Series:
    s = pd.Series(cell_type).value_counts(normalize=True)
    s.name = "fraction"
    return s


def _color_for(ct_name: str) -> str:
    return CELLTYPE_COLORS.get(ct_name, "#000000")


def fit_reference_umap(
    Z_bm_reference: np.ndarray,
    celltype_bm_reference: Optional[np.ndarray] = None,
    max_reference_cells: int = 20000,
    seed: int = 0,
) -> ReferenceUMAP:
    """Fits once; cache the result (e.g. on PRISMModel) and reuse across calls."""
    import umap

    rng = np.random.default_rng(seed)
    if len(Z_bm_reference) > max_reference_cells:
        idx = rng.choice(len(Z_bm_reference), size=max_reference_cells, replace=False)
        Z_ref = Z_bm_reference[idx]
        ct_ref = celltype_bm_reference[idx] if celltype_bm_reference is not None else None
    else:
        Z_ref = Z_bm_reference
        ct_ref = celltype_bm_reference

    reducer = umap.UMAP(random_state=seed)
    coords = reducer.fit_transform(Z_ref.astype(np.float32))

    return ReferenceUMAP(
        reducer=reducer, coords=coords, celltype=ct_ref,
        xlim=(coords[:, 0].min(), coords[:, 0].max()),
        ylim=(coords[:, 1].min(), coords[:, 1].max()),
    )


def plot_umap_overlay(
    ref_umap: ReferenceUMAP,
    Z_generated: np.ndarray,
    cell_type_generated: np.ndarray,
    out_path: str,
) -> str:
    """
    Two-panel figure sharing the SAME fixed reference UMAP layout (see
    ReferenceUMAP / fit_reference_umap -- generated cells are projected into
    it via .transform(), not refit jointly):
      left  -- reference bone marrow cells alone, colored by their real
               cell type (only drawn if ref_umap.celltype is available --
               consensus_bm_reference.npz itself carries no cell-type
               labels; this needs the separate bm_reference_celltypes.npz
               artifact aligned to it).
      right -- generated cells (colored by cell type) over the reference
               (grey), as before.
    Colors are shared across both panels and fixed per cell type (see
    CELLTYPE_COLORS) rather than assigned by draw order, so the same cell
    type reads as the same color in both, and across different report calls.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    gen_coords = ref_umap.reducer.transform(Z_generated.astype(np.float32))

    has_ref_labels = ref_umap.celltype is not None
    fig, axes = plt.subplots(1, 2 if has_ref_labels else 1, figsize=(13 if has_ref_labels else 7, 6))
    if not has_ref_labels:
        axes = [axes]

    if has_ref_labels:
        ax_ref = axes[0]
        for ct_name in sorted(set(ref_umap.celltype.tolist())):
            m = ref_umap.celltype == ct_name
            ax_ref.scatter(ref_umap.coords[m, 0], ref_umap.coords[m, 1], s=3, c=_color_for(ct_name), label=ct_name, linewidths=0)
        ax_ref.set_xlim(*ref_umap.xlim)
        ax_ref.set_ylim(*ref_umap.ylim)
        ax_ref.set_xlabel("UMAP 1")
        ax_ref.set_ylabel("UMAP 2")
        ax_ref.set_title("Reference bone marrow cells (real, by cell type)")
        ax_ref.legend(markerscale=3, fontsize=7, bbox_to_anchor=(1.02, 1), loc="upper left")

    ax_overlay = axes[-1]
    ax_overlay.scatter(ref_umap.coords[:, 0], ref_umap.coords[:, 1], s=3, c="lightgrey", label="reference BM", linewidths=0)
    for ct_name in sorted(set(cell_type_generated.tolist())):
        m = cell_type_generated == ct_name
        ax_overlay.scatter(gen_coords[m, 0], gen_coords[m, 1], s=6, c=_color_for(ct_name), label=ct_name, linewidths=0)
    ax_overlay.set_xlim(*ref_umap.xlim)
    ax_overlay.set_ylim(*ref_umap.ylim)
    ax_overlay.set_xlabel("UMAP 1")
    ax_overlay.set_ylabel("UMAP 2")
    ax_overlay.set_title("Generated bone marrow-like cells vs. reference")
    ax_overlay.legend(markerscale=2, fontsize=7, bbox_to_anchor=(1.02, 1), loc="upper left")

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def build_report(
    X_gen: np.ndarray,
    Z_gen: np.ndarray,
    cell_type_gen: np.ndarray,
    ref_umap: Optional[ReferenceUMAP] = None,
    umap_out_path: Optional[str] = None,
) -> GenerationReport:
    composition = cell_type_composition(cell_type_gen)
    stats = summarize_generated(X_gen, cell_type_gen)

    umap_path = None
    if umap_out_path is not None and ref_umap is not None:
        umap_path = plot_umap_overlay(
            ref_umap=ref_umap, Z_generated=Z_gen, cell_type_generated=cell_type_gen,
            out_path=umap_out_path,
        )

    return GenerationReport(
        cell_type_composition=composition,
        read_count_stats=stats,
        umap_figure_path=umap_path,
    )
