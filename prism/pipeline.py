"""
Main inference entry point: loads the trained PRISM model bundle once, then
runs a user's blood AnnData through encoder -> NMF -> flow -> gene decoder
to produce the package's five outputs.
"""
import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import anndata as ad
import numpy as np
import pandas as pd

from . import config
from ._artifacts import get_artifact_dir
from ._celltypist_map import assign_celltype_coarse
from ._encoder import EncoderBundle, apply_z_calibration, encode_cells, load_encoder, z_to_percentile
from ._generation import GenerationBundle, generate_per_celltype, load_generation_models
from ._programs import ProgramBundle, load_programs, program_scores
from ._report import GenerationReport, ReferenceUMAP, build_report, fit_reference_umap
from ._retrieval import knn_predict_composition
from ._scvi_embed import embed_blood_counts
from ._utils import assert_raw_counts, resolve_device


@dataclass
class PRISMModel:
    """Loaded trained model bundle -- construct once via prism.load_model(), reuse across calls."""
    artifacts_dir: str
    device: str
    encoder: EncoderBundle
    programs: ProgramBundle
    generation: GenerationBundle
    bm_reference_Z: np.ndarray            # consensus_bm_reference.npz's Z_bm, for the UMAP overlay + retrieval
    bm_reference_celltype: np.ndarray     # bm_reference_celltypes.npz, aligned to bm_reference_Z
    bm_program_means_per_ct: Dict[str, np.ndarray]  # real reference population's per-cell-type mean P
    _reference_umap: Optional[ReferenceUMAP] = field(default=None, init=False, repr=False)

    def reference_umap(self, seed: int = 0) -> ReferenceUMAP:
        """
        Lazily fit-once, cache, and reuse the reference-cells UMAP layout so
        every run_inference() call (and every donor within one) plots
        generated cells into the SAME fixed 2D space -- see
        prism._report.ReferenceUMAP for why refitting per call is wrong.
        """
        if self._reference_umap is None:
            self._reference_umap = fit_reference_umap(
                self.bm_reference_Z, self.bm_reference_celltype, seed=seed,
            )
        return self._reference_umap


def load_model(local_dir: Optional[str] = None, device: str = config.DEVICE_DEFAULT) -> PRISMModel:
    """
    Load the full trained PRISM model bundle. Downloads weights from
    Hugging Face Hub on first use (cached thereafter) unless local_dir
    already contains them -- see prism._artifacts.get_artifact_dir.
    """
    device = resolve_device(device)
    artifacts_dir = str(get_artifact_dir(local_dir=local_dir))

    encoder = load_encoder(artifacts_dir, device=device)
    programs = load_programs(artifacts_dir)
    generation = load_generation_models(artifacts_dir, device=device)

    bm_ref = np.load(os.path.join(artifacts_dir, "consensus_bm_reference.npz"), allow_pickle=True)
    bm_reference_Z = np.asarray(bm_ref["Z_bm"], dtype=np.float32)
    bm_reference_P = np.asarray(bm_ref["P_bm"], dtype=np.float32)

    ct_ref = np.load(os.path.join(artifacts_dir, "bm_reference_celltypes.npz"), allow_pickle=True)
    categories = ct_ref["categories"]
    bm_reference_celltype = categories[ct_ref["codes"]]
    if len(bm_reference_celltype) != len(bm_reference_Z):
        raise RuntimeError(
            f"bm_reference_celltypes.npz has {len(bm_reference_celltype)} entries but "
            f"consensus_bm_reference.npz's Z_bm has {len(bm_reference_Z)} -- these must be "
            "row-aligned (same order) or the reference UMAP panel would mislabel cells."
        )

    bm_program_means_per_ct = {
        ct: bm_reference_P[bm_reference_celltype == ct].mean(axis=0).astype(np.float32)
        for ct in generation.unique_types
        if (bm_reference_celltype == ct).any()
    }

    return PRISMModel(
        artifacts_dir=artifacts_dir, device=device, encoder=encoder,
        programs=programs, generation=generation, bm_reference_Z=bm_reference_Z,
        bm_reference_celltype=bm_reference_celltype,
        bm_program_means_per_ct=bm_program_means_per_ct,
    )


@dataclass
class DonorResult:
    donor_id: str
    n_cells_total: int
    n_bonemarrowlike: int
    bonemarrowlike_threshold_percentile: float  # e.g. 90.0 -- config.MARROWLIKE_PERCENTILE
    bonemarrowlike_threshold_zscore: float      # the actual per-cell marrow_z cutoff this percentile
                                                 # corresponds to for this trained model (see
                                                 # prism._encoder.z_to_percentile) -- cells at or
                                                 # above this raw z-score were called bone marrow-like
    celltype_proportions: pd.Series          # Output 1
    program_scores_donor: np.ndarray         # Output 2: [5], this donor's own program scores
    program_scores_per_cell: np.ndarray      # [n_cells_total, 5], diagnostic
    generated_adata: ad.AnnData              # Output 3: N_GEN synthetic cells
    report: GenerationReport                 # Output 4


@dataclass
class InferenceResult:
    per_donor: Dict[str, DonorResult] = field(default_factory=dict)


def _fallback_program_vector(P_bonemarrowlike: np.ndarray, marrow_z: np.ndarray, tail_q: float) -> np.ndarray:
    n = len(marrow_z)
    n_tail = max(1, int(np.ceil(n * tail_q)))
    tail_idx = np.argsort(-marrow_z)[:n_tail]
    return P_bonemarrowlike[tail_idx].mean(axis=0).astype(np.float32) if len(P_bonemarrowlike) else \
        np.zeros(5, dtype=np.float32)


_LIBRARY_SIZE_COLUMN_CANDIDATES = ("total_counts", "n_counts")


def run_inference(
    model: PRISMModel,
    adata: ad.AnnData,
    donor_key: str = "person_id",
    celltype_key: str = "celltype_coarse",
    counts_layer: Optional[str] = None,
    library_size_key: Optional[str] = None,
    n_gen: int = config.N_GEN_DEFAULT,
    umap_out_dir: Optional[str] = None,
    sample_counts: bool = True,
    seed: int = 0,
) -> InferenceResult:
    """
    Run the full PRISM inference pipeline on a user's blood AnnData,
    producing all five I/O-contract outputs per donor present in the data.

    See the top-level README's "Input requirements" table for what
    adata must contain. Multiple donors are supported -- outputs are
    computed and generation is run independently per donor.

    library_size_key:
        adata.obs column with each cell's true total transcript count,
        e.g. scanpy's own `total_counts` (from sc.pp.calculate_qc_metrics)
        computed BEFORE any gene-panel reduction. If None, auto-detects a
        column named "total_counts" or "n_counts"; if neither is present,
        falls back to summing adata.X/counts_layer directly. That fallback
        UNDERESTIMATES true library size if adata's genes have already been
        reduced to some panel smaller than the full transcriptome (a common
        preprocessing step) -- confirmed directly against a real donor: a
        ~30-40% library-size underestimate here propagates linearly into
        generated cells' read-count scale (Output 3/4), since generation's
        per-cell-type library size is this value times a fixed trained
        blood->BM scaling factor. Supply the true value via this column (or
        make sure adata.X/counts_layer holds the full transcriptome) if
        your input has already been gene-filtered.
    """
    if donor_key not in adata.obs.columns:
        raise ValueError(f"adata.obs['{donor_key}'] is required (donor identifier column).")

    X_raw = adata.layers[counts_layer] if counts_layer is not None else adata.X
    assert_raw_counts(X_raw, name="adata.X" if counts_layer is None else f"adata.layers['{counts_layer}']")

    adata = assign_celltype_coarse(
        adata, trained_unique_types=model.generation.unique_types, celltype_key=celltype_key,
    )
    # assign_celltype_coarse may have dropped cells with unmappable CellTypist
    # labels -- recompute X_raw/lib_sizes from the (possibly smaller) result,
    # never the original, or donor_ids/celltypes below go out of alignment.
    X_raw = adata.layers[counts_layer] if counts_layer is not None else adata.X

    # --- scVI embed (raw counts -> 32-dim joint latent) ---
    Z_scvi = embed_blood_counts(
        adata, scvi_model_path=os.path.join(model.artifacts_dir, "scvi_model.pt"),
        donor_key=donor_key, counts_layer=counts_layer,
        device=model.device, seed=seed,
    )

    # --- encoder: 32-dim scVI latent -> tissue logits + PRISM's own 32-dim Z ---
    tissue_logits, Z = encode_cells(model.encoder.model, Z_scvi, device=model.device)
    marrow_logit = (tissue_logits[:, 1] - tissue_logits[:, 0]).astype(np.float32)
    marrow_z = apply_z_calibration(marrow_logit, model.encoder.mcal)
    marrow_pctile = z_to_percentile(marrow_z, model.encoder.blood_z_ref)
    bonemarrowlike_mask = marrow_pctile >= config.MARROWLIKE_PERCENTILE
    # The actual raw marrow_z cutoff config.MARROWLIKE_PERCENTILE corresponds to for
    # this trained model -- informational (attached to DonorResult below), not used
    # for the mask itself (that's computed via the percentile rank directly above).
    bonemarrowlike_threshold_zscore = float(np.percentile(model.encoder.blood_z_ref, config.MARROWLIKE_PERCENTILE))

    # --- NMF program scores, all cells ---
    P_all = program_scores(Z, model.programs.prog_model, model.programs.prog_scaler, model.programs.prog_method)

    donor_ids = adata.obs[donor_key].astype(str).values
    celltypes = adata.obs[celltype_key].astype(str).values

    lib_col = library_size_key
    if lib_col is None:
        lib_col = next((c for c in _LIBRARY_SIZE_COLUMN_CANDIDATES if c in adata.obs.columns), None)
    if lib_col is not None:
        lib_sizes = adata.obs[lib_col].astype(np.float64).values
    else:
        print(
            "[prism] No library_size_key given and no 'total_counts'/'n_counts' column found -- "
            "computing library size by summing adata.X/counts_layer directly. This UNDERESTIMATES "
            "true library size if those genes have already been reduced from the full transcriptome "
            "(see run_inference's library_size_key docstring)."
        )
        lib_sizes = np.asarray(X_raw.toarray() if hasattr(X_raw, "toarray") else X_raw, dtype=np.float64).sum(axis=1)

    result = InferenceResult()
    for donor in sorted(pd.unique(donor_ids)):
        d_mask = donor_ids == donor
        d_bml_mask = d_mask & bonemarrowlike_mask

        n_total = int(d_mask.sum())
        n_bml = int(d_bml_mask.sum())

        ct_bml = celltypes[d_bml_mask]
        # Output 1: direct empirical composition of this donor's own labeled
        # bone marrow-like cells -- unchanged, matches the handoff's literal
        # definition ("among the input blood cells identified as marrow-
        # like, the breakdown by cell type"). Generation itself uses a
        # separate, retrieval-based proportion estimate below (see
        # predicted_proportions), matching the original training pipeline.
        proportions = pd.Series(ct_bml).value_counts(normalize=True) if n_bml > 0 else pd.Series(dtype=float)

        P_bml = P_all[d_bml_mask]

        # --- Per-cell-type program-score conditioning: 3-tier fallback, ---
        # matching paper/bm_generation_v6.py's generate_bm_twostage_per_ct
        # exactly (verified against that source -- see config.py's
        # FALLBACK_TAIL_Q / MIN_CELLS_PER_TYPE_FOR_CONDITIONING docs).
        program_conditioning_per_ct: Dict[str, np.ndarray] = {}
        for ct in model.generation.unique_types:
            ct_mask = ct_bml == ct
            n_ct = int(ct_mask.sum())
            if n_ct >= 1:
                # Tier 1: donor's own mean for this type.
                program_conditioning_per_ct[ct] = P_bml[ct_mask].mean(axis=0).astype(np.float32)
                if n_ct < config.MIN_CELLS_PER_TYPE_FOR_CONDITIONING:
                    print(
                        f"[prism] Donor {donor!r}: only {n_ct} bone marrow-like "
                        f"{ct!r} cell(s) -- its conditioning vector is based on "
                        "limited data and may be noisy."
                    )
            elif ct in model.bm_program_means_per_ct:
                # Tier 2: real reference population's per-cell-type mean --
                # donor has zero cells of this type, but the trained
                # reference (280k+ real BM cells) does.
                program_conditioning_per_ct[ct] = model.bm_program_means_per_ct[ct]
                print(
                    f"[prism] Donor {donor!r}: no bone marrow-like {ct!r} cells -- "
                    "using the reference population's mean for this type."
                )
            # else: left unset, picked up by tier 3 (global fallback) below.

        # Tier 3 (last resort): this donor's own tail-mean, pooled across
        # ALL their cell types -- computed over their FULL blood population
        # (not just the bone-marrow-like-gated subset), matching the
        # original's P_fallback exactly (an earlier version of this file
        # scoped this to the gated subset, which doesn't match).
        P_donor_all = P_all[d_mask]
        marrow_z_donor_all = marrow_z[d_mask]
        fallback_vec = _fallback_program_vector(P_donor_all, marrow_z_donor_all, config.FALLBACK_TAIL_Q)
        for ct in model.generation.unique_types:
            program_conditioning_per_ct.setdefault(ct, fallback_vec)

        # --- Stage 1 (generation only): k-NN retrieval-predicted cell-type ---
        # proportions, matching paper/bm_generation_v6.py's
        # generate_bm_from_blood_v3 exactly -- NOT the donor's own label
        # counts (that's Output 1 above). Query set is this donor's own top
        # FALLBACK_TAIL_Q fraction of their FULL blood population by
        # marrowness (the same tail P_fallback above uses), each cell's
        # RETRIEVAL_K=20 nearest real reference BM neighbors are pooled and
        # their real cell types tallied -- smooths over a donor's own
        # sparse/noisy per-type label counts.
        n_tail = max(1, int(np.ceil(n_total * config.FALLBACK_TAIL_Q)))
        tail_idx = np.argsort(-marrow_z_donor_all)[:n_tail]
        Z_donor_all = Z[d_mask]
        predicted_proportions = knn_predict_composition(
            Z_donor_all[tail_idx], model.bm_reference_Z, model.bm_reference_celltype,
            k=config.RETRIEVAL_K,
        )
        predicted_proportions = {ct: float(predicted_proportions.get(ct, 0.0)) for ct in model.generation.unique_types}
        total_p = sum(predicted_proportions.values())
        if total_p <= 0:
            raise ValueError(
                f"Donor {donor!r}: k-NN retrieval predicted zero proportion for every "
                "trained cell type -- this should not happen given a non-empty reference "
                "database. Check bm_reference_Z/bm_reference_celltype alignment."
            )
        predicted_proportions = {ct: p / total_p for ct, p in predicted_proportions.items()}

        blood_lib_per_ct: Dict[str, float] = {}
        d_ct = celltypes[d_mask]
        d_lib = lib_sizes[d_mask]
        for ct in model.generation.unique_types:
            vals = d_lib[d_ct == ct]
            if len(vals) > 0:
                blood_lib_per_ct[ct] = float(np.median(vals))

        gen = generate_per_celltype(
            model.generation, predicted_proportions=predicted_proportions,
            program_conditioning_per_ct=program_conditioning_per_ct,
            blood_lib_per_ct=blood_lib_per_ct, n_cells=n_gen, seed=seed,
            sample_counts=sample_counts,
        )

        gen_adata = ad.AnnData(
            X=gen["X"],
            obs=pd.DataFrame(
                {"cell_type": gen["cell_type"], "donor_id": donor, "source": "prism_generated"},
                index=[f"{donor}_gen_{i:05d}" for i in range(len(gen["cell_type"]))],
            ),
            var=pd.DataFrame(index=gen["gene_names"]),
        )
        gen_adata.obsm["X_prism"] = gen["Z"]

        umap_path = os.path.join(umap_out_dir, f"{donor}_umap.png") if umap_out_dir else None
        report = build_report(
            X_gen=gen["X"], Z_gen=gen["Z"], cell_type_gen=gen["cell_type"],
            ref_umap=model.reference_umap(seed=seed) if umap_path else None,
            umap_out_path=umap_path,
        )

        result.per_donor[donor] = DonorResult(
            donor_id=donor, n_cells_total=n_total, n_bonemarrowlike=n_bml,
            bonemarrowlike_threshold_percentile=config.MARROWLIKE_PERCENTILE,
            bonemarrowlike_threshold_zscore=bonemarrowlike_threshold_zscore,
            celltype_proportions=proportions,
            program_scores_donor=P_bml.mean(axis=0).astype(np.float32) if n_bml > 0 else fallback_vec,
            program_scores_per_cell=P_all[d_mask],
            generated_adata=gen_adata, report=report,
        )

    return result


def print_training_config(model: PRISMModel) -> Dict:
    """Print and return training_config.json -- the encoder/NMF/flow/decoder training hyperparameters."""
    path = os.path.join(model.artifacts_dir, "training_config.json")
    with open(path) as f:
        cfg = json.load(f)
    print(json.dumps(cfg, indent=2))
    return cfg
