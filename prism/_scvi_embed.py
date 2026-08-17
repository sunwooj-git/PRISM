"""
Maps a user's raw blood gene counts into the 32-dim scVI joint latent
space PRISM's tissue-classifying encoder was trained on.

Why this step exists: PRISM's encoder (prism._encoder.PRISMEncoderNet) does
not take gene expression as input. It was trained on top of an scVI model
that jointly embedded the full blood+bone-marrow training cohort (batch-
corrected across the 37 training donors). A donor's raw counts have to be
run through that same scVI model before the encoder can produce a
comparable embedding.

This uses scvi-tools' scArches-style query mapping (`load_query_data`),
which is robust to both genuinely new donors (extends the batch registry
for an unseen category) and donors already in the training cohort (reuses
the existing one) -- but does NOT fine-tune afterward, despite that being
scArches' usual next step. That's deliberate, not an oversight:

  This model's z_encoder was trained with NO batch/categorical covariate
  in its own input (verified directly: `z_encoder.encoder.n_cat_list ==
  []`, and its first Linear layer's in_features exactly equals the gene
  count, no extra concatenated columns) -- a deliberate architecture
  choice where the encoder is meant to be batch-INVARIANT by construction,
  with batch correction handled entirely on the decoder side instead.
  scArches' fine-tuning mechanism works by registering a gradient hook on
  each layer that zeroes every input except the columns belonging to the
  (extended) batch one-hot -- see scvi.nn.FCLayers.set_online_update_hooks
  and its `_hook_fn_weight`. When a layer has zero categorical input
  columns to begin with (True here for every z_encoder layer), that hook's
  "keep only the categorical columns" guard (`if categorical_dims > 0`)
  never fires, so it returns an all-zero gradient unconditionally. Verified
  directly against this exact model: z_encoder's parameters show EXACTLY
  0.0 change after 1, 3, and 150 fine-tune epochs alike, and skipping
  training entirely (just flipping the `is_trained_` flag scvi-tools
  otherwise requires) reproduces the same Z to floating-point noise
  (~3e-5). Fine-tuning here would cost several minutes per donor for zero
  effect on the embedding PRISM actually uses -- so it's skipped outright.
"""
import shutil
import tempfile
from pathlib import Path
from typing import Optional

import anndata as ad
import numpy as np
import torch


def _prepare_reference_dir(scvi_model_path: str, cache_dir: Optional[str] = None) -> Path:
    """scvi-tools' load_query_data() expects a directory containing model.pt."""
    if cache_dir is not None:
        ref_dir = Path(cache_dir)
        ref_dir.mkdir(parents=True, exist_ok=True)
        dst = ref_dir / "model.pt"
        if not dst.exists():
            shutil.copy(scvi_model_path, dst)
        return ref_dir
    ref_dir = Path(tempfile.mkdtemp(prefix="prism_scvi_ref_"))
    shutil.copy(scvi_model_path, ref_dir / "model.pt")
    return ref_dir


def embed_blood_counts(
    adata_counts: ad.AnnData,
    scvi_model_path: str,
    donor_key: str,
    counts_layer: Optional[str] = None,
    device: str = "cpu",
    seed: int = 0,
    reference_cache_dir: Optional[str] = None,
) -> np.ndarray:
    """
    Embed raw blood counts into PRISM's 32-dim scVI joint latent space.

    Parameters
    ----------
    adata_counts:
        AnnData with RAW counts (not normalized) in .X or counts_layer, and
        var_names as gene symbols. Only genes overlapping the trained scVI
        gene panel are used; scvi-tools handles the reindex/reference-vars
        alignment internally (missing genes are zero-filled, a
        `Found X% reference vars in query data` info line reports overlap).
    donor_key:
        adata_counts.obs column identifying the donor -- registered as
        scVI's batch covariate for API/registry compatibility (matching how
        the reference model was trained, batch_key="person_id"), but see
        the module docstring: this model's encoder never actually
        conditions on batch, so the specific donor id has no effect on the
        resulting embedding, known donor or brand new one alike.

    Returns
    -------
    Z: [n_cells, 32] float32 array, ready for prism._encoder.encode_cells.
    """
    import scvi  # deferred import: heavy, only needed for this step

    torch.manual_seed(seed)

    ref_dir = _prepare_reference_dir(scvi_model_path, reference_cache_dir)

    X = adata_counts.layers[counts_layer] if counts_layer is not None else adata_counts.X
    if hasattr(X, "toarray"):
        X = X.toarray()
    X = np.asarray(X, dtype=np.float32)

    if not np.allclose(X, np.round(X), atol=1e-3):
        raise ValueError(
            "prism.embed_blood_counts expects RAW counts (non-negative integers), "
            "but the input data does not look like raw counts. Do not pre-normalize "
            "before calling this function -- PRISM's scVI embedding step needs raw "
            "counts, and the package applies all normalization internally downstream."
        )

    query = ad.AnnData(X=X.copy())
    query.var_names = adata_counts.var_names
    query.layers["counts"] = X.copy()
    query.obs["person_id"] = adata_counts.obs[donor_key].astype(str).values
    query.obs["log1p_total_counts"] = np.log1p(X.sum(axis=1))

    scvi.model.SCVI.prepare_query_anndata(query, str(ref_dir))
    query_model = scvi.model.SCVI.load_query_data(query, str(ref_dir))
    query_model.to_device(device)
    # No .train() call -- see module docstring. get_latent_representation()
    # only requires this flag as a "don't query an untrained model" guard;
    # it isn't otherwise gated on training having happened.
    query_model.is_trained_ = True

    Z = query_model.get_latent_representation()
    return np.asarray(Z, dtype=np.float32)
