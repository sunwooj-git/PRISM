"""
Smoke test: runs the full inference pipeline against examples/toy_sample.h5ad
and checks that all five I/O-contract outputs come back well-formed.

Requires local model weights -- point PRISM_WEIGHTS_DIR at a directory
containing them (see prism/_artifacts.py) or place them under weights/ at
the repo root. Skipped otherwise, since CI shouldn't require downloading
~150MB of weights (and running a real scVI fine-tune step) on every push;
wire PRISM_WEIGHTS_DIR to a cached/pre-fetched location in CI to enable it.
"""
import os

import anndata as ad
import pytest

import prism

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_WEIGHTS_DIR = os.path.join(HERE, "..", "weights")

WEIGHTS_DIR = os.environ.get("PRISM_WEIGHTS_DIR", DEFAULT_WEIGHTS_DIR)
HAVE_WEIGHTS = os.path.exists(os.path.join(WEIGHTS_DIR, "scvi_model.pt"))


@pytest.mark.skipif(not HAVE_WEIGHTS, reason="local model weights not available")
def test_quickstart_end_to_end():
    model = prism.load_model(local_dir=WEIGHTS_DIR)
    adata = ad.read_h5ad(os.path.join(HERE, "..", "examples", "toy_sample.h5ad"))

    result = prism.run_inference(
        model, adata, donor_key="person_id", n_gen=200, seed=0,
    )

    assert set(result.per_donor.keys()) == {"toy_donor_001"}
    donor = result.per_donor["toy_donor_001"]

    # Output 1: bone marrow-like cell-type proportions
    assert donor.celltype_proportions.sum() > 0.99

    # Output 2: donor-specific program scores
    assert donor.program_scores_donor.shape == (5,)

    # Output 3: synthetic cells
    assert donor.generated_adata.n_obs == 200
    assert donor.generated_adata.n_vars == len(model.generation.gene_names)

    # Output 4: summary report
    assert donor.report.cell_type_composition.sum() > 0.99
    assert not donor.report.read_count_stats.empty


@pytest.mark.skipif(not HAVE_WEIGHTS, reason="local model weights not available")
def test_print_training_config():
    model = prism.load_model(local_dir=WEIGHTS_DIR)
    cfg = prism.print_training_config(model)
    assert "encoder_nmf_config" in cfg
    assert "flow_train_kwargs" in cfg
    assert "gene_decoder_kwargs" in cfg
