"""Package-wide constants and defaults for PRISM inference."""
from typing import Optional

# Canonical generation size used throughout the project (paper + package).
N_GEN_DEFAULT = 3000

# Bone-marrow-like threshold, expressed as a percentile of the trained
# blood cohort's calibrated marrowness z-score distribution (consensus_
# blood_reference.npz's marrow_z_blood). Matches the paper's canonical
# "thres90_q0.10" run (top decile of blood = bone marrow-like).
MARROWLIKE_PERCENTILE = 90.0

# Informational only: below this many of a donor's own bone marrow-like
# cells of a given type, pipeline.run_inference prints a low-sample-size
# warning for that type's conditioning vector -- it does NOT discard the
# vector. An earlier version used this as a hard cutoff (falling back
# straight to the cross-type pooled tail mean below it), but that pooled
# fallback is dominated by whichever cell type is most common among the
# donor's bone marrow-like cells; verified directly that even a single-
# digit sample of a type's own cells conditions the flow correctly, while
# the pooled fallback pulled a just-under-threshold type toward the
# dominant type's region instead. Matches the original training pipeline's
# `min_bone_marrowlike_gate` default (paper/bm_generation_v6.py).
MIN_CELLS_PER_TYPE_FOR_CONDITIONING = 10

# Per-cell-type program-score conditioning fallback order, matching
# paper/bm_generation_v6.py's generate_bm_twostage_per_ct exactly (verified
# against that source, not just approximated):
#   1. the donor's own mean, whenever they have >=1 bone marrow-like cell
#      of that type (see MIN_CELLS_PER_TYPE_FOR_CONDITIONING above)
#   2. the REAL reference population's per-cell-type mean program score
#      (computed once from consensus_bm_reference.npz + bm_reference_
#      celltypes.npz -- see PRISMModel.bm_program_means_per_ct), for a type
#      with zero bone marrow-like cells of its own but a defined reference
#   3. this donor's own global tail-mean (below), pooled across ALL their
#      OWN cell types -- last resort only, if even the reference lacks
#      that type (shouldn't happen given 280k+ reference cells span all 11
#      trained categories, but kept for robustness)
#
# FALLBACK_TAIL_Q is also reused as the retrieval query-set fraction for
# Stage 1 (predicted cell-type proportions, see prism._retrieval) -- in the
# original pipeline both this fallback's tail and the k-NN retrieval query
# are the SAME top-tail_q-fraction selection over the donor's ENTIRE blood
# population (not just their bone-marrow-like-gated cells), by marrowness,
# not a coincidence of reusing one constant for two different things.
FALLBACK_TAIL_Q = 0.10

# k-NN retrieval neighbors per query cell for Stage 1 (predicted cell-type
# proportions) -- matches the trained/canonical value throughout the
# original pipeline (paper/bm_retrieval_v4.py, paper/bm_generation_v6.py).
RETRIEVAL_K = 20

DEVICE_DEFAULT = "cuda"  # falls back to cpu automatically, see _utils.resolve_device

# Weights are distributed via a Zenodo deposit (download-on-first-use, see
# prism._artifacts.get_artifact_dir). https://doi.org/10.5281/zenodo.21982005
# -- currently under restricted access pending publication, so downloads
# via get_artifact_dir will 403 for unauthenticated users until it's made
# public; local_dir= / PRISM_WEIGHTS_DIR remain the working option until then.
ZENODO_RECORD_ID: Optional[str] = "21982005"

# Artifact filenames expected in the weights bundle (see prism._artifacts).
ARTIFACT_FILES = [
    "consensus_model.pt",
    "consensus_arch.json",
    "consensus_summary.json",
    "consensus_programs.npz",
    "prog_model.joblib",
    "consensus_bm_reference.npz",
    "bm_reference_celltypes.npz",
    "blood_marrowz_ref.npy",
    "scvi_model.pt",
    "flow_celltype_model.pt",
    "generative_config.json",
    "gene_decoder.pt",
    "gene_decoder_config.json",
    "training_config.json",
]
