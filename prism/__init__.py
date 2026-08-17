"""
PRISM (PRogram-conditioned Inference of donor-Specific Marrow)

Infers donor-specific bone marrow biology from peripheral blood scRNA-seq.

Quickstart
----------
    import anndata as ad
    import prism

    model = prism.load_model()                 # downloads weights on first use
    adata = ad.read_h5ad("your_blood_data.h5ad")
    result = prism.run_inference(model, adata, donor_key="person_id")

    donor = next(iter(result.per_donor.values()))
    donor.celltype_proportions      # Output 1
    donor.program_scores_donor      # Output 2
    donor.generated_adata           # Output 3
    donor.report                    # Output 4 (composition, read-count stats, UMAP path)
    prism.print_training_config(model)  # bonus: training hyperparameters
"""
from .pipeline import (
    DonorResult,
    InferenceResult,
    PRISMModel,
    load_model,
    print_training_config,
    run_inference,
)

__all__ = [
    "PRISMModel",
    "DonorResult",
    "InferenceResult",
    "load_model",
    "run_inference",
    "print_training_config",
]

__version__ = "0.1.0"
