"""
Minimal quickstart: load a user's blood AnnData, run PRISM inference,
inspect the five outputs.

    python examples/quickstart.py
"""
import os

import anndata as ad

import prism

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    model = prism.load_model()  # add local_dir="/path/to/weights" to skip the Zenodo download
    adata = ad.read_h5ad(os.path.join(HERE, "toy_sample.h5ad"))

    result = prism.run_inference(
        model, adata, donor_key="person_id",
        umap_out_dir=os.path.join(HERE, "outputs"),
    )

    for donor_id, donor in result.per_donor.items():
        print(f"\n=== Donor: {donor_id} ===")
        print(f"Total cells: {donor.n_cells_total}, bone marrow-like: {donor.n_bonemarrowlike}")
        print(
            f"Bone marrow-like threshold: top {100 - donor.bonemarrowlike_threshold_percentile:.0f}% "
            f"of blood by marrowness (percentile={donor.bonemarrowlike_threshold_percentile}, "
            f"z-score cutoff={donor.bonemarrowlike_threshold_zscore:.3f})"
        )
        print("\n[Output 1] Bone marrow-like cell-type proportions:")
        print(donor.celltype_proportions)
        print("\n[Output 2] Donor-specific program scores (P1-P5):")
        print(donor.program_scores_donor)
        print(f"\n[Output 3] Generated synthetic cells: {donor.generated_adata.shape}")
        print("\n[Output 4] Generated cell-type composition:")
        print(donor.report.cell_type_composition)
        print("\n[Output 4] Per-cell-type read-count stats:")
        print(donor.report.read_count_stats)
        print(f"\n[Output 4] UMAP overlay figure: {donor.report.umap_figure_path}")

    print("\n[Bonus] Training configuration:")
    prism.print_training_config(model)


if __name__ == "__main__":
    main()
