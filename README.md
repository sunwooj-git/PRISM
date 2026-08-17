# PRISM

**PR**ogram-conditioned **I**nference of donor-**S**pecific **M**arrow

PRISM infers donor-specific bone marrow (BM) biology from peripheral blood
single-cell RNA-seq data. Bone marrow scRNA-seq is scarce (sampling it is
invasive); peripheral blood is accessible and biologically continuous with
marrow — many cell types are shared, and blood carries latent signatures of
marrow biology. PRISM identifies bone marrow-like cells within a donor's own
blood, extracts interpretable transcriptional programs, and generates
donor-specific synthetic BM cells — in contrast to methods that target
population-level realism rather than preserving individual donor identity.

**Pipeline**: a tissue-classifying encoder (blood vs. bone marrow, built on
an scVI joint embedding) produces a per-cell "marrowness" score → the top
decile of a donor's own blood by marrowness is treated as bone marrow-like →
those cells are projected onto 5 trained NMF transcriptional programs → a
conditional normalizing flow + negative-binomial gene decoder generate
synthetic bone marrow cells conditioned on that donor's own program scores
and cell types.

The model is trained once, shipped with fixed weights. Running PRISM on your
own data is inference only — no training happens as part of normal use.

## Installation

```bash
git clone https://github.com/sunwooj-git/PRISM.git
cd PRISM
pip install -e .
```

Requires Python >=3.10 (scvi-tools' dependency floor). Trained weights
(~150MB) download automatically from PRISM's Zenodo deposit the first time
you call `prism.load_model()`, and are cached under `~/.cache/prism/weights`
(override with the `PRISM_WEIGHTS_DIR` environment variable, or pass
`local_dir=` directly to point at a local copy).

## Quickstart

```python
import anndata as ad
import prism

model = prism.load_model()
adata = ad.read_h5ad("your_blood_data.h5ad")

result = prism.run_inference(model, adata, donor_key="person_id")

donor = next(iter(result.per_donor.values()))
donor.n_bonemarrowlike, donor.n_cells_total          # bone marrow-like cell counts
donor.bonemarrowlike_threshold_percentile            # e.g. 90.0 -- top decile of blood by marrowness
donor.bonemarrowlike_threshold_zscore                # the raw marrow_z cutoff that percentile is, for this model
donor.celltype_proportions      # Output 1: bone marrow-like cell-type breakdown
donor.program_scores_donor      # Output 2: this donor's P1-P5 program scores
donor.generated_adata           # Output 3: 3,000 synthetic bone marrow-like cells
donor.report                    # Output 4: composition + read-count stats + UMAP overlay
prism.print_training_config(model)  # bonus: training hyperparameters
```

A runnable version of this, plus a small synthetic demo dataset, is in
[`examples/`](examples/) — see `examples/quickstart.py` and
`examples/toy_sample.h5ad`.

## Input requirements

PRISM accepts exactly one input: a blood scRNA-seq `AnnData`.

| Field | Requirement | Notes |
|---|---|---|
| `adata.X` (or a named layer, via `counts_layer=`) | **Raw counts**, not pre-normalized | PRISM applies all required normalization internally (deterministic, no auto-detection). Pre-normalizing first will silently produce wrong marrowness scores and generated output. |
| `adata.var_names` | Gene symbols | Needs reasonable overlap with PRISM's trained ~10,457-gene panel; overlap is reported (`Found X% reference vars in query data`) during the scVI embedding step. |
| `adata.obs[donor_key]` | A donor identifier column (default name `"person_id"`) | Required. Program scores and generation are computed per donor; multiple donors in one call are handled independently. |
| `adata.obs["celltype_coarse"]` | Optional | If present, used directly (validated against PRISM's trained category set: `b_cell, dendritic, erythroid, macrophage, megakaryocyte, monocyte, neutrophil, nk_cell, plasma_cell, progenitor, t_cell`). If absent, PRISM runs CellTypist (`Immune_All_Low`, majority voting) internally and maps its output onto that same set — you never need to run CellTypist yourself. |
| `adata.obs["total_counts"]` / `"n_counts"` (or pass `library_size_key=`) | Recommended if `adata.X`/`counts_layer` genes have already been reduced from your full transcriptome (e.g. an HVG or targeted panel) | Generation's per-cell-type library size is this value times a fixed trained scaling factor. If it's computed by summing an already gene-reduced `adata.X` instead of the full transcriptome, it under-estimates true library size and generated cells' read counts come out proportionally too low — confirmed directly against a real donor (summing a ~10,457-gene-reduced panel produced counts ~30-40% low; using the donor's true precomputed total undid it). If `adata.X`/`counts_layer` already holds your full transcriptome, no action needed — PRISM falls back to summing it automatically. |

## Repo layout

- **`prism/`** — the installable package (inference only: encoder, scVI
  embedding, NMF program projection, flow + gene decoder generation,
  CellTypist labeling, reporting).
- **`examples/`** — `quickstart.py`, `toy_sample.h5ad` (synthetic single-donor
  demo data), `make_toy_sample.py` (regenerates it).
- **`scripts/`** — maintainer-only utilities (`repair_missing_artifacts.py`
  — see its docstring for what it fixed and why, kept for provenance).
- **`weights/`** — local staging copy of the trained model artifacts (not
  committed to git — see `.gitignore`; distributed via Zenodo).

The original training/evaluation code (`paper/`: encoder + NMF training,
flow/gene-decoder training, k-NN retrieval baseline, evaluation metrics) is
kept locally for reproducing the paper's results but is not part of this
repo. Not imported by `prism/` and not needed to run inference.

## Known limitations / design notes

- Generated expression is sampled from the trained negative-binomial
  decoder (mean + dispersion), not literally observed counts — realistic in
  distribution, not a real cell.
- Per-cell-type program-score conditioning follows a 3-tier fallback that
  matches `paper/bm_generation_v6.py`'s original `generate_bm_twostage_per_ct`
  exactly: (1) the donor's own mean whenever they have at least one bone
  marrow-like cell of that type (verified this locates generated cells
  correctly even from single-digit samples; a low-sample-size note is
  printed below `prism.config.MIN_CELLS_PER_TYPE_FOR_CONDITIONING` cells,
  informational only), (2) the real reference population's per-cell-type
  mean for a type the donor has zero cells of, (3) the donor's own
  cross-type tail mean (over their full blood population, not just the
  bone marrow-like-gated subset) as a last resort only.
- Output 1 (`celltype_proportions`) is the donor's own direct, empirical
  bone marrow-like cell-type composition. Generation itself (Output 3/4)
  uses a *separate* proportion estimate instead: k-NN retrieval against
  the real reference bone marrow cohort (k=20, matching the original
  pipeline's `paper/bm_retrieval_v4.py`/`generate_bm_from_blood_v3`
  exactly), which pools each of the donor's own top-10%-marrowness blood
  cells' nearest real reference neighbors and tallies their real cell
  types. This smooths over a donor's own sparse per-type label counts --
  a cell type absent from a donor's own CellTypist labels can still get
  nonzero generated proportion if cells embedding near their bone
  marrow-like cells are common in the reference (verified directly: real
  patient8 blood only has direct labels for 3 of the 11 trained types
  among its bone marrow-like cells, but retrieval-based generation
  produces 5, each landing on its correct real reference cluster).
- The CellTypist fine-to-coarse label mapping (`prism/_celltypist_map.py`'s
  `_build_coarse_map` / `_COARSE_RULES`) is the project's own rule, not
  independently re-verified in this workspace against the exact CellTypist
  version/labels used when the training data was annotated. Its "ILC" rule
  also targets a category (`innate_lymphoid`) outside PRISM's trained 11 —
  any cell landing there is caught by the hard-error check rather than
  silently reaching generation, but review it if ILC cells are common in
  your data.
- `weights/consensus_embeddings.npz` and `consensus_blood_reference.npz`
  (full training-cohort diagnostic dumps) are intentionally excluded from
  the download set `prism.load_model()` fetches — nothing in the inference
  path reads them.

## Citation

If you use PRISM, please cite:

```
[paper citation — add on publication]
```

## License

MIT — see [`LICENSE`](LICENSE).
