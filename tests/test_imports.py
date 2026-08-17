"""Always-runnable sanity checks that don't need the trained weights."""
import prism
from prism import config


def test_public_api_importable():
    assert callable(prism.load_model)
    assert callable(prism.run_inference)
    assert callable(prism.print_training_config)


def test_artifact_file_list_has_no_duplicates():
    assert len(config.ARTIFACT_FILES) == len(set(config.ARTIFACT_FILES))


def test_celltypist_coarse_map():
    import numpy as np

    from prism._celltypist_map import _build_coarse_map

    labels = np.array(["Naive B cells", "Classical monocytes", "NKT cells", "Some unrecognized label"])
    coarse = _build_coarse_map(labels)
    assert coarse["Naive B cells"] == "b_cell"
    assert coarse["Classical monocytes"] == "monocyte"
    assert coarse["NKT cells"] == "t_cell"  # specific-before-general: NKT, not NK
    assert coarse["Some unrecognized label"] == "Some unrecognized label"  # unmapped -> kept as-is


def test_labels_to_prism_categories_flags_unmapped():
    import numpy as np

    from prism._celltypist_map import labels_to_prism_categories

    trained = ["b_cell", "monocyte", "t_cell"]
    raw = np.array(["Naive B cells", "Classical monocytes", "Epithelial cells", "ILC1"])
    mapped = labels_to_prism_categories(raw, trained)

    assert mapped[0] == "b_cell"
    assert mapped[1] == "monocyte"
    # "Epithelial cells" has no rule at all; "ILC1" maps to "innate_lymphoid",
    # which isn't in `trained` either -- both must come back as unmapped, not
    # silently pass through as some other trained category.
    assert mapped[2] == "__unmapped__"
    assert mapped[3] == "__unmapped__"
