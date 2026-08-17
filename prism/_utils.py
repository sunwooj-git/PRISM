"""Small shared helpers -- device resolution and the one place normalization happens."""
import numpy as np
import torch


def resolve_device(requested: str = "cuda") -> str:
    if requested == "cuda" and not torch.cuda.is_available():
        return "cpu"
    return requested


def log1p_cp10k(X: np.ndarray) -> np.ndarray:
    """
    Deterministic log1p(counts-per-10k). Always applied -- no auto-detection
    of whether X "looks like" raw counts. Used only for the summary report's
    normalized views of generated expression (see prism._report); the
    scVI/encoder path never normalizes this way (scVI consumes raw counts
    directly, with its own internal library-size handling).
    """
    X = np.asarray(X, dtype=np.float64)
    lib = X.sum(axis=1, keepdims=True) + 1e-8
    return np.log1p((X / lib) * 1e4)


def assert_raw_counts(X: np.ndarray, name: str = "adata.X") -> None:
    sample = X[: min(X.shape[0], 200)]
    if hasattr(sample, "toarray"):
        sample = sample.toarray()
    sample = np.asarray(sample)
    if not np.allclose(sample, np.round(sample), atol=1e-3) or sample.max() <= 0:
        raise ValueError(
            f"{name} does not look like raw counts (expected non-negative "
            "integers). PRISM applies all required normalization internally -- "
            "do not pre-normalize input data."
        )
