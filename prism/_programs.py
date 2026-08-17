"""
Projects encoder embeddings into the 5 trained NMF transcriptional programs.

Users are always scored through the single fixed NMF model shipped with the
package (prog_model.joblib) -- program identity ("program 3") is therefore
already consistent across every user by construction, since everyone shares
the same fitted model; there is no per-user refit. match_components_hungarian
is used only as a defensive integrity check (does the loaded prog_model
actually reproduce consensus_programs.npz's saved reference loadings?), not
as a per-user relabeling step.
"""
import json
import os
from dataclasses import dataclass
from typing import Dict

import joblib
import numpy as np
from scipy.optimize import linear_sum_assignment


def _row_norm(A: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(A, axis=1, keepdims=True)
    return A / np.maximum(n, 1e-12)


def match_components_hungarian(A: np.ndarray, B: np.ndarray) -> Dict[str, np.ndarray]:
    """Match rows of A to rows of B by maximizing absolute correlation."""
    A0, B0 = _row_norm(A), _row_norm(B)
    C = np.abs(A0 @ B0.T)
    cost = 1.0 - C
    r, c = linear_sum_assignment(cost)
    perm = c[np.argsort(r)]
    corr = C[np.arange(C.shape[0]), perm]
    return {"perm": perm.astype(np.int64), "corr": corr.astype(np.float64), "C": C.astype(np.float64)}


def program_scores(Z: np.ndarray, prog_model, prog_scaler, prog_method: str = "nmf") -> np.ndarray:
    """Project embeddings Z [N, d_latent] into program space -> [N, K]."""
    Zs = prog_scaler.transform(Z) if prog_scaler is not None else Z
    if prog_method == "nmf":
        col_min = prog_model._col_min
        Zs_nn = np.maximum(Zs - col_min, 0.0)
        return prog_model.transform(Zs_nn).astype(np.float32)
    return prog_model.transform(Zs).astype(np.float32)


@dataclass
class ProgramBundle:
    prog_model: object
    prog_scaler: object
    prog_method: str
    reference_components: np.ndarray  # [K, d_latent], from consensus_programs.npz
    match_corr: np.ndarray            # per-program correlation vs. reference (integrity check)


def load_programs(artifacts_dir: str) -> ProgramBundle:
    bundle = joblib.load(os.path.join(artifacts_dir, "prog_model.joblib"))
    prog = np.load(os.path.join(artifacts_dir, "consensus_programs.npz"), allow_pickle=True)
    C_ref = prog["components"]

    match = match_components_hungarian(bundle["prog_model"].components_, C_ref)
    if not np.array_equal(match["perm"], np.arange(len(match["perm"]))) or np.min(match["corr"]) < 0.99:
        raise RuntimeError(
            "Loaded prog_model.joblib does not reproduce consensus_programs.npz's "
            f"reference program loadings (perm={match['perm']}, corr={match['corr']}). "
            "Program identity (P1..P5) would not be consistent with the trained "
            "reference -- refusing to proceed rather than silently mislabel programs."
        )

    return ProgramBundle(
        prog_model=bundle["prog_model"],
        prog_scaler=bundle["prog_scaler"],
        prog_method=bundle.get("prog_method", "nmf"),
        reference_components=C_ref,
        match_corr=match["corr"],
    )
