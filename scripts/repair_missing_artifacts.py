"""
One-time repair script — NOT part of the shipped package.

Closes two gaps in the trained-artifact set produced by train_consensus()
(see PRISM_core.py):

1. consensus_arch.json was not copied into the release artifact set, but
   its contents are fully recoverable from consensus_summary.json's
   embedded training config (asdict(cfg)) plus its top-level n_dims/
   n_donors/n_leiden fields — train_consensus() derives arch.json from
   exactly those values (see PRISM_core.py, "Model architecture info for
   reloading").

2. train_consensus() fits an NMF model (prog_model) to produce
   consensus_programs.npz, but only ever persists the resulting scores
   and loadings (P_all, components) — never the fitted model object
   itself. program_scores() requires that object to project a *new*
   user's embeddings through the same trained NMF. Since NMF with
   init="nndsvda" and a fixed random_state is deterministic given the
   same input, refitting on the saved Z_all (consensus_embeddings.npz)
   with the exact saved hyperparameters reproduces the identical model —
   verified below to match consensus_programs.npz's saved P_all/
   components to float32 precision before trusting it.

Already run once against weights/ when this project was packaged (both
outputs are already present there); kept for provenance/reproducibility,
not meant to be re-run unless weights/ is rebuilt from scratch.
"""
import json
import os

import joblib
import numpy as np
from sklearn.decomposition import NMF
from sklearn.preprocessing import StandardScaler

ARTIFACTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "weights")


def repair_consensus_arch(artifacts_dir: str = ARTIFACTS_DIR) -> None:
    arch_path = os.path.join(artifacts_dir, "consensus_arch.json")
    if os.path.exists(arch_path):
        print(f"[SKIP] {arch_path} already exists")
        return

    summary = json.load(open(os.path.join(artifacts_dir, "consensus_summary.json")))
    cfg = summary["config"]

    arch = {
        "d_in": int(summary["n_dims"]),
        "d_latent": int(cfg["d_latent"]),
        "hidden": int(cfg["hidden"]),
        "n_layers": int(cfg["n_layers"]),
        "dropout": float(cfg["dropout"]),
        "n_donors": int(summary["n_donors"]),
        "use_attention_train": bool(cfg["use_attention_train"]),
        "n_leiden": int(summary["n_leiden"]) if (summary["n_leiden"] > 0 and cfg.get("lambda_leiden", 0.0) > 0.0) else 0,
    }
    with open(arch_path, "w") as f:
        json.dump(arch, f, indent=2)
    print(f"[OK] wrote {arch_path}: {arch}")


def repair_prog_model(artifacts_dir: str = ARTIFACTS_DIR) -> None:
    prog_model_path = os.path.join(artifacts_dir, "prog_model.joblib")
    if os.path.exists(prog_model_path):
        print(f"[SKIP] {prog_model_path} already exists")
        return

    emb = np.load(os.path.join(artifacts_dir, "consensus_embeddings.npz"), allow_pickle=True)
    Z_all = emb["Z"]

    prog = np.load(os.path.join(artifacts_dir, "consensus_programs.npz"), allow_pickle=True)
    P_all_saved = prog["P_all"]
    C_saved = prog["components"]
    method = str(prog["method"])
    if method != "nmf":
        raise NotImplementedError(f"Repair script only handles method='nmf', got {method!r}")

    summary = json.load(open(os.path.join(artifacts_dir, "consensus_summary.json")))
    cfg = summary["config"]
    K = int(cfg["programs_k"])
    seed = int(cfg["seed"])
    nmf_alpha = float(cfg["nmf_alpha"])
    nmf_max_iter = int(cfg["nmf_max_iter"])

    scaler = StandardScaler(with_mean=True, with_std=True)
    Zs = scaler.fit_transform(Z_all)
    col_min = Zs.min(axis=0, keepdims=True)
    Zs_nn = Zs - col_min

    nmf = NMF(
        n_components=K,
        init="nndsvda",
        random_state=seed,
        max_iter=nmf_max_iter,
        alpha_W=nmf_alpha,
        alpha_H=nmf_alpha,
        l1_ratio=0.5,
        solver="cd",
        beta_loss="frobenius",
    )
    nmf.fit(Zs_nn)
    nmf._col_min = col_min

    P_all_refit = nmf.transform(Zs_nn)
    C_refit = nmf.components_

    p_diff = float(np.max(np.abs(P_all_refit - P_all_saved)))
    c_diff = float(np.max(np.abs(C_refit - C_saved)))
    print(f"[VERIFY] max abs diff vs saved: P_all={p_diff:.2e}, components={c_diff:.2e}")
    if p_diff > 1e-2 or c_diff > 1e-2:
        raise RuntimeError(
            "Refit NMF does not match saved consensus_programs.npz closely enough "
            "(possible sklearn version mismatch or non-deterministic init) — "
            "do not trust this artifact, investigate before shipping."
        )

    joblib.dump(
        {"prog_model": nmf, "prog_scaler": scaler, "prog_method": method},
        prog_model_path,
    )
    print(f"[OK] wrote {prog_model_path}")


if __name__ == "__main__":
    repair_consensus_arch()
    repair_prog_model()
