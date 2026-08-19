"""
Trained-weight distribution: download-on-first-use from PRISM's Zenodo
deposit, cached locally so repeat runs don't re-download.

The pip package itself ships no model weights (~215MB total for the files
actually needed at inference time -- see config.ARTIFACT_FILES). Excluded
from that set are consensus_embeddings.npz (full-cohort diagnostic dump,
~170MB, not read by any inference-time code path) and consensus_blood_
reference.npz (~70MB, superseded at inference time by the calibration
parameters already stored in consensus_summary.json) -- both remain
available in the full training-artifact release for reproduction, just
not fetched by this loader.
"""
import os
import shutil
import ssl
import urllib.request
from pathlib import Path
from typing import Optional

import certifi

from . import config


def _default_cache_dir() -> Path:
    override = os.environ.get("PRISM_WEIGHTS_DIR")
    if override:
        return Path(override)
    return Path.home() / ".cache" / "prism" / "weights"


def _https_opener() -> urllib.request.OpenerDirector:
    # Explicit certifi CA bundle rather than the platform default: on
    # python.org-installed Python on macOS, urllib's default SSL context
    # doesn't pick up the system trust store, causing
    # CERTIFICATE_VERIFY_FAILED on first download -- confirmed directly
    # against this project's own Zenodo deposit, not a hypothetical.
    ctx = ssl.create_default_context(cafile=certifi.where())
    return urllib.request.build_opener(urllib.request.HTTPSHandler(context=ctx))


def get_artifact_dir(
    local_dir: Optional[str] = None,
    record_id: Optional[str] = None,
    force_download: bool = False,
) -> Path:
    """
    Return a local directory containing all files in config.ARTIFACT_FILES,
    downloading any that are missing from PRISM's Zenodo deposit.

    Parameters
    ----------
    local_dir:
        If given, use this directory instead of the default cache location.
        Existing files here are trusted and not re-downloaded (unless
        force_download=True); pass this to point at artifacts you already
        have on disk (e.g. during development, or before a Zenodo deposit
        exists).
    """
    target_dir = Path(local_dir) if local_dir is not None else _default_cache_dir()
    target_dir.mkdir(parents=True, exist_ok=True)

    missing = [f for f in config.ARTIFACT_FILES if force_download or not (target_dir / f).exists()]
    if not missing:
        return target_dir

    record_id = record_id or config.ZENODO_RECORD_ID
    if not record_id:
        raise FileNotFoundError(
            f"Missing {len(missing)} weight file(s) in {target_dir}, and no "
            "Zenodo record is configured yet (prism.config.ZENODO_RECORD_ID "
            "is unset). Point prism.load_model() at a local copy instead: "
            "local_dir='/path/to/weights', or set the PRISM_WEIGHTS_DIR "
            f"environment variable. Missing: {missing}"
        )

    print(
        f"[prism] Fetching {len(missing)} weight file(s) from "
        f"Zenodo record {record_id} -> {target_dir} (first use only)"
    )
    opener = _https_opener()
    for fname in missing:
        url = f"https://zenodo.org/records/{record_id}/files/{fname}?download=1"
        with opener.open(url) as response, open(target_dir / fname, "wb") as out:
            shutil.copyfileobj(response, out)

    return target_dir
