from __future__ import annotations

import os
import sys
import numpy as np
import pandas as pd

"""BODMAS dataset: download, load, temporal split, feature scaling.

BODMAS (Blue Hexagon Open Dataset for Malware AnalysiS; Yang et al.,
Deep Learning & Security Workshop @ IEEE S&P 2021) ships 134,435 real
Windows PE feature vectors (57,293 malware + 77,142 benign, Aug 2019 -
Sep 2020), 2381-dim, extracted with the same LIEF-based pipeline as
EMBER, plus per-sample timestamps. Labels: y=0 benign, y=1 malicious
(matches this repo's own convention already, no relabeling needed).

This module makes NO use of synthetic stand-in data anywhere. Every
number that comes out of it is computed on the real, downloaded BODMAS
feature vectors.

Download strategy (three tiers, tried in order until one works):
    1. gdown, official Google Drive folder (maintained by the authors)
    2. kagglehub, community mirror "dhoogla/bodmas"
    3. clear manual-download instructions printed to the user

Methodological note (why a temporal split, not a random one):
    Pendlebury et al., "TESSERACT: Eliminating Experimental Bias in
    Malware Classification across Space and Time" (USENIX Security 2019)
    show that i.i.d.-shuffled train/test splits on malware timelines
    systematically inflate reported accuracy, because they let the
    classifier "see the future" (variants of test-set malware families
    leak into training). We therefore reserve the chronologically LATEST
    slice of BODMAS as the test set and never touch it during trigger
    construction, embedding, attacks, or hyperparameter selection.
"""

BODMAS_DRIVE_FOLDER_ID = "1Uf-LebLWyi9eCv97iBal7kL1NgiGEsv_"
BODMAS_DRIVE_FOLDER_URL = f"https://drive.google.com/drive/folders/{BODMAS_DRIVE_FOLDER_ID}"
KAGGLE_DATASET = "dhoogla/bodmas"

NPZ_NAME = "bodmas.npz"
CSV_NAME = "bodmas_metadata.csv"


# ---------------------------------------------------------------------------
# Download (three tiers)
# ---------------------------------------------------------------------------
def _try_gdown(dest_dir: str):
    import gdown
    print("[data] Tier 1: gdown, official Google Drive folder…")
    gdown.download_folder(url=BODMAS_DRIVE_FOLDER_URL, output=dest_dir,
                          quiet=False, use_cookies=False)
    npz = os.path.join(dest_dir, NPZ_NAME)
    csv = os.path.join(dest_dir, CSV_NAME)
    if os.path.exists(npz) and os.path.exists(csv):
        return npz, csv
    # gdown sometimes nests the folder one level deep
    for root, _, files in os.walk(dest_dir):
        if NPZ_NAME in files and CSV_NAME in files:
            return os.path.join(root, NPZ_NAME), os.path.join(root, CSV_NAME)
    return None, None


def _try_kaggle(dest_dir: str):
    import kagglehub
    print("[data] Tier 2: kagglehub, community mirror dhoogla/bodmas…")
    path = kagglehub.dataset_download(KAGGLE_DATASET)
    npz = csv = None
    for root, _, files in os.walk(path):
        for f in files:
            fl = f.lower()
            if fl.endswith(".npz") and npz is None:
                npz = os.path.join(root, f)
            if fl.endswith(".csv") and "meta" in fl and csv is None:
                csv = os.path.join(root, f)
            if fl.endswith(".csv") and csv is None:
                csv = os.path.join(root, f)
    if npz is None:
        return None, None
    # Copy into dest_dir under the canonical names so the rest of the
    # pipeline doesn't care which tier succeeded.
    import shutil
    os.makedirs(dest_dir, exist_ok=True)
    npz_dst = os.path.join(dest_dir, NPZ_NAME)
    shutil.copy(npz, npz_dst)
    csv_dst = None
    if csv is not None:
        csv_dst = os.path.join(dest_dir, CSV_NAME)
        shutil.copy(csv, csv_dst)
    return npz_dst, csv_dst


_MANUAL_INSTRUCTIONS = f"""
[data] Automatic download failed via both gdown and kagglehub.

Please download manually (takes ~1 minute):
  1. Open: {BODMAS_DRIVE_FOLDER_URL}
  2. Download '{NPZ_NAME}' (~250 MB) and '{CSV_NAME}' (~12 MB)
  3. In Colab, upload them (folder icon on the left) into: {{dest_dir}}
     or run this in a cell:
         from google.colab import files
         files.upload()   # then move the two files into {{dest_dir}}
  4. Re-run this cell.

Alternative mirror (Kaggle, requires a Kaggle account):
  kaggle.com/datasets/dhoogla/bodmas
"""


def download_bodmas(dest_dir: str = "./bodmas_data"):
    """Ensure bodmas.npz + bodmas_metadata.csv exist under dest_dir.

    Returns (npz_path, csv_path). Raises RuntimeError with manual
    instructions if every automated path fails.
    """
    os.makedirs(dest_dir, exist_ok=True)
    npz_path = os.path.join(dest_dir, NPZ_NAME)
    csv_path = os.path.join(dest_dir, CSV_NAME)
    if os.path.exists(npz_path) and os.path.exists(csv_path):
        print(f"[data] Found cached BODMAS files in {dest_dir}, skipping download.")
        return npz_path, csv_path

    try:
        npz_path2, csv_path2 = _try_gdown(dest_dir)
        if npz_path2 and csv_path2:
            print("[data] gdown succeeded.")
            return npz_path2, csv_path2
        print("[data] gdown ran but did not produce both files; falling back.")
    except Exception as e:
        print(f"[data] gdown failed ({type(e).__name__}: {e}); falling back.")

    try:
        npz_path2, csv_path2 = _try_kaggle(dest_dir)
        if npz_path2 and csv_path2:
            print("[data] kagglehub succeeded.")
            return npz_path2, csv_path2
        print("[data] kagglehub ran but did not produce both files.")
    except Exception as e:
        print(f"[data] kagglehub failed ({type(e).__name__}: {e}).")

    raise RuntimeError(_MANUAL_INSTRUCTIONS.format(dest_dir=dest_dir))


# ---------------------------------------------------------------------------
# Load + validate
# ---------------------------------------------------------------------------
def _detect_timestamp_column(meta: pd.DataFrame):
    """BODMAS's own docs describe bodmas_metadata.csv only as '3 columns:
    SHA-256, when the sample first appeared, and malware family' without
    giving literal header strings, and mirrors occasionally differ. Rather
    than hard-code a guessed column name, detect the timestamp column by
    trying to parse each column as a datetime.

    The real file mixes tz-aware and tz-naive timestamp strings within the
    SAME column (e.g. "2016-10-30 15:59:55" next to rows with a "+00:00"
    suffix) -- pandas' format auto-inference locks onto whichever format
    it sees first and then throws on rows that don't match it. format=
    "mixed" parses each element independently instead of assuming one
    consistent format for the whole column; utc=True normalizes the
    tz-aware/naive mix to a single comparable UTC timeline.
    """
    for c in meta.columns:
        try:
            pd.to_datetime(meta[c], format="mixed", utc=True, errors="raise")
            return c
        except Exception:
            continue
    return None


def load_bodmas(npz_path: str, csv_path: str):
    """Load the real feature matrix, labels, and timestamps.

    Returns (X, y, timestamps) where timestamps is a pandas.DatetimeIndex
    aligned row-for-row with X (re-sorted ascending, defensively, even
    though the authors say the files already are).
    """
    data = np.load(npz_path)
    X = data["X"].astype(np.float32)
    y = data["y"].astype(np.int64)
    assert set(np.unique(y).tolist()) <= {0, 1}, \
        f"Unexpected label values in bodmas.npz: {np.unique(y)}"

    meta = pd.read_csv(csv_path)
    if len(meta) != X.shape[0]:
        raise ValueError(
            f"Row-count mismatch between {NPZ_NAME} (n={X.shape[0]}) and "
            f"{CSV_NAME} (n={len(meta)}). This usually means a mirror "
            f"served a stale/partial file; re-download.")

    ts_col = _detect_timestamp_column(meta)
    if ts_col is None:
        # Fall back to the authors' documented column order.
        meta.columns = ["sha256", "timestamp", "family"] + list(meta.columns[3:])
        ts_col = "timestamp"
    timestamps = pd.to_datetime(meta[ts_col], format="mixed", utc=True).values

    order = np.argsort(timestamps)
    X, y, timestamps = X[order], y[order], timestamps[order]

    print(f"[data] BODMAS loaded: {X.shape[0]} samples, {X.shape[1]} features, "
          f"{int((y == 1).sum())} malware / {int((y == 0).sum())} benign, "
          f"date range {pd.Timestamp(timestamps[0]).date()} .. "
          f"{pd.Timestamp(timestamps[-1]).date()}")
    return X, y, timestamps


# ---------------------------------------------------------------------------
# Temporal split
# ---------------------------------------------------------------------------
def temporal_split(X, y, timestamps, cfg, rng, n_owner=None, n_reference=None, n_test=None):
    """Chronological split: the LATEST `test_frac` slice is the test set
    (never used for training/trigger-construction/hyperparameter choice).
    The historical pool (everything before it) is randomly split into a
    disjoint owner-train / reference-train (both "see" the same era of
    malware, which is the realistic case for an owner vs. an independent
    contemporaneous auditor).

    Optional n_owner/n_reference/n_test subsample each split down for
    runtime control (subsampling preserves the temporal boundary: it never
    moves samples across the historical/test line, it only thins each
    side).
    """
    n = len(y)
    n_test_total = int(cfg.test_frac * n)
    cut = n - n_test_total
    hist_idx = np.arange(cut)
    test_idx = np.arange(cut, n)

    perm = rng.permutation(hist_idx)
    n_owner_avail = int(cfg.owner_frac_of_hist * len(perm))
    n_ref_avail = int(cfg.reference_frac_of_hist * len(perm))
    owner_idx = perm[:n_owner_avail]
    ref_idx = perm[n_owner_avail: n_owner_avail + n_ref_avail]

    if n_owner is not None and n_owner < len(owner_idx):
        owner_idx = rng.choice(owner_idx, size=n_owner, replace=False)
    if n_reference is not None and n_reference < len(ref_idx):
        ref_idx = rng.choice(ref_idx, size=n_reference, replace=False)
    if n_test is not None and n_test < len(test_idx):
        test_idx = rng.choice(test_idx, size=n_test, replace=False)

    splits = {
        "owner": (X[owner_idx], y[owner_idx]),
        "reference": (X[ref_idx], y[ref_idx]),
        "test": (X[test_idx], y[test_idx]),
    }
    print(f"[data] temporal split -> owner={len(owner_idx)}  "
          f"reference={len(ref_idx)}  test(future)={len(test_idx)}  "
          f"test starts at {pd.Timestamp(timestamps[cut]).date()}")
    return splits


# ---------------------------------------------------------------------------
# Feature scaling (RobustScaler, fit on owner-train ONLY -- no leakage)
# ---------------------------------------------------------------------------
class RobustFeatureScaler:
    """Median/IQR scaling, fit once on owner-train, applied to every split
    (reference, test, attacker data) so the L_inf PGD budget means the same
    thing everywhere. BODMAS ships raw, unnormalized features with very
    different per-feature ranges (e.g. file-size-like fields vs. 0/1
    flags); an un-scaled L_inf ball would be dominated by a handful of
    large-range dimensions.
    """

    def __init__(self, clip: float = 8.0):
        self.clip = clip
        self.median_ = None
        self.iqr_ = None

    def fit(self, X):
        self.median_ = np.median(X, axis=0)
        q75 = np.percentile(X, 75, axis=0)
        q25 = np.percentile(X, 25, axis=0)
        iqr = q75 - q25
        iqr[iqr < 1e-6] = 1.0   # degenerate (near-constant) columns
        self.iqr_ = iqr
        return self

    def transform(self, X):
        Z = (X - self.median_) / self.iqr_
        return np.clip(Z, -self.clip, self.clip).astype(np.float32)

    def fit_transform(self, X):
        return self.fit(X).transform(X)
