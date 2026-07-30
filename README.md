# BAW: Benign Adversarial Watermarking

Research code for **BAW: Benign Adversarial Watermarking for Ownership Verification of Machine Learning Malware Detectors**.

BAW studies black-box ownership verification for static, feature-based malware detectors under watermark-key exposure. The current implementation operates on BODMAS feature vectors. It is a **feature-space research prototype**: optimized trigger vectors are not claimed to correspond to valid, functionality-preserving PE files.

## Repository layout

```text
.
├── notebooks/
│   └── baw_bodmas_colab.ipynb       # lightweight Colab entry point
├── scripts/
│   ├── run_all.py                    # Track A + Track B
│   └── export_per_seed_csv.py        # export auditable per-seed results
├── src/baw/
│   ├── config.py                     # experiment configuration
│   ├── nn.py                         # NumPy MLP and optimization
│   ├── data_bodmas.py                # download, loading, temporal split, scaling
│   ├── baw.py                        # standard BAW
│   ├── baw_robust.py                 # robustness-aware trigger selection
│   ├── baseline.py                   # symmetric malware-to-benign baseline
│   ├── baseline_adi.py               # adapted Adi-style baseline
│   ├── attacks.py                    # fine-tuning, pruning, distillation
│   ├── attacks_adaptive.py           # Fine-Pruning and OOD diagnostics
│   ├── exploit.py                    # feature-space key-exposure evaluation
│   ├── surrogate.py                  # non-differentiable tree experiment
│   ├── stats_utils.py                # confidence intervals and paired tests
│   ├── main_real.py                  # experiment orchestration
│   └── figures_real.py               # plotting helpers
├── tests/
├── data/README.md
├── results/README.md
├── CITATION.cff
├── REPRODUCIBILITY.md
└── pyproject.toml
```

## Installation

```bash
git clone https://github.com/lab-secureai/baw.git
cd baw
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -e .
```

For development and tests:

```bash
python -m pip install -e ".[dev]"
pytest
```

## Run the experiments

Quick smoke run:

```bash
python scripts/run_all.py --quick
```

Full configuration used by the paper:

```bash
python scripts/run_all.py
```

Results are written to `results_real/` by the underlying configuration unless `--outdir` is supplied.

The full experiment is computationally heavier than the smoke run. Record the Git commit, Python version, dependency versions, and command for every paper result.

## Colab

Open `notebooks/baw_bodmas_colab.ipynb`, replace the repository URL placeholder, and run the cells in order. The notebook imports the package; the implementation is not duplicated inside a giant notebook cell.

## Dataset

The code uses BODMAS static PE feature vectors and timestamps. Dataset files are not committed to this repository. The loader tries the configured download sources and otherwise prints manual instructions.

## Reproducibility

See [REPRODUCIBILITY.md](REPRODUCIBILITY.md). For a public artifact release, include generated per-seed results rather than only aggregate table values.

## Scope and safety interpretation

- BAW triggers are generated from benign-origin feature vectors.
- The current code does not establish that an optimized feature vector maps to a valid PE file.
- `direct_exploit_fraction` is therefore a feature-space carrier-label proxy, not proof of a realizable executable exploit.
- Key-assisted evasion results apply only to the implemented attack family and are not a formal security guarantee.

## Citation

Will be updated after accepted

## License

MIT. Dataset licenses and terms remain separate.
