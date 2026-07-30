# Reproducibility checklist

## Before running

1. Record the Git commit:
   ```bash
   git rev-parse HEAD
   ```
2. Record the environment:
   ```bash
   python --version
   python -m pip freeze > results/environment.txt
   ```
3. Keep the exact dataset filenames and hashes:
   ```bash
   sha256sum bodmas_data/* > results/dataset_sha256.txt
   ```
4. Do not tune on the chronological test period.

## Recommended paper artifacts

Release the following with the camera-ready version:

- `results_real.json`;
- `track_a_per_seed.csv`;
- configuration dump;
- dataset hashes, not the dataset itself;
- environment file;
- exact command line;
- commit hash;
- scripts used to construct every table.

## Controls that should be exported

For each seed and scheme, retain:

- clean utility before and after embedding;
- pre-embedding trigger response;
- post-embedding trigger response;
- embedding lift;
- mean and maximum independent-model response;
- verification gap;
- watermark response and task utility after every removal attack;
- direct-carrier proxy;
- key-free and with-key evasion rates.

If a metric is not currently generated, add it before claiming full reproducibility of that result.

## Randomness

Track A uses seed-matched comparisons. Do not report only aggregate means; retain raw per-seed values so paired tests can be audited.

## Data separation

For a stronger BAW-robust evaluation, keep trigger-selection rehearsal data disjoint from the attacker data used for final removal evaluation.
