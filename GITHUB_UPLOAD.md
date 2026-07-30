# Publishing this repository on GitHub

## 1. Replace placeholders

Replace `<YOUR_USERNAME>` in:

- `README.md`
- `CITATION.cff`
- `notebooks/baw_bodmas_colab.ipynb`
- `paper/artifact_availability.tex`

Recommended repository name:

```text
baw-malware-watermark
```

## 2. Initialize and push

```bash
git init
git add .
git commit -m "Initial reproducibility release for BAW"
git branch -M main
git remote add origin https://github.com/<YOUR_USERNAME>/baw-malware-watermark.git
git push -u origin main
```

## 3. Before paper submission

- Remove notebook outputs.
- Do not commit BODMAS files.
- Run `pytest`.
- Run a smoke experiment.
- Commit the exact scripts used for the paper.
- Export per-seed results.
- Tag the submitted version:

```bash
git tag -a csonet2026-submission -m "Code associated with the CSoNet 2026 submission"
git push origin csonet2026-submission
```

For the camera-ready release, create a GitHub Release and archive the same commit on Zenodo to obtain a DOI.

## 4. Link placement in the manuscript

Best location: the end of **Experimental Methodology / Statistical Analysis and Reproducibility**.

Optional second mention: one sentence at the end of the contribution list in the Introduction.

Avoid placing a raw repository URL in the abstract.

If review is anonymous, do not use a personal GitHub repository. Use an anonymous artifact link or omit identifying metadata until the camera-ready stage.
