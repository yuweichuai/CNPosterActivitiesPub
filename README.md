# Community Notes: peer-review replication package

This package contains the 15 analysis notebooks from `1_1_data_summary.ipynb` through `5_post_engagement.ipynb`, their analysis helpers, and the processed data needed to run them. Data collection, API access, classifier execution, and preprocessing before these analyses are outside the package. No API credentials or external project folders are required.

## Contents and execution order

| Notebook | Analysis                                                                             |
| --- |--------------------------------------------------------------------------------------|
| `1_1_data_summary.ipynb` | Sample description, balance, correction recurrence and follow-up                     |
| `1_2_pre_correction_nmr_count.ipynb` | Descriptive summaries of processed prior-NMR histories                               |
| `2_1_count_analysis_daily.ipynb` | Daily posting models and trajectories                                                |
| `2_2_count_analysis_weekly.ipynb` | Weekly posting, sensitivity analyses, alternative estimators and matching            |
| `2_3_conditional_corrections.ipynb` | Responses by prior correction history                                                |
| `2_4_add_correction.ipynb` | First corrections with subsequent corrections included and additional specifications |
| `3_1_count_analysis_he_features.ipynb` | Joint interactions, subgroup models and multiplicity adjustment                      |
| `3_2_count_analysis_he_prior_nmr.ipynb` | Heterogeneity by prior NMR history                                                   |
| `3_3_count_analysis_corrected_quality.ipynb` | Heterogeneity by correction note score                                               |
| `3_4_count_analysis_deleted_posts.ipynb` | Comparison of events associated with deleted and accessible posts                    |
| `3_5_count_analysis_moral_outrage.ipynb` | Reply outrage and posting responses                                                  |
| `4_1_content_analysis.ipynb` | Content matching, production and composition                                         |
| `4_2_content_analysis_he_high_mis.ipynb` | Content outcomes among accounts with high baseline misleadingness                    |
| `4_3_content_analysis_he_prior_nmr.ipynb` | Content outcomes among accounts with repeated prior NMR notes                        |
| `5_post_engagement.ipynb` | Subsequent per-post repost and reply engagement                                      |

`utils.py` contains the analysis helpers. `review_setup.py` handles portable paths and Stata initialization. `data_manifest.json` lists every supplied dataset, its columns, row count, provenance, and SHA-256 checksum. `DATA_DICTIONARY.md` explains the fields and transformations. `SHA256SUMS.txt` records checksums for the distributed package files.

## Environment

The package was prepared using Python 3.11.3 and Stata/MP 19.5 on macOS. A licensed Stata installation with PyStata is required for notebooks 2–5. The descriptive notebooks 1_1 and 1_2 do not initialize Stata. Python package versions are listed in `requirements.txt`.

Create an environment and install the requirements:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m ipykernel install --user --name cn-review --display-name "CN review"
```

On Windows, activate the environment with `.venv\Scripts\activate`. Set `STATA_PATH` to your Stata installation directory and `STATA_EDITION` to `mp`, `se`, or `be` as appropriate. For example, in a Unix shell:

```bash
export STATA_PATH="/path/to/Stata"
export STATA_EDITION="mp"
```

If `STATA_PATH` is unset, the setup helper checks common installation locations. It does not include a machine-specific account path. PyStata and `sfi` are supplied with Stata; do not install an unrelated package named `stata` or `sfi` from PyPI.

The Stata packages used in the local environment were:

| Command | Version |
| --- | --- |
| `ppmlhdfe` | 2.3.3, 02 Nov 2025 |
| `reghdfe` | 6.12.3, 08 Aug 2023 |
| `ftools` | 2.49.1, 08 Aug 2023 |
| `sdid` | 2.0.6, 09 Jun 2026 |
| `honestdid` | 1.3.1, 16 Apr 2024 |
| `coefplot` | 1.8.6, 22 Feb 2023 |

Run `setup_stata.do` in Stata if these commands are missing. That setup file installs missing commands from SSC and leaves existing installations unchanged; downloads may supply newer versions than those above. Bootstrap and optimization results can vary with software versions. The original model commands, seeds and replication counts are retained.

## Run the analyses

Start from this folder:

```bash
python validate_package.py
python check_environment.py
python run_analysis.py --kernel cn-review
```

To run one notebook:

```bash
python run_analysis.py 4_1_content_analysis.ipynb --kernel cn-review
```

Alternatively, open the notebooks in JupyterLab, select the configured kernel, and run each from a fresh kernel in numeric order. The runner uses a separate kernel for every notebook and stops if execution fails. Bootstrap and HonestDiD cells can take substantially longer than the main regressions; the default cell timeout is unlimited.

Figures, regenerated matching datasets, and executed notebooks are written under `outputs/`. The supplied notebooks have no saved outputs or execution history. The runner does not overwrite them or any supplied input data.

Frozen content-matched event membership is included so that 4_2 and 5 can also run independently. Notebook 4_1 re-estimates matching and asserts that its selected events match the frozen membership. Its regenerated matched dataset is saved under `outputs/data/`.
