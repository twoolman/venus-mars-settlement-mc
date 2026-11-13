# Venus Aerostat Colony versus Mars Surface Colony: Monte Carlo and Neural Network Analysis

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24042380.svg)](https://doi.org/10.5281/zenodo.24042380)

This repository holds the R code, input parameter table and summary results for the study *A Venus Cloud-Level Aerostat Colony as a Superior Offworld Settlement Strategy to a Mars Surface Colony: A Monte Carlo and Neural Network Analysis* by Thomas Alan Woolman (On Target Technologies, Inc.).

The study builds comparable first-principles physical and economic models of a crewed aerostat colony floating at 54 to 60 km in the Venus cloud deck and a crewed surface colony on Mars. It propagates the uncertainty in 367 input parameters through Monte Carlo simulation, and it tests ten one-sided null hypotheses at an alpha of 0.05. An H2O deep learning network is fitted inside the simulation as a differentiable surrogate. It is refitted on nonparametric bootstrap resamples to give per-prediction perturbation sensitivities, a paired permuted-refit null arm gives each input its own p-value, and PERMANOVA (vegan adonis2) is run on the local sensitivity matrix.

## Results

D is defined so that D > 0 favors Venus in every test, and n = 20,000 Monte Carlo draws were used per test.

| Test | Response D | Median D | p_MC | Decision |
|---|---|---|---|---|
| RQ1 radiation | Mars minus Venus dose equivalent (mSv/yr) | 143.8 | 0.00015 | Reject H1-0 |
| RQ2 water and oxygen, city collector | Mars minus Venus delivered cost (USD/kg) | -32.1 | 0.639 | Fail to reject |
| RQ2 water and oxygen, section collector | Mars minus Venus delivered cost (USD/kg) | 19.6 | 0.353 | Fail to reject |
| RQ3 exports, rocket launch | Profit per kg exported (USD/kg) | -3,524 | 0.772 | Fail to reject |
| RQ3 exports, full launch chain | Profit per kg exported (USD/kg) | -4,364 | 0.791 | Fail to reject |
| RQ3 microbial protein | Earth-supplied minus Venus-delivered cost (USD/kg) | -1,103 | 0.699 | Fail to reject |
| RQ3 helium-3 and heavy water | Net co-product margin (USD/t acid-derived water) | -271,144 | 0.715 | Fail to reject |
| RQ4 colony economy, city collector | Venus minus Mars NPV (USD/colonist) | 227.1 million | 0.024 | Reject H4-0 |
| RQ4 colony economy, section collector | Venus minus Mars NPV (USD/colonist) | 229.4 million | 0.022 | Reject H4-0 |
| RQ5 gravity, 5% materiality margin | Venus minus Mars productive hours (h/colonist-yr) | 259.2 | 0.304 | Fail to reject |

The full results, with intervals, holdout R2, significant-input counts and adonis2 statistics, are in `results/hypothesis_summary.csv`.

## Repository contents

| Path | Contents |
|---|---|
| `code/venus_mars_mc_ann_v26.R` | The complete analysis script (version 26), which produced every result in the paper |
| `data/venus_mars_v26_parameter_table.csv` | The 367 uncertain inputs, each with distribution, bounds, units, basis tag and source |
| `data/crop_price_table.csv` | Crop price inputs used by the RQ3 export models |
| `results/hypothesis_summary.csv` | One row per test: median D, 95% interval, P(D > 0), p_MC, decision, surrogate holdout R2 and adonis2 results |
| `tables/*_variable_tests.csv` | Per-input sensitivities, permuted-refit p-values and significance flags for each test |
| `results/adonis2/*_adonis2.txt` | PERMANOVA tables for each test |
| `figures/` | Distribution of D and input sensitivity plots for each test |
| `results/sessionInfo.txt` | R session information from the full run |

Each parameter's basis tag marks it as sourced from the literature [S], derived from physics within the model [D] or assumed [A], and 41 parameters carry a [VERIFY] flag for primary sourcing.

The raw Monte Carlo draws and local sensitivity matrices (about 400 MB) are not included, because the script regenerates them exactly from the fixed random seed (20261013).

## Running the analysis

Requirements:

- R 4.6 or later, with the `h2o` and `vegan` packages (the full run used h2o 3.44.0.3 and vegan 2.7-6)
- A Java runtime supported by H2O (H2O certifies Java 8 to 17; the full run used Java 18, which H2O does not certify)
- About 16 GB of free memory for the H2O cluster

To run the full analysis, source the script in R:

```r
source("code/venus_mars_mc_ann_v26.R")
```

The full run (ten tests, 20,000 draws each, 100 bootstrap refits and a 40-refit null arm per input) took about 126 hours on a Windows 11 laptop. Completed tests are checkpointed, so an interrupted run can be restarted and will skip the finished tests. Output goes to `C:/mars_venus_paper/output/v26` when that folder exists, and otherwise to `venus_mars_mc_ann_v26_output` in the working directory.

For a quick check of the pipeline at reduced settings (3,000 draws, 4 bootstrap refits), set the environment variable `VM_QUICK=1` before running:

```r
Sys.setenv(VM_QUICK = "1")
source("code/venus_mars_mc_ann_v26.R")
```

Quick-mode results are for testing only and do not reproduce the paper's figures.

Note that the script stops any H2O cluster already listening on localhost port 54321 before it starts its own, so other H2O jobs should finish first.

## Citation

If you use this code, please cite the archived release:

Woolman, T. A. (2026). *Venus aerostat colony versus Mars surface colony: Monte Carlo and neural network analysis* (Version 1.0.1) [Computer software]. Zenodo. https://doi.org/10.5281/zenodo.24042381

The same metadata is in `CITATION.cff`, which GitHub uses for the **Cite this repository** button.

## License

The code is released under the MIT License (see `LICENSE`).
