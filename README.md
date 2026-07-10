# Global Macro Regime Detection and FICC Allocation

An independent rebuild and extension of a team course project on macro-regime
detection and regime-conditioned asset allocation.

## Status

This repository is currently a research scaffold. The only preserved source
module from the original project is `legacy/bayesian_v2.py`. It depends on
missing original modules and is retained for audit and redesign, not as a
working production implementation.

## Original framework

The course project combined:

1. CPI and non-farm payroll data to define four growth/inflation regimes.
2. Kalman-filtered global equity features and a four-state Gaussian HMM.
3. A Bayesian update using an 11-feature macro evidence vector.
4. Regime-conditional Sharpe-ratio or information-ratio allocation.
5. A monthly walk-forward backtest over six FICC ETFs.

The original paper is preserved at `docs/original_paper.pdf`.

## Planned rebuild

- Point-in-time data ingestion with explicit release timestamps
- Reproducible feature engineering and Kalman filtering
- Properly separated HMM state estimation and economic state labeling
- Filtered, rather than smoothed, real-time state probabilities
- Walk-forward model refitting
- Robust likelihood estimation and calibration
- Portfolio optimization with turnover and transaction costs
- Baselines, ablations, sensitivity analysis, and statistical validation
- Unit tests and reproducible experiments

## Repository layout

```text
configs/                   experiment and data settings
data/raw/                  local raw data, ignored by git
data/processed/            local derived data, ignored by git
docs/                      paper and methodological notes
legacy/                    preserved original code
notebooks/                 exploratory analysis only
src/regime_allocation/     rewritten package
tests/                     automated tests
```

## Attribution

The original paper and initial framework were produced as a team course
project by Zekai Yao, Mianchen Zhang, Gavin Huang, and Serin Gleave. This
repository is intended to document Gavin Huang's subsequent independent
reimplementation, corrections, and extensions. Contributions from the
original project should not be represented as solely individual work.

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows PowerShell
pip install -e ".[dev]"
```

## Current limitations

The repository does not yet contain the original HMM, macro data generation,
Kalman filter, allocation, or backtest modules. `legacy/bayesian_v2.py` will
therefore not run until those pieces are independently reimplemented.
