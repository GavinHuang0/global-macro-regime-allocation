# Legacy course-project artifacts

This directory preserves the original team manuscript for provenance and comparison.

- [original_paper.pdf](original_paper.pdf) is the four-author course manuscript.

The incomplete `bayesian_v2.py` orchestration script was removed from the working
tree and remains available in Git history. Its HMM, data-generation,
macro-evidence, allocation, and backtest dependencies were not recovered, and
no research pipeline imported it. The manuscript retains the original model
description and results. Nothing in this directory is imported by the new
`regime_allocation` package.

The rebuild was motivated by specific architectural problems in the course
version: full-sample scaling and HMM fitting; smoothed or Viterbi state paths
that could use future observations; observation-month rather than release-date
macro alignment; daily forward filling of monthly releases as repeated
evidence; an HMM transition matrix treated as a macro-regime transition law;
likelihood labels derived from mapped HMM states rather than the stated macro
definition; unclear use of the Bayesian posterior in allocation; unstable
regime-specific covariance estimates; score-shifted long-only weights; and no
complete transaction-cost or adjusted-price accounting. Model 01 was designed
as a clean, independently testable alternative rather than a patch to that
code.

Authorship of the original material remains with the course team: Zekai Yao,
Mianchen Zhang, Gavin Huang, and Serin Gleave. New model packages represent
Gavin Huang's subsequent independent rebuild.
