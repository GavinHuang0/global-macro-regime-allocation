# Legacy course-project artifacts

This directory preserves the original team manuscript for provenance and comparison.

- [original_paper.pdf](original_paper.pdf) is the four-author course manuscript.

The manuscript documents the original model design and results. The original
execution environment and complete implementation are not part of this
repository. Model 01 and Model 02 are independent implementations in the
`regime_allocation` package; they do not import material from this directory.

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
Mianchen Zhang, Gavin Huang, and Serin Gleave. Model 01 and Model 02 represent
Gavin Huang's subsequent independent rebuild and extensions.
