"""
bayesian_v2.py
==============
Data-driven Bayesian regime inference — NO hand-coded weights.

Architecture:
  Prior:      HMM transition matrix × yesterday's posterior
              P(regime_t = k) = Σ_j  A[j,k] · P(regime_{t-1} = j)
  Likelihood: Multivariate Gaussian fitted from historical (macro_evidence, regime) pairs
              P(e_t | regime_k) = N(e_t; μ_k, Σ_k)
              e_t = 11-dim macro evidence vector (CPI, NFP, 10Y Treasury)
  Posterior:  P(regime | e_t, market) ∝ Prior × Likelihood

Evidence vector e_t (11 dimensions):
  CPI block  (4): cpi_yoy, cpi_mom, cpi_surprise, cpi_accel
  NFP block  (3): nfp_change, nfp_surprise, nfp_trend
  10Y block  (4): yield_level, yield_1d_chg, yield_5d_chg, yield_20d_vol

No news, no text, no AI — pure structured macro data as Bayesian evidence.

Data flow:
  1. HMM runs on market CSV → regime labels + transition matrix A
  2. macro_evidence.py → 11-dim structured feature vector from CPI/NFP/10Y
  3. This script:
     a. Fits HMM → prior from transition matrix
     b. Aligns macro evidence with HMM regime labels
     c. Fits P(e_t | regime_k) per regime from historical data
     d. Evaluates likelihood for today's e_t
     e. Bayesian update → posterior

Output:
  - data/bayesian_v2_chart.png
"""

import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from sklearn.preprocessing import StandardScaler

# ── local imports ─────────────────────────────────────────────────
from hmm_regime import (
    GaussianHMM, N_REGIMES, load_features,
    assign_regime_labels,
    REGIME_NAMES, REGIME_LABELS, REGIME_COLORS,
)
from macro_evidence import (
    build_macro_features, load_macro_evidence,
    fit_macro_likelihood, evaluate_macro_likelihood,
    get_latest_evidence, MACRO_FEATURES,
)

DATA_DIR = pathlib.Path(__file__).parent / "data"


# ══════════════════════════════════════════════════════════════════
# 1. Fit HMM and extract transition matrix + current regime prior
# ══════════════════════════════════════════════════════════════════

def fit_hmm() -> tuple[GaussianHMM, StandardScaler, list[str], np.ndarray, pd.DatetimeIndex]:
    """
    Fit HMM on market data. Returns:
      - model: trained GaussianHMM
      - scaler: fitted StandardScaler
      - labels_map: state → regime name mapping
      - gamma: smoothed regime probabilities (T, K)
      - dates: DatetimeIndex
    """
    features, dates = load_features()
    scaler = StandardScaler()
    X = scaler.fit_transform(features.values)

    model = GaussianHMM(n_states=N_REGIMES, n_iter=200, tol=1e-4)
    model.fit(X)

    labels_map = assign_regime_labels(model, scaler)
    gamma = model.predict_proba(X)

    return model, scaler, labels_map, gamma, dates


# ══════════════════════════════════════════════════════════════════
# 2. Prior: HMM transition matrix × previous posterior
# ══════════════════════════════════════════════════════════════════

def compute_prior(
    model: GaussianHMM,
    labels_map: list[str],
    gamma: np.ndarray,
) -> np.ndarray:
    """
    Prior = A^T @ yesterday's posterior (HMM smoothed).

    The HMM's transition matrix A[i,j] = P(state_{t+1}=j | state_t=i)
    captures temporal regime dynamics learned purely from market data.
    """
    prev_posterior = gamma[-1]  # (K,)
    prior = model.A.T @ prev_posterior  # (K,)
    prior = prior / prior.sum()

    # Reorder to match REGIME_NAMES order
    prior_ordered = np.zeros(len(REGIME_NAMES))
    for k in range(model.n_states):
        regime_name = labels_map[k]
        idx = REGIME_NAMES.index(regime_name)
        prior_ordered[idx] = prior[k]

    prior_ordered = prior_ordered / prior_ordered.sum()

    print(f"[Prior] From HMM transition matrix (data-driven, no hand-coded weights)")
    for i, name in enumerate(REGIME_NAMES):
        print(f"        {name:24s}  {prior_ordered[i]:.3f}")

    return prior_ordered


# ══════════════════════════════════════════════════════════════════
# 3. Build regime labels for macro evidence alignment
# ══════════════════════════════════════════════════════════════════

def build_regime_labels(
    model: GaussianHMM,
    scaler: StandardScaler,
    labels_map: list[str],
) -> pd.Series:
    """
    Use HMM Viterbi decoding to assign a regime label to every day
    in the market data. Returns a Series indexed by date.
    """
    features, dates = load_features()
    X = scaler.transform(features.values)
    states = model.decode(X)

    regime_labels = pd.Series(
        [labels_map[s] for s in states],
        index=dates,
        name="regime",
    )
    return regime_labels


# ══════════════════════════════════════════════════════════════════
# 4. Bayesian update
# ══════════════════════════════════════════════════════════════════

def bayesian_update(prior: np.ndarray, likelihood: np.ndarray) -> np.ndarray:
    unnorm = prior * likelihood
    posterior = unnorm / unnorm.sum()
    return posterior


# ══════════════════════════════════════════════════════════════════
# 5. Plotting
# ══════════════════════════════════════════════════════════════════

def plot_results(
    prior: np.ndarray,
    likelihood: np.ndarray,
    posterior: np.ndarray,
    n_days: int,
    evidence_date: str,
    e_t: np.ndarray,
):
    labels = [REGIME_LABELS[r] for r in REGIME_NAMES]
    colors = [REGIME_COLORS[r] for r in REGIME_NAMES]

    fig = plt.figure(figsize=(20, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.4, wspace=0.35)

    fig.suptitle(
        "Bayesian v2 — Macro Evidence Regime Inference (No Hand-Coded Weights)\n"
        f"Prior (HMM Transition) × Likelihood (CPI + NFP + 10Y Treasury)  |  "
        f"Trained on {n_days} days  |  Evidence date: {evidence_date}",
        fontsize=13, fontweight="bold", y=1.02,
    )

    # ── 1. Prior vs Posterior ─────────────────────────────────────
    ax = fig.add_subplot(gs[0, 0])
    x = np.arange(len(REGIME_NAMES))
    w = 0.35
    ax.bar(x - w / 2, prior, w, label="Prior (HMM)", color="#95a5a6", alpha=0.7)
    bars_post = ax.bar(x + w / 2, posterior, w, label="Posterior",
                       color=colors, edgecolor="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("Probability")
    ax.set_title("Prior (HMM) vs Posterior")
    ax.legend(fontsize=7)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.set_ylim(0, max(posterior.max(), prior.max()) * 1.35)
    for bar in bars_post:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.005,
                f"{h:.1%}", ha="center", va="bottom", fontsize=8, fontweight="bold")

    # ── 2. Posterior pie ──────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 1])
    ax.pie(
        posterior, labels=labels, colors=colors,
        autopct=lambda pct: f"{pct:.1f}%" if pct > 3 else "",
        startangle=140, pctdistance=0.75,
        textprops={"fontsize": 9},
    )
    ax.set_title("Posterior Distribution")

    # ── 3. Likelihood bar ─────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 2])
    ax.barh(labels, likelihood, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_xlabel("Likelihood Weight")
    ax.set_title("Likelihood P(macro evidence | regime)")
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    for i, v in enumerate(likelihood):
        ax.text(v + 0.005, i, f"{v:.1%}", va="center", fontsize=9)

    # ── 4. Today's evidence vector ────────────────────────────────
    ax = fig.add_subplot(gs[1, :])
    feature_labels = [f.replace("_", "\n") for f in MACRO_FEATURES]
    bar_colors_feat = (["#e74c3c"] * 4 +   # CPI block = red
                       ["#3498db"] * 3 +    # NFP block = blue
                       ["#2ecc71"] * 4)     # 10Y block = green
    bars = ax.bar(feature_labels, e_t, color=bar_colors_feat, edgecolor="black",
                  linewidth=0.5, alpha=0.8)
    ax.set_ylabel("Value")
    ax.set_title(f"Today's Macro Evidence Vector e_t  ({evidence_date})", fontweight="bold")
    ax.axhline(0, color="black", linewidth=0.5)

    # Add value labels
    for bar in bars:
        h = bar.get_height()
        va = "bottom" if h >= 0 else "top"
        offset = 0.01 * (1 if h >= 0 else -1) * max(abs(e_t))
        ax.text(bar.get_x() + bar.get_width() / 2, h + offset,
                f"{h:.3f}", ha="center", va=va, fontsize=7)

    # Legend for feature blocks
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#e74c3c", alpha=0.8, label="CPI (Inflation)"),
        Patch(facecolor="#3498db", alpha=0.8, label="NFP (Labor Market)"),
        Patch(facecolor="#2ecc71", alpha=0.8, label="10Y Treasury (Rates)"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=8)

    out_path = DATA_DIR / "bayesian_v2_chart.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\n[Saved] {out_path}")
    plt.show()


# ══════════════════════════════════════════════════════════════════
# 6. Main entry point
# ══════════════════════════════════════════════════════════════════

def run_bayesian_v2() -> np.ndarray:
    """
    Full data-driven Bayesian pipeline:
      Prior   = HMM transition matrix (from market data)
      Evidence = structured macro vector (CPI, NFP, 10Y)
      Likelihood = P(evidence | regime) learned from historical alignment
      Posterior = Prior × Likelihood
    """
    print("=" * 60)
    print("  Bayesian v2 — Macro Evidence Regime Inference")
    print("=" * 60)

    # ── Step 1: Fit HMM on market data ───────────────────────────
    print("\n[Step 1] Fitting HMM on market data ...")
    model, scaler, labels_map, gamma, dates = fit_hmm()
    print(f"         {len(dates)} observations, {model.n_states} regimes")
    print(f"         Regime mapping: {labels_map}")

    # ── Step 2: Compute prior from HMM transition matrix ─────────
    print("\n[Step 2] Computing prior from HMM transition matrix ...")
    prior = compute_prior(model, labels_map, gamma)

    # ── Step 3: Fetch macro evidence (CPI, NFP, 10Y) ─────────────
    print("\n[Step 3] Building macro evidence vector ...")
    macro = load_macro_evidence()
    if macro is None or len(macro) == 0:
        print("         No cached macro data found, fetching from FRED + yfinance ...")
        macro = build_macro_features(start="2015-01-01")

    if len(macro) == 0:
        print("         ERROR: Could not load macro evidence. Using uniform likelihood.")
        likelihood = np.ones(len(REGIME_NAMES)) / len(REGIME_NAMES)
        posterior = bayesian_update(prior, likelihood)
        return posterior

    # ── Step 4: Align macro evidence with HMM regime labels ───────
    print("\n[Step 4] Aligning macro evidence with HMM regime labels ...")
    regime_labels = build_regime_labels(model, scaler, labels_map)

    # Intersect dates
    common_dates = macro.index.intersection(regime_labels.index)
    macro_aligned = macro.loc[common_dates]
    regimes_aligned = regime_labels.loc[common_dates]
    print(f"         {len(common_dates)} overlapping days")

    # ── Step 5: Fit P(evidence | regime) from historical data ─────
    print("\n[Step 5] Fitting likelihood P(e_t | regime) ...")
    params = fit_macro_likelihood(macro_aligned, regimes_aligned, REGIME_NAMES)

    # ── Step 6: Evaluate likelihood for today's evidence ──────────
    print("\n[Step 6] Evaluating likelihood for latest evidence ...")
    e_t, evidence_date = get_latest_evidence(macro)
    print(f"         Evidence date: {evidence_date}")
    print(f"         e_t = {dict(zip(MACRO_FEATURES, e_t.round(4)))}")

    likelihood = evaluate_macro_likelihood(e_t, params, REGIME_NAMES)
    print(f"[Likelihood] Macro-evidence driven (learned from {len(common_dates)} days):")
    for i, name in enumerate(REGIME_NAMES):
        print(f"             {name:24s}  {likelihood[i]:.3f}")

    # ── Step 7: Bayesian update ───────────────────────────────────
    print("\n[Step 7] Bayesian update: posterior = prior × likelihood")
    posterior = bayesian_update(prior, likelihood)

    print("\n── Regime Posterior (Macro Evidence) ──")
    for i, name in enumerate(REGIME_NAMES):
        print(f"  {name:24s}  prior={prior[i]:.3f}  "
              f"lk={likelihood[i]:.3f}  posterior={posterior[i]:.3f}")
    dominant = REGIME_NAMES[np.argmax(posterior)]
    print(f"\n  → Dominant regime: {dominant} ({posterior.max():.1%})")

    # ── Step 8: Plot ──────────────────────────────────────────────
    plot_results(prior, likelihood, posterior, len(common_dates), evidence_date, e_t)

    return posterior


# ══════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    run_bayesian_v2()
