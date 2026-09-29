# Can Inflation Base Effects Predict Sovereign Bond Returns?

## A cross-country falsification study

When a large CPI print from 12 months ago drops out of the year-over-year
calculation, reported inflation mechanically falls. This project asks whether
that predictable arithmetic is nevertheless under-priced by sovereign bond
markets.

**Result:** the pre-specified signal does not demonstrate positive bond alpha in
this public-data implementation. The null result is retained rather than
reversing the signal or selecting a favorable specification after the fact.

![Cumulative value added across the primary and robustness specifications](docs/figures/strategy_summary.png)

## Headline evidence

The primary strategy uses the raw NSA CPI print rolling out of the YoY window,
a six-month EWM z-score, a one-month implementation lag, and a constrained
duration-neutral optimizer.

| Specification | Sample | N | Ann. return | Ann. vol. | Return/vol | NW t | NW p |
|---|---:|---:|---:|---:|---:|---:|---:|
| Primary NSA/EWM optimizer | 1991-09 to 2024-12 | 400 | -0.08% | 2.75% | -0.03 | -0.18 | 0.854 |
| Expanding month-of-year optimizer | 1996-04 to 2024-12 | 345 | 0.54% | 2.84% | 0.19 | 1.18 | 0.239 |
| Primary signal, rank/DV01 implementation | 1991-09 to 2024-12 | 400 | -0.02% | 1.42% | -0.01 | -0.09 | 0.928 |

The seasonal robustness variant is mildly positive but statistically
insignificant. It is reported as a diagnostic, not promoted to a replacement
model. The optimizer-free rank/DV01 portfolio is also indistinguishable from
zero.

The proposed expectations mechanism is unsupported as well:

| Forecast revision proxy | N | Coefficient | HAC t | p-value | R² |
|---|---:|---:|---:|---:|---:|
| Michigan monthly revision | 426 | -0.0736 | -1.27 | 0.203 | 0.0050 |
| SPF same-target revision | 143 | -0.0301 | -0.63 | 0.528 | 0.0032 |

The hypothesis requires positive mechanism coefficients under the project's
sign convention. Both estimates have the opposite sign, wide confidence
intervals, and negligible explanatory power.

## Research design

### Signal timing

1. Compute monthly log inflation from NSA CPI price levels.
2. Shift it 12 months to identify the print rolling out of the YoY window.
3. Apply the pre-specified six-month EWM z-score and cap at ±2.
4. Delay implementation by one month.
5. Form holdings at month `t` and earn the complete-case return at `t+1`.

The first month without prior holdings is excluded rather than silently
recorded as a zero return.

### Core robustness checks

- **Real-time seasonal normalization:** each calendar month is standardized
  using only prior observations from that same month, with a five-observation
  warm-up.
- **Optimizer-free benchmark:** the top two signals are long and the bottom two
  short; inverse-duration notionals produce unit gross exposure and exact DV01
  neutrality.
- **HAC inference:** Newey-West standard errors are reported for the strategy
  mean and mechanism regressions.
- **Costs and stability:** the notebook reports fixed-basis-point transaction
  cost scenarios, formal pre/post-2010 HAC tests, and trailing 10-year estimates.

## Did the signal decay after being priced in?

The data do not support that account. The 2010 split was already present in the
study and is retained as a fixed stability diagnostic rather than selected from
the returns. It is not claimed to be a publication or market-adoption date.

| Implementation and period | N | Ann. return | Ann. vol. | Return/vol | NW t | NW p |
|---|---:|---:|---:|---:|---:|---:|
| Optimizer, pre-2010 | 220 | -0.30% | 2.80% | -0.11 | -0.52 | 0.605 |
| Optimizer, 2010 onward | 180 | 0.20% | 2.69% | 0.07 | 0.29 | 0.774 |
| Rank/DV01, pre-2010 | 220 | -0.13% | 1.45% | -0.09 | -0.43 | 0.670 |
| Rank/DV01, 2010 onward | 180 | 0.11% | 1.38% | 0.08 | 0.31 | 0.754 |

The annualized post-minus-pre change is **+0.50%** for the optimizer
(`t = 0.58`, `p = 0.565`) and **+0.25%** for the rank/DV01 portfolio
(`t = 0.52`, `p = 0.600`). Both changes are positive and statistically
insignificant. The evidence therefore indicates neither historical positive
alpha nor subsequent decay.

![Trailing 10-year return estimates and Newey-West confidence intervals](docs/figures/temporal_stability.png)

The rolling windows overlap and are descriptive. They show how imprecisely
performance varies through time, but they are not independent tests and cannot
identify when investors may have learned the signal.

## Data and reproducibility

The showcased run uses the committed snapshot in
`data/snapshots/2026-09-28/`. Its manifest records source identifiers, coverage,
retrieval time, and SHA-256 checksums. This makes the saved result reproducible
without a FRED key or dependence on future source revisions.

| Dataset | Source | Frequency |
|---|---|---|
| CPI price levels, NSA | FRED / OECD | Monthly |
| 10Y government bond yields | FRED / OECD | Monthly averages |
| Inflation expectations | University of Michigan via FRED | Monthly |
| CPI forecast revisions | Philadelphia Fed SPF | Quarterly |

The US yield series is monthly [`GS10`](https://fred.stlouisfed.org/series/GS10),
an average of business-day observations, rather than a month-end resampling of
daily `DGS10`. SPF revisions use official
[deadline and release-date documentation](https://www.philadelphiafed.org/-/media/frbp/assets/surveys-and-data/survey-of-professional-forecasters/spf-documentation.pdf).

The common-country signal ends in 2024-12 because the referenced OECD CPI
series for Germany, the UK, and Canada end in 2023-11; those observations can
identify prints rolling off through the following 12 months.

## Reproduce the study

Python 3.10 and [`uv`](https://docs.astral.sh/uv/) are required.

```bash
uv sync --locked --all-extras
uv run pytest -q
uv run jupyter nbconvert --to notebook --execute --inplace \
  notebooks/01_data_and_mechanism.ipynb \
  notebooks/02_signal_and_backtest.ipynb
```

To create a new dated snapshot from the official sources:

```bash
cp .env.example .env
# Add a free FRED_API_KEY to .env
uv run python scripts/refresh_data.py --date YYYY-MM-DD
```

Refreshes never overwrite an existing non-empty snapshot. HTTPS certificate
verification remains enabled; corporate CA bundles can be supplied through
`SSL_CERT_FILE`.

## Repository structure

```text
.
├── data/snapshots/2026-09-28/   # Frozen public-data inputs and manifest
├── docs/figures/                 # README research figure
├── notebooks/
│   ├── 01_data_and_mechanism.ipynb
│   └── 02_signal_and_backtest.ipynb
├── scripts/refresh_data.py       # Explicit, non-overwriting live refresh
├── src/inflation_base_effects/
│   ├── data.py
│   ├── signals.py
│   ├── portfolio.py
│   └── evaluation.py
└── tests/                        # Timing, data, constraints, and robustness
```

## Limitations

- Returns are `carry - duration × yield change` approximations, not futures or
  total-return indices.
- Monthly-average yields cannot identify announcement-day market adjustment.
- Current-vintage CPI histories may differ from the real-time values available
  to investors.
- The four-country optimizer remains position-cap dominated.
- Michigan expectations change forecast horizon; SPF evidence is US-only.
- Transaction costs are illustrative fixed-basis-point scenarios.
- A positive robustness point estimate is not an untouched discovery sample.
- The 2010 split is a fixed diagnostic rather than a causal market-adoption
  date; overlapping rolling windows are descriptive.

## What the project demonstrates

- translating an economic mechanism into an explicitly timed signal;
- separating a pre-specified test from post-hoc diagnostics;
- constrained portfolio construction and an optimizer-free benchmark;
- complete-case return accounting, HAC inference, mechanism tests, and a
  disciplined temporal-stability analysis;
- reproducible research with frozen inputs, tested code, and honest reporting
  of a failed hypothesis.

## License

MIT — see [LICENSE](LICENSE).
