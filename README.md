# Can Inflation Base Effects Predict Sovereign Bond Returns?

I test whether a signal based on last year's CPI prints predicts sovereign
bond returns in the US, Germany, the UK, and Canada.
A large price increase leaving the annual comparison puts downward pressure on
reported inflation. If bond investors underreact to that predictable component,
it could create an opportunity to buy duration ahead of the adjustment.

I find little evidence of that opportunity. The primary duration-neutral
portfolio earns approximately -0.08% per year before costs, with a 95% confidence
interval of -0.92% to +0.77%. Tests using US inflation-expectations surveys also
provide little support for the proposed expectations channel.

## Results

I compare an optimizer-based portfolio with a seasonal signal variant and an
optimizer-free rank/DV01 portfolio. Returns are synthetic bond-return estimates
constructed from yields. Annual return is the annualized arithmetic mean;
return/volatility is the ratio of that mean to annualized volatility.

| Specification | Sample | Months | Ann. return | Ann. vol. | Return/vol | NW t | NW p |
|---|---|---:|---:|---:|---:|---:|---:|
| Primary NSA/EWM optimizer | 1991-09 to 2024-12 | 400 | -0.08% | 2.75% | -0.03 | -0.18 | 0.861 |
| Expanding month-of-year optimizer | 1996-04 to 2024-12 | 345 | 0.54% | 2.84% | 0.19 | 1.18 | 0.237 |
| Primary signal, rank/DV01 portfolio | 1991-09 to 2024-12 | 400 | -0.02% | 1.42% | -0.01 | -0.09 | 0.928 |

The seasonal signal needs a longer warm-up. On the common April 1996–December
2024 sample, the primary mean is approximately -0.01% per year, the seasonal
mean is +0.54%, and the rank/DV01 mean is -0.03%. None differs significantly
from zero under Newey-West inference. The notebooks report both sample
conventions, confidence intervals, drawdowns, and turnover.

![Cumulative value added on common dates](docs/figures/strategy_summary.png)

The US survey regressions estimate how forecast revisions relate to the negative
of the CPI print leaving the annual window. My hypothesis implies a positive
coefficient: downward mechanical pressure on inflation should accompany downward
forecast revisions.

| Forecast revision proxy | Observations | Coefficient | HAC t | p-value | R² |
|---|---:|---:|---:|---:|---:|
| Michigan monthly revision | 426 | -0.0736 | -1.27 | 0.203 | 0.0050 |
| SPF same-target revision | 143 | -0.0301 | -0.63 | 0.528 | 0.0032 |

Both estimates are negative, imprecise, and explain less than 1% of revision
variation. These proxies provide little evidence for the expectations channel;
they cover US respondents and cannot establish how investors in all four bond
markets form expectations.

## Economic mechanism and method

Prices can remain high while annual inflation falls. In the example below, a
one-time price increase from 100 to 101 produces 1% annual inflation for twelve
months. Annual inflation then falls to zero while the price level stays at 101.
An equal proportional increase a year later offsets the departing increase.

![Price levels, annual inflation, and the mechanical entry and exit of a CPI increase](docs/figures/mechanical_base_effect.png)

The timing bars show changes in annual inflation in percentage points. I use
this accounting identity to motivate the signal, then test expectations and
returns separately.

I calculate monthly log inflation from not seasonally adjusted (NSA) CPI levels
and identify the print leaving the twelve-month window. I standardize it with
a six-month exponentially weighted mean and standard deviation, cap the score
at ±2, and apply a one-month implementation delay. Holdings formed in month
`t` earn returns in `t+1`; months without the required holdings or returns are
excluded.

The optimizer balances expected return against a blend of 12- and 36-month
covariance estimates. I constrain duration exposure to zero, gross exposure to
2, each position to ±0.5, and annual forecast volatility to at most 10%. The
assumed information coefficient of 0.05 determines sizing. I assess its influence
alongside signal smoothing, implementation delay, covariance half-lives, and
Newey-West lag length.

The seasonal comparison standardizes each calendar month using only earlier
observations from that same month, with a five-observation warm-up. The rank
portfolio buys the top two signals and sells the bottom two, with inverse-duration
weights, unit gross exposure, and zero net DV01.

## Portfolio behaviour and interpretation

The UK contributes positively and the US negatively to the four-country portfolio.
I separate country contributions, carry, and yield-change returns, then
re-estimate portfolios excluding each country to assess concentration. Those
re-estimated portfolios retain the same signal and constraints.

![Country contributions and carry versus yield-change attribution](docs/figures/portfolio_attribution.png)

Position limits bind in almost every formation month. With four assets capped
at 0.5, the gross limit of 2 is already implied by the position limits. The
volatility ceiling never binds. Average monthly turnover is close to two units
of notional, so a 1 bp one-way charge costs approximately 0.24 percentage points
per year.

The strategy has little correlation with an equal-weight long-only bond-proxy
benchmark, but its benchmark-adjusted intercept is imprecise. A fixed 50/50
combination does not improve the benchmark's return/volatility ratio.

### Temporal stability

I compare performance before and after January 2010 and estimate trailing
ten-year means. The annualized post-minus-pre change is +0.51% for the optimizer
(`t = 0.59`, `p = 0.558`) and +0.25% for the rank portfolio
(`t = 0.52`, `p = 0.601`). Neither change is statistically significant.

![Trailing ten-year means and Newey-West confidence intervals](docs/figures/temporal_stability.png)

The 2010 split is a descriptive sample division. It identifies neither a market
adoption date nor a causal change in investor behaviour. The rolling windows
overlap, and the study has no untouched holdout sample. I treat the specification
comparisons as exploratory; their p-values are not adjusted for multiple testing.

### Measurement limits

The primary confidence interval includes both losses and gains of economic
interest. I cannot establish positive predictive performance from these results,
and I cannot rule out an effect that the available data measure imprecisely.

- Returns use `carry - duration × yield change`. They omit convexity, instrument
  cash flows, financing, and FX hedges. Cumulative value added is additive P&L per
  unit of reference notional.
- Monthly-average yields do not represent executable month-end prices or
  announcement-day returns. Substituting US month-end yields produces about
  -0.45% annual portfolio return in the sampling check; the other countries still
  use monthly averages.
- Duration neutrality offsets a common parallel yield movement in the proxy
  units. Country-specific rate movements, currency exposure, and funding remain.
- Current-vintage data may differ from information available to investors at
  each historical date. Michigan expectations also change target horizon
  between surveys; the SPF comparison holds the target quarter fixed.

## Notebooks

| Notebook | Contents |
|---|---|
| [1. Economic mechanism and data](notebooks/01_data_and_mechanism.ipynb) | Inflation arithmetic, data coverage, timing, and US expectations tests |
| [2. Portfolio experiment](notebooks/02_signal_and_backtest.ipynb) | Construction, performance, attribution, country exclusions, benchmarks, costs, and stability |
| [3. Sensitivity and measurement](notebooks/03_sensitivity_and_measurement.ipynb) | Duration, return accounting, US yield sampling, parameter sensitivity, and inference |

Each notebook includes executed tables and figures. Calculations shared across
the notebooks live in `src/inflation_base_effects/`; tests cover timing, data
integrity, constraints, attribution, and statistical helpers.

## Data and reproduction

| Dataset | Source | Frequency |
|---|---|---|
| NSA CPI price levels | FRED / OECD | Monthly |
| 10-year government yields | FRED / OECD | Monthly averages |
| Inflation expectations | University of Michigan via FRED | Monthly |
| CPI forecasts | Philadelphia Fed SPF | Quarterly |

I use the frozen inputs in `data/snapshots/2026-09-28/`. The manifest records
source identifiers, coverage, and retrieval time; a checksum file verifies the
saved data. Separate GS10/DGS10 inputs in `data/supplementary/2026-10-01/` support
the US sampling comparison and have their own provenance and checksums.

The US monthly yield series is [GS10](https://fred.stlouisfed.org/series/GS10),
an average of business-day observations. I use
[DGS10](https://fred.stlouisfed.org/series/DGS10) for month-end sampling.
SPF revisions use the official survey deadlines and forecast definitions in the
[SPF documentation](https://www.philadelphiafed.org/-/media/frbp/assets/surveys-and-data/survey-of-professional-forecasters/spf-documentation.pdf).

The common-country signal ends in December 2024. The referenced OECD CPI series
for Germany, the UK, and Canada end in November 2023; their last prints enter
the rolling-off signal twelve months later, followed by the implementation delay.

Python 3.10 and [uv](https://docs.astral.sh/uv/) are required. Executing the
notebooks uses the saved inputs and needs no API key.

```bash
uv sync --locked --all-extras
uv run pytest -q
uv run jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=1800 \
  notebooks/01_data_and_mechanism.ipynb \
  notebooks/02_signal_and_backtest.ipynb \
  notebooks/03_sensitivity_and_measurement.ipynb
```

To retrieve another dated snapshot, copy `.env.example` to `.env`, add a FRED
API key, and run `uv run python scripts/refresh_data.py --date YYYY-MM-DD`.
The US yield download uses `uv run python scripts/freeze_us_yields.py --date YYYY-MM-DD`
and needs no key. Downloads require an unused destination directory;
the notebooks keep their explicit snapshot dates. HTTPS certificate verification
remains enabled, with `SSL_CERT_FILE` available for a custom CA bundle.

CI checks formatting, lint, tests, notebook execution, and package build.

## License

MIT. See [LICENSE](LICENSE).
