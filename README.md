# Option-Pricing Model — Reconstruction Workflow (any ticker)

**Goal:** understand the model Barchart uses to compute the option prices,
implied volatility (IV), and greeks shown on
`barchart.com/stocks/quotes/SPCX/options?expiration=2026-12-18-m`, and
determine which candidate model is *closest* to what they display. The workflow
started SPCX-specific and now **generalizes to any US ticker with listed
options** — enter it in a box in the local web app.

## Interactive dashboard — type any ticker

```shell
pip install yfinance matplotlib plotly     # one-time
python src/app.py                          # serves http://127.0.0.1:8000
```

Open the URL, type a ticker (SPCX, AAPL, SPY, TSLA, …), press **Analyze**. The
backend pulls that ticker's live chain, inverts Black-Scholes per option, builds
the vol surface / greeks(T) / parameters(T), scores the candidate models, and
returns the dashboard. Each run is ~30–60s (fetch + surface + American-binomial
scoring). Dividend yield `q` is read from trailing dividends per ticker.

The **3D vol surface is interactive** — drag to rotate, scroll to zoom,
double-click to reset. Plotly is **bundled locally** (no CDN): the app serves it
once at `/plotly.js` (browser-cached), and the static file build inlines it, so
the 3D renders offline. The heatmap bins span the **observed moneyness range**
with a data-derived bin size (shown in its title).

> **Why a local app and not a published Artifact?** A hosted Artifact runs under
> a strict CSP that blocks *all* network requests, so a static page can't fetch
> live option data for a typed ticker. The interactive box therefore needs this
> Python backend. The published Artifact remains a **static snapshot** of one
> ticker; `python src/build_dashboard.py --symbol XYZ` regenerates that snapshot
> file for any symbol.

---

**Approach:** Barchart does not publish equations on the page, and the site is
Cloudflare-gated (live scraping is unreliable) — so this workflow does not
depend on reading their docs. Instead it (1) hypothesizes the small set of
models any quote vendor could plausibly use, (2) implements each, and (3)
provides a harness that determines the closest model — analytically now, and
empirically the moment you paste in a real chain snapshot.

> **Epistemic status.** The conclusion below is an inference from option theory
> plus a real chain from **Yahoo** (`src/fetch_chain.py`). It is *not* a quote
> from Barchart's documentation, and Yahoo's IV is Yahoo's calc, not Barchart's.
> To test Barchart specifically, paste their displayed IV/delta into the
> `bar_iv`/`bar_delta` columns of `data/spcx_2026-12-18.csv` and re-run.

> **Data status (2026-07-20).** Barchart's endpoints need session tokens and the
> page is Cloudflare-gated, so the chain was pulled from **Yahoo** instead:
> `data/spcx_2026-12-18.csv` — SPCX spot **$123.76**, 80 calls + 73 puts, ATM IV
> ≈ 78%. Saved run: `data/results_2026-12-18.txt`.

---

## The hypotheses (candidate models)

| # | Model | Exercise | Why it's a candidate |
|---|-------|----------|----------------------|
| 1 | **Black-Scholes-Merton** (with continuous dividend yield `q`) | European | The de-facto display model for listed equity/ETF options across quote vendors. Closed-form price + greeks; IV by inversion. |
| 2 | **Cox-Ross-Rubinstein binomial** | American | *Theoretically correct*: every listed US single-name/ETF option is American-exercise, so early exercise (esp. ITM puts, pre-dividend calls) matters. |
| 3 | **Black-76** (option on the forward) | European | Used for futures/index options. For equities it is **algebraically identical to BSM** once `F = S·e^{(r−q)T}` — included only to show it is not a distinct hypothesis. |

### Parameters (identical inputs to every model)

| Symbol | Meaning | Source |
|--------|---------|--------|
| `S` | SPCX underlying price | live quote at snapshot time |
| `K` | strike | the option row |
| `T` | years to expiry | (calendar days to **2026-12-18**) / 365 |
| `r` | risk-free rate (cont-comp) | short-dated US Treasury (~3M) |
| `q` | dividend yield (cont-comp) | fetched **per ticker** from trailing-12-month dividends (`t.dividends / spot`). **SPCX pays none → q = 0**, so American calls ≡ European calls and only puts can differ; for a dividend payer `q > 0` also enables early exercise on calls near ex-div dates |
| `σ` | volatility | **solved** so model price = market (mid) price → this *is* the displayed IV |

IV is not an input Barchart looks up — it is the number that makes the chosen
model reprice the option to the market mid. That inversion is the whole game,
so *which model you invert* is exactly the question.

### Do we need a time series for the "time-dependent" parameters?

No — mind the two different "times":

- **Calendar time `t`** (the snapshot). Implied vol needs **no history**: it is
  backed out of the *current* price. Barchart displaying a per-option IV column
  is itself proof they invert a **constant-parameter BSM per option**.
- **Time-to-maturity `T`.** The maturity-dependent parameters — the vol surface
  `σ(K,T)` and rate curve `r(T)` — are **cross-sectional**: you get them from
  many strikes/expirations (and the Treasury curve) at **one** instant.
  `src/term_structure.py` builds `σ_ATM(T)` from SPCX's ~19 live expirations;
  it comes out cleanly downward-sloping (~95% at weeks → ~68% at 2y+).

You only need multiple **calendar dates** for *different* questions — realized
vol, IV dynamics, out-of-sample validation, or calibrating a stochastic-vol
model — none of which are needed to identify Barchart's pricing model.

---

## Conclusion — the closest model

**Black-Scholes-Merton with a continuous dividend yield, IV obtained by
inverting the model to the option's market mid.** Reasoning:

1. **BSM and Black-76 are the same model here.** Black-76 on the forward
   collapses to BSM for an equity/ETF. So the real contest is
   **European (BSM) vs American (CRR binomial)**.

2. **For a low-dividend name like SPCX the two are indistinguishable on the
   call side and most of the put side.** The `--demo` run (below) prices a
   realistic SPCX-like chain under both models:

   - **Calls:** max |CRR − BSM| price gap ≈ **$0.001**; IV misread ≈ **1 bp**.
     Below Barchart's display resolution ($0.01 price, 0.01% IV) → **identical**.
   - **Puts:** they agree near/out-of-the-money but diverge **deep in-the-money**,
     where the American early-exercise premium is real: the gap grows to
     ~**$0.18 / ~230 bp of IV** by 30% ITM.

3. Therefore the parsimonious model that reproduces the displayed chain to
   precision is **BSM**. The *only* place a snapshot can actually distinguish a
   binomial/American engine from BSM is **deep-ITM puts** — so that is where the
   empirical test should focus.

**How to prove it against the real page:** paste a chain snapshot (including a
few deep-ITM puts) into a CSV and run `--snapshot`. If Barchart's displayed IV
for the deep-ITM puts matches **CRR** better than **BSM**, they run a binomial
American model; if BSM wins everywhere, they display the European closed form.

### What the real Yahoo chain showed (`--snapshot`)

Inverting each model to the market **mid** on the actual 2026-12-18 chain:

| | RMSE vs Yahoo IV (full chain) | RMSE on liquid core (\|K/S−1\|≤0.25) | solver fails |
|--|--|--|--|
| BSM / Black-76 | 0.153 | 0.035 | 3 |
| CRR (American) | 0.114 | 0.031 | 1 |

- **On the liquid core the models are effectively tied** — the ~3.5 vol-point
  RMSE is just Yahoo computing IV from *last* while we invert from *mid*, not a
  model difference. This confirms BSM ≈ CRR where quotes are trustworthy.
- **CRR's full-chain edge is entirely from deep-ITM puts** (K/S up to 3.2,
  strikes $330–400 vs $124 spot). There `IV(BSM) − IV(CRR)` blows out to
  **1,800–2,960 bp** because a European model must distort IV wildly to reprice
  a near-intrinsic American put. Those strikes are illiquid, so this is *not*
  proof of Barchart's engine — it's proof European breaks down there.

**Bottom line:** for the liquid part of the SPCX chain that matters, BSM (=
Black-76) reproduces the market and is the closest simple model; the chain
can only distinguish American from European in the illiquid deep-ITM-put tail.
Confirming Barchart's exact engine needs their `bar_iv` pasted in.

---

## Run it

The pricing/harness code is **standard library only** (Python 3.10+).
The fetch step needs `yfinance` (`pip install yfinance`).

```shell
# 1) Download the real SPCX chain from Yahoo into data/
python src/fetch_chain.py --expiration 2026-12-18 --asof 2026-07-20

# 2) Score models against it (Yahoo IV as reference until you paste Barchart's)
python src/determine_model.py --snapshot data/spcx_2026-12-18.csv

# 3) Vol term structure across all expirations + realized vol contrast
python src/term_structure.py --symbol SPCX

# 3b) Vol SURFACE + greeks(T) + parameters(T) up to 12 months -> data/ + figures/
python src/surfaces.py --symbol SPCX --months 12

# 3c) Render the full static, self-contained dashboard HTML for any ticker
python src/build_dashboard.py --symbol SPCX      # -> dashboard.html

# 3d) Interactive: any ticker in a box (drag-to-rotate 3D, Strategies tab)
python src/app.py                                # -> http://127.0.0.1:8000

# 3e) Term-structure trade ideas + payoff curves (CLI; also a dashboard tab)
python src/strategies.py --symbol SPCX

# 4) Analytical determination + distinguishability report (no data needed)
python src/determine_model.py --demo
python src/determine_model.py --demo --spot 123.76 --rate 0.043 --div-yield 0 --t-days 151
```

All of `surfaces.py`, `build_dashboard.py` and `app.py` take **any `--symbol`**;
they fetch that ticker's chain, dividend yield, and expirations automatically.

### Capturing a real snapshot

The live page is Cloudflare-gated, so the reliable path is manual:
copy the option chain rows into a CSV shaped like
[`data/snapshot_template.csv`](data/snapshot_template.csv) — one row per
option, with Barchart's displayed IV/delta. Include several **deep in-the-money
puts** — that is the only region that separates the American from the European
model. Then run `--snapshot`.

---

## Files

```
spcx_model/
├── README.md                     ← you are here (hypotheses + conclusion)
├── dashboard.html                ← static, self-contained dashboard (build_dashboard output)
├── docs/
│   └── MODELS.md                 ← the math for models, surface, greeks(T), params(T)
├── src/
│   ├── pricing.py                ← BSM, CRR binomial, Black-76, IV solvers, greeks
│   ├── fetch_chain.py            ← download a chain / chain_rows() from Yahoo (any ticker)
│   ├── term_structure.py         ← σ_ATM(T) across expiries + realized vol
│   ├── surfaces.py               ← data-binned vol surface + greeks(T) + params(T);
│   │                                figure builders + interactive plotly 3D + plotly_js()
│   ├── determine_model.py        ← score_rows()/score_models() + --demo + --snapshot
│   ├── strategies.py             ← term-structure trades: analyze() + payoff curves
│   ├── build_dashboard.py        ← compute_context(symbol) + render_dashboard(); Analysis + Strategies tabs
│   └── app.py                    ← local web app: ticker box; serves bundled /plotly.js
├── data/
│   ├── audit/                    ← per-run audit bundles + index.csv (see below)
│   ├── spcx_2026-12-18.csv       ← real fetched chain (spot $123.76, 153 rows)
│   ├── vol_surface.csv           ← σ(K,T) surface points (≤12m, with raw mid)
│   ├── term_structure_params.csv ← ATM IV/greeks/rate/forward per expiry
│   ├── results_2026-12-18.txt    ← saved --snapshot output
│   └── snapshot_template.csv     ← schema / where to paste Barchart's own IV
└── figures/
    ├── vol_surface_heatmap.png   ← σ(K,T) heatmap (canonical view)
    ├── vol_surface_3d.png        ← σ(K,T) 3D
    ├── greeks_vs_maturity.png    ← ATM Δ,Γ,Θ,V,ρ vs T
    └── params_vs_maturity.png    ← σ_ATM(T), r(T), forward F(T)
```

## Surface / greeks(T) / parameters(T) — up to 12 months

All three are **cross-sectional from one snapshot** — every expiration within 12
months (for SPCX ~10–13), each option's IV backed out of its market mid by
inverting Black-Scholes with the ticker's `q`. `src/surfaces.py` writes the CSVs
and figures; `build_dashboard.py`/`app.py` render them for any ticker:

- **Vol surface σ(K,T)** — `figures/vol_surface_heatmap.png` + an **interactive
  3D** (drag to rotate, scroll to zoom). The heatmap **bins span the observed
  moneyness range** with a data-derived bin size (annotated in the title, e.g.
  `bins 0.88–1.17, Δm=0.01`) rather than a fixed window. Shows the smile across
  strikes and a downward **term structure** (short-dated ATM ~90% → ~72% at a
  year): backwardation, typical of an elevated-vol name.
- **Greeks(T)** — `figures/greeks_vs_maturity.png`. ATM Δ rises (forward drift),
  Γ decays ∝1/√T, vega grows ∝√T, Θ (per day) is most negative at the front,
  ρ grows ~linearly in T. Textbook BSM maturity behavior; underlying numbers in
  `data/term_structure_params.csv`.
- **Parameters(T)** — `figures/params_vs_maturity.png`: σ_ATM(T) (real, from the
  option cross-section), forward F(T)=S·e^{(r−q)T}, and r(T) (a **flat 4.3%
  placeholder** — swap `rate_for()` for a live Treasury curve to make it a true
  curve; IV is nearly insensitive to r at these maturities).

The interactive 3D uses **Plotly bundled locally** (no CDN): the app serves it
once at `/plotly.js` (browser-cached), and `build_dashboard.py` inlines it into
`dashboard.html` so the file — and a re-published Artifact — render the rotatable
surface offline.

## Strategies tab — term-structure trades with payoff curves

The dashboard has two tabs: **Analysis** (the surface / greeks / scoring above)
and **Strategies**. The Strategies tab reads the vol term structure and proposes
concrete trades that **sell the elevated near/event-window vol against the
cheaper long-dated backwardation**, each priced from live mids (`strategies.py`):

- **A) Call time butterfly** — long near + mid shoulders, short 2× the event
  tenor; near delta/vega-neutral, long gamma. Wins if the hump flattens.
- **B) ATM call calendar** — sell the event tenor, own long-dated vol; long
  back-vega, positive theta. Harvests the backwardation.
- **C) Bullish call diagonal** — long far ATM, short near OTM; net long delta
  financed by the rich event vol.

Each card shows the legs (strike/mid/IV/greeks), net premium + net greeks, and a
**payoff curve** evaluated at the earliest leg expiry (longer legs revalued by
BSM, IV held constant), with breakevens marked. Legs + summary are written to the
run's audit bundle (`strategy_legs.csv`, `strategies.csv`). **Illustrative only —
not investment advice**; SPCX spreads are wide, so price against real fills.

## Audit trail — every calculation is saved

Every run of the app or `build_dashboard.py` writes an **auditable bundle** of
the exact data used and the results to `data/audit/<SYMBOL>_<timestamp>/`:

| File | Contents |
|------|----------|
| `manifest.json` | run metadata (symbol, spot, q, rate, expiries, featured expiry, reference-IV source) + the full scoring summary and verdict — machine-readable |
| `surface_points.csv` | every OTM strike used, with the raw **`mid` (input) → `iv` (output)** |
| `term_structure.csv` | per-expiry ATM IV, all five greeks, rate, forward |
| `scoring_chain.csv` | the **raw** featured-expiry chain (bid/ask/last/Yahoo-IV) fed to the model scoring |
| `scoring_results.csv` | the per-model RMSE table |

`data/audit/index.csv` is an append-only log — one row per run (timestamp,
symbol, spot, q, featured expiry, reference IV, lowest-RMSE model, bundle path),
so you can trace any number on the dashboard back to the inputs it came from.
The dashboard footer prints the bundle path for that run.

## Workflow at a glance

```
        hypotheses            implementation             determination
   ┌──────────────────┐   ┌────────────────────┐   ┌──────────────────────┐
   │ BSM (European)   │   │ pricing.py          │   │ determine_model.py    │
   │ CRR (American)   │──▶│  price + greeks     │──▶│  --demo: how far      │
   │ Black-76 (=BSM)  │   │  IV by inversion    │   │    apart are they?    │
   └──────────────────┘   └────────────────────┘   │  --snapshot: which    │
   params: S,K,T,r,q,σ                             │    fits real data?     │
                                                    └──────────────────────┘
                                                    verdict: closest model
```
