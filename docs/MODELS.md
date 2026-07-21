# Models & Parameters

The math behind each candidate in `src/pricing.py`, and how to source the
parameters. All rates continuously compounded; `T` in years; `σ` annualized.

## Shared quantities

```
d1 = [ ln(S/K) + (r − q + ½σ²)·T ] / (σ·√T)
d2 = d1 − σ·√T
F  = S·e^{(r−q)·T}          (forward price of the underlying)
N(·) = standard normal CDF,  φ(·) = standard normal PDF
```

## 1. Black-Scholes-Merton (European, dividend yield q)  — the hypothesis

```
Call = S·e^{−qT}·N(d1) − K·e^{−rT}·N(d2)
Put  = K·e^{−rT}·N(−d2) − S·e^{−qT}·N(−d1)
```

Closed-form greeks (returned in Barchart display units by `bsm_greeks`):

| Greek | Formula (call) | Display unit |
|-------|----------------|--------------|
| Delta | `e^{−qT}·N(d1)` | per $1 |
| Gamma | `e^{−qT}·φ(d1) / (S·σ·√T)` | per $1² |
| Vega  | `S·e^{−qT}·φ(d1)·√T` → ÷100 | per 1 vol point (1%) |
| Theta | `−S·e^{−qT}φ(d1)σ/(2√T) − rK·e^{−rT}N(d2) + qS·e^{−qT}N(d1)` → ÷365 | per calendar day |
| Rho   | `K·T·e^{−rT}·N(d2)` → ÷100 | per 1% rate |

Puts use the mirror forms (see `pricing.py`). **Implied vol** = solve
`BSM(σ) = market_mid` (Newton with analytic vega, bisection fallback).

## 2. Cox-Ross-Rubinstein binomial (American)

`n`-step tree (`crr_price`, default 400 steps; demo uses up to 1500 for smoothness):

```
dt = T/n,   u = e^{σ√dt},   d = 1/u,   p = (e^{(r−q)dt} − d)/(u − d)
```

Backward induction takes `max(continuation, intrinsic)` at every node → captures
**early-exercise premium**. Greeks by finite difference (`crr_greeks`). This is
the theoretically correct model for listed US options (all American-exercise).
It converges to BSM as early-exercise value → 0 (i.e., low `q`, no near-dated
dividend). The `--snapshot`/dashboard IV inversion uses **160 steps** for speed
(ample for basis-point work over a 150-row chain); the demo/greeks go up to 1500.

## 3. Black-76 (European on the forward)

```
Call = e^{−rT}·[ F·N(d1) − K·N(d2) ],   with d1,d2 built from F
```

Substituting `F = S·e^{(r−q)T}` makes this **identical** to BSM for an equity.
Kept in the candidate set only to demonstrate it is not a separate hypothesis;
the harness confirms it ties BSM to the last basis point.

---

## 4. Volatility surface, greeks(T) & parameter term structure

Everything here is **cross-sectional** — built from one snapshot's expiration
cross-section (all expiries ≤ 12 months), never a time series. Implemented in
`src/surfaces.py`; rendered for any ticker by `build_dashboard.py` / `app.py`.

### Per-option implied vol
For each expiration and each **out-of-the-money** strike (calls for `K ≥ S`,
puts for `K < S` — tight quotes, no early-exercise noise), solve BSM to the
market mid: the `σ(K,T)` with `BSM(σ) = mid`. OTM-only keeps the sheet clean;
wide near-intrinsic deep-ITM quotes are excluded.

### Heatmap moneyness bins (data-driven)
The heatmap bins over the **observed** moneyness range, not a fixed window
(`data_moneyness_grid`):
```
m_min, m_max = min / max observed K/S across all expiries
Δm   = nice_step( (m_max − m_min) / target_bins )         # target_bins ≈ 48
       nice_step ∈ {0.0025, .005, .01, .02, .025, .05, .1}
grid = floor(m_min/Δm)·Δm  …  ceil(m_max/Δm)·Δm   (step Δm)
```
Each expiry's smile is interpolated onto this common grid (`build_grid`); the
range and `Δm` are printed in the figure title (e.g. `bins 0.88–1.17, Δm=0.01`).

### ATM greeks(T)
Per expiration the ATM strike is the one nearest the **forward** `F = S·e^{(r−q)T}`,
and ATM IV is the smile interpolated at `K = F`. The five greeks are the
closed-form BSM values (`bsm_greeks`) at that `(K, σ_ATM)`. Versus `T` they trace
the textbook shapes: Δ↑ (forward drift), Γ ∝ 1/√T, vega ∝ √T, Θ/day most negative
at the front, ρ ~linear.

### Parameter term structure
`σ_ATM(T)` (data-driven), `F(T) = S·e^{(r−q)T}`, and `r(T)` (flat placeholder in
`rate_for()` — swap for a live Treasury curve). Saved to
`data/term_structure_params.csv`.

### Interactive 3D
`plotly_surface_html()` renders the same surface as a rotatable Plotly 3D
(drag / scroll / double-click). Plotly is **bundled locally** — inlined into the
static `dashboard.html`, or served once at `/plotly.js` by the app (no CDN), so
the surface rotates offline.

---

## Parameter sourcing (edit `DEFAULT_PARAMS` in `src/determine_model.py`)

| Param | How to get it | Notes |
|-------|---------------|-------|
| `spot` (S) | SPCX last/mark at the moment you copy the chain | use the same S for every row of a snapshot |
| `t_days` | days from snapshot date to **2026-12-18** | `T = t_days/365`. Vendors differ on 365 vs 252/trading-day count — try both if a fit is off |
| `rate` (r) | ~3-month US Treasury yield, cont-comp | small effect on IV; larger on rho |
| `div_yield` (q) | computed **per ticker** from trailing-12-month dividends (`div_yield()` in `surfaces.py`: `Σ dividends_last_365d / spot`, sanity-capped at 50%). **SPCX = 0** | With q=0 an American **call** equals the European call (never exercise early), so BSM ≡ CRR for all calls; only **puts** can differ. For `q > 0`, early call exercise near ex-dividend dates also matters |
| `sigma` (σ) | *not supplied* — solved per option | the solved σ **is** the IV Barchart displays |

### Day-count / rate conventions worth testing
If BSM reconstructs Barchart's IV with a small constant bias, the usual causes
are (a) trading-day vs calendar-day `T`, (b) mid vs last as the target price,
(c) the exact `r` and `q`. The `--snapshot` RMSE will drop sharply once these
match, which is itself evidence about their conventions.

### Two "time" axes — what needs history and what doesn't
Do **not** confuse calendar time `t` with time-to-maturity `T`.

- **Implied vol needs no time series.** Each option's IV is backed out of its
  *current* price by inverting BSM. Barchart showing a per-option IV column is
  evidence they invert a **constant-parameter BSM per option**, not one global
  vol. A single snapshot is sufficient to reconstruct/verify their numbers.
- **Maturity-dependent structure is cross-sectional.** `σ(K,T)` (smile × term
  structure) and `r(T)` come from many strikes/expirations and the Treasury
  curve **at one instant** — see `src/term_structure.py`, which builds
  `σ_ATM(T)` from SPCX's ~19 live expirations. No extra calendar dates needed.
- **Multiple calendar dates are needed only for different questions**: realized
  vol, IV *dynamics*, out-of-sample model validation, or calibrating a
  stochastic-vol model. None are required to identify Barchart's pricing model.
  (SPCX is newly listed — its 60d and 120d realized vols are identical because
  the history is short, another reason to use the implied cross-section.)
