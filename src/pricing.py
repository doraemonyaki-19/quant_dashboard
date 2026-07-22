"""Option-pricing library for the SPCX model-reconstruction workflow.

Pure standard library (math only) so it runs in any Python 3.10+ without
installing scipy/numpy. Everything here is used to *hypothesize* which model
Barchart uses for the SPCX option chain and to score candidates against a
snapshot of their displayed values.

Candidate models implemented
----------------------------
- Black-Scholes-Merton (BSM), European, continuous dividend yield q
- Cox-Ross-Rubinstein (CRR) binomial, American, early exercise + dividends
- Black-76, European option on a forward/future (F = S e^{(r-q)T})

All three take the same economic parameters; they differ only in whether
early exercise is allowed and whether the drift is applied to spot or forward.

Conventions
-----------
- Rates r, q are continuously compounded, annualized (decimals, e.g. 0.045).
- T is in years (calendar days / 365).
- sigma is annualized volatility (decimal, e.g. 0.55 == 55%).
- Prices are per share (a listed contract multiplies by 100).
- Greeks are returned in "display" units used by Barchart:
    delta  : per $1 move in underlying          (dimensionless, -1..1)
    gamma  : delta change per $1 move            (per $)
    theta  : price change PER CALENDAR DAY       (year theta / 365)
    vega   : price change per 1 vol POINT (1%)   (raw vega / 100)
    rho    : price change per 1% rate move       (raw rho / 100)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

SQRT_2PI = math.sqrt(2.0 * math.pi)


# --------------------------------------------------------------------------- #
# Normal distribution helpers (no scipy dependency)
# --------------------------------------------------------------------------- #
def _norm_cdf(x: float) -> float:
  """Standard normal CDF via the error function."""
  return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_pdf(x: float) -> float:
  return math.exp(-0.5 * x * x) / SQRT_2PI


# --------------------------------------------------------------------------- #
# Inputs / outputs
# --------------------------------------------------------------------------- #
@dataclass
class OptionInputs:
  spot: float          # S, underlying price
  strike: float        # K
  t: float             # T, years to expiry
  rate: float          # r, risk-free (cont. comp.)
  div_yield: float     # q, dividend yield (cont. comp.)
  sigma: float         # volatility
  is_call: bool


@dataclass
class Greeks:
  price: float
  delta: float
  gamma: float
  theta: float   # per calendar day
  vega: float    # per 1 vol point
  rho: float     # per 1% rate


# --------------------------------------------------------------------------- #
# Black-Scholes-Merton (European, dividend yield q)
# --------------------------------------------------------------------------- #
def bsm_price(o: OptionInputs) -> float:
  S, K, T, r, q, sig = o.spot, o.strike, o.t, o.rate, o.div_yield, o.sigma
  if T <= 0 or sig <= 0:
    intrinsic = (S - K) if o.is_call else (K - S)
    return max(intrinsic, 0.0)
  sqrtT = math.sqrt(T)
  d1 = (math.log(S / K) + (r - q + 0.5 * sig * sig) * T) / (sig * sqrtT)
  d2 = d1 - sig * sqrtT
  disc_r = math.exp(-r * T)
  disc_q = math.exp(-q * T)
  if o.is_call:
    return S * disc_q * _norm_cdf(d1) - K * disc_r * _norm_cdf(d2)
  return K * disc_r * _norm_cdf(-d2) - S * disc_q * _norm_cdf(-d1)


def bsm_greeks(o: OptionInputs) -> Greeks:
  """Closed-form BSM greeks in Barchart display units."""
  S, K, T, r, q, sig = o.spot, o.strike, o.t, o.rate, o.div_yield, o.sigma
  price = bsm_price(o)
  if T <= 0 or sig <= 0:
    delta = (1.0 if S > K else 0.0) if o.is_call else (-1.0 if S < K else 0.0)
    return Greeks(price, delta, 0.0, 0.0, 0.0, 0.0)
  sqrtT = math.sqrt(T)
  d1 = (math.log(S / K) + (r - q + 0.5 * sig * sig) * T) / (sig * sqrtT)
  d2 = d1 - sig * sqrtT
  disc_r = math.exp(-r * T)
  disc_q = math.exp(-q * T)
  pdf_d1 = _norm_pdf(d1)

  gamma = disc_q * pdf_d1 / (S * sig * sqrtT)
  vega_raw = S * disc_q * pdf_d1 * sqrtT               # per 1.00 vol
  if o.is_call:
    delta = disc_q * _norm_cdf(d1)
    theta_yr = (-S * disc_q * pdf_d1 * sig / (2 * sqrtT)
                - r * K * disc_r * _norm_cdf(d2)
                + q * S * disc_q * _norm_cdf(d1))
    rho_raw = K * T * disc_r * _norm_cdf(d2)           # per 1.00 rate
  else:
    delta = -disc_q * _norm_cdf(-d1)
    theta_yr = (-S * disc_q * pdf_d1 * sig / (2 * sqrtT)
                + r * K * disc_r * _norm_cdf(-d2)
                - q * S * disc_q * _norm_cdf(-d1))
    rho_raw = -K * T * disc_r * _norm_cdf(-d2)

  return Greeks(
      price=price,
      delta=delta,
      gamma=gamma,
      theta=theta_yr / 365.0,
      vega=vega_raw / 100.0,
      rho=rho_raw / 100.0,
  )


# --------------------------------------------------------------------------- #
# Black-76 (European option on the forward)
# --------------------------------------------------------------------------- #
def black76_price(o: OptionInputs) -> float:
  S, K, T, r, q, sig = o.spot, o.strike, o.t, o.rate, o.div_yield, o.sigma
  if T <= 0 or sig <= 0:
    intrinsic = (S - K) if o.is_call else (K - S)
    return max(intrinsic, 0.0)
  F = S * math.exp((r - q) * T)
  sqrtT = math.sqrt(T)
  d1 = (math.log(F / K) + 0.5 * sig * sig * T) / (sig * sqrtT)
  d2 = d1 - sig * sqrtT
  disc = math.exp(-r * T)
  if o.is_call:
    return disc * (F * _norm_cdf(d1) - K * _norm_cdf(d2))
  return disc * (K * _norm_cdf(-d2) - F * _norm_cdf(-d1))


# --------------------------------------------------------------------------- #
# Cox-Ross-Rubinstein binomial (American, early exercise)
# --------------------------------------------------------------------------- #
def crr_price(o: OptionInputs, steps: int = 400) -> float:
  """American option price via CRR binomial tree with dividend yield q."""
  S, K, T, r, q, sig = o.spot, o.strike, o.t, o.rate, o.div_yield, o.sigma
  if T <= 0 or sig <= 0:
    intrinsic = (S - K) if o.is_call else (K - S)
    return max(intrinsic, 0.0)
  dt = T / steps
  u = math.exp(sig * math.sqrt(dt))
  d = 1.0 / u
  disc = math.exp(-r * dt)
  p = (math.exp((r - q) * dt) - d) / (u - d)
  p = min(max(p, 0.0), 1.0)  # guard against tiny numerical drift

  # Precompute S*u^j and d^k once (O(steps)) so the O(steps^2) backward pass
  # never calls pow(): asset price at node (i, j) is upow[j]*dpow[i-j], which is
  # bit-identical to S*(u**j)*(d**(i-j)) but hoists the exponentiation out.
  is_call = o.is_call
  upow = [S * (u ** j) for j in range(steps + 1)]
  dpow = [d ** k for k in range(steps + 1)]
  pm = 1.0 - p  # precompute once; the discount stays factored out so the

  # arithmetic is bit-identical to disc*(p*v_up + (1-p)*v_dn) below.
  # terminal payoffs
  values = [0.0] * (steps + 1)
  for j in range(steps + 1):
    ST = upow[j] * dpow[steps - j]
    payoff = (ST - K) if is_call else (K - ST)
    values[j] = payoff if payoff > 0.0 else 0.0

  # backward induction with early exercise
  for i in range(steps - 1, -1, -1):
    for j in range(i + 1):
      cont = disc * (p * values[j + 1] + pm * values[j])
      ST = upow[j] * dpow[i - j]
      exercise = (ST - K) if is_call else (K - ST)
      values[j] = cont if cont > exercise else exercise
  return values[0]


def crr_greeks(o: OptionInputs, steps: int = 400) -> Greeks:
  """American greeks by finite difference around the CRR price."""
  price = crr_price(o, steps)
  h_s = max(o.spot * 1e-3, 1e-4)
  up = crr_price(_bump(o, spot=o.spot + h_s), steps)
  dn = crr_price(_bump(o, spot=o.spot - h_s), steps)
  delta = (up - dn) / (2 * h_s)
  gamma = (up - 2 * price + dn) / (h_s * h_s)

  h_v = 1e-3
  vprice = crr_price(_bump(o, sigma=o.sigma + h_v), steps)
  vega = (vprice - price) / h_v / 100.0

  h_t = min(1.0 / 365.0, o.t / 2) if o.t > 0 else 0.0
  theta = 0.0
  if h_t > 0:
    tprice = crr_price(_bump(o, t=o.t - h_t), steps)
    theta = (tprice - price) / h_t / 365.0  # per calendar day

  h_r = 1e-4
  rprice = crr_price(_bump(o, rate=o.rate + h_r), steps)
  rho = (rprice - price) / h_r / 100.0

  return Greeks(price, delta, gamma, theta, vega, rho)


def _bump(o: OptionInputs, **kw) -> OptionInputs:
  return OptionInputs(
      spot=kw.get("spot", o.spot),
      strike=kw.get("strike", o.strike),
      t=kw.get("t", o.t),
      rate=kw.get("rate", o.rate),
      div_yield=kw.get("div_yield", o.div_yield),
      sigma=kw.get("sigma", o.sigma),
      is_call=kw.get("is_call", o.is_call),
  )


# --------------------------------------------------------------------------- #
# Implied volatility solvers (invert a chosen model to match market price)
# --------------------------------------------------------------------------- #
_PRICERS = {
    "bsm": bsm_price,
    "black76": black76_price,
    "crr": crr_price,
}


def implied_vol(model: str, o: OptionInputs, target_price: float,
                lo: float = 1e-4, hi: float = 5.0, tol: float = 1e-5,
                steps: int = 160, max_iter: int = 60) -> float:
  """Solve for sigma so that model price == target_price.

  Bisection backbone (robust for the CRR tree, which has no clean analytic
  vega). `steps` sets CRR tree resolution during the solve — 160 is ample for
  basis-point IV work and keeps a 150-row chain fast. Returns NaN if the target
  is outside no-arbitrage bounds for the model.
  """
  base = _PRICERS[model]
  if model == "crr":
    pricer = lambda oo: crr_price(oo, steps)  # noqa: E731
  else:
    pricer = base

  def f(sig: float) -> float:
    return pricer(_bump(o, sigma=sig)) - target_price

  f_lo, f_hi = f(lo), f(hi)
  if f_lo * f_hi > 0:
    # Target not bracketed -> price below intrinsic or above cap. No IV.
    return float("nan")

  a, b = lo, hi
  for _ in range(max_iter):
    m = 0.5 * (a + b)
    fm = f(m)
    if abs(fm) < tol or (b - a) < 1e-6:
      return m
    if f_lo * fm < 0:
      b, f_hi = m, fm
    else:
      a, f_lo = m, fm
  return 0.5 * (a + b)


def implied_vol_bsm(o: OptionInputs, target_price: float) -> float:
  """Fast Newton IV for BSM (has analytic vega); falls back to bisection."""
  sig = 0.5
  for _ in range(100):
    g = bsm_greeks(_bump(o, sigma=sig))
    diff = g.price - target_price
    if abs(diff) < 1e-7:
      return sig
    vega_raw = g.vega * 100.0
    if vega_raw < 1e-8:
      break
    sig -= diff / vega_raw
    if sig <= 1e-4 or sig >= 5.0:
      break
  return implied_vol("bsm", o, target_price)


# --------------------------------------------------------------------------- #
# Convenience: full greek set for a named model
# --------------------------------------------------------------------------- #
def greeks_for(model: str, o: OptionInputs) -> Greeks:
  if model == "bsm":
    return bsm_greeks(o)
  if model == "crr":
    return crr_greeks(o)
  if model == "black76":
    # Black-76 greeks by finite difference (rarely the display model, but
    # included so the harness can score it on equal footing).
    price = black76_price(o)
    h = max(o.spot * 1e-3, 1e-4)
    up = black76_price(_bump(o, spot=o.spot + h))
    dn = black76_price(_bump(o, spot=o.spot - h))
    delta = (up - dn) / (2 * h)
    gamma = (up - 2 * price + dn) / (h * h)
    vprice = black76_price(_bump(o, sigma=o.sigma + 1e-3))
    vega = (vprice - price) / 1e-3 / 100.0
    theta = 0.0
    if o.t > 1.0 / 365.0:
      tprice = black76_price(_bump(o, t=o.t - 1.0 / 365.0))
      theta = (tprice - price) / (1.0 / 365.0) / 365.0
    rprice = black76_price(_bump(o, rate=o.rate + 1e-4))
    rho = (rprice - price) / 1e-4 / 100.0
    return Greeks(price, delta, gamma, theta, vega, rho)
  raise ValueError(f"unknown model {model!r}")
