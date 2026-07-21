"""SPCX vol term structure (cross-sectional) vs realized vol (time-series).

Purpose: make explicit the two different "time" axes behind the parameters.

  1. IV term structure  sigma_ATM(T)  -- a function of TIME-TO-MATURITY T.
     Recovered from MANY EXPIRATIONS at ONE snapshot (cross-sectional). No
     price history needed: each ATM IV is implied from that option's current
     mid by inverting Black-Scholes (q=0 for SPCX).

  2. Realized volatility  sigma_realized  -- needs a PRICE TIME SERIES
     (many calendar dates). Shown here only to contrast: this is the quantity
     that actually requires history, and it is NOT what Barchart inverts.

  3. Rate term structure r(T): one rate per maturity, read off the Treasury
     curve at the snapshot. Approximated here as flat (edit RATE_CURVE); the
     ATM IV solve is nearly insensitive to r at these maturities.

Usage:
    python src/term_structure.py --symbol SPCX
"""

from __future__ import annotations

import argparse
import datetime as dt
import math
import statistics
import sys
from pathlib import Path

import yfinance as yf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pricing import OptionInputs, implied_vol_bsm  # noqa: E402

Q_SPCX = 0.0  # SPCX pays no dividend

# Crude flat rate; replace with an actual Treasury curve r(T) if you want.
def rate_for(_t_years: float) -> float:
  return 0.043


def spot_price(t: yf.Ticker) -> float:
  for g in (lambda: t.fast_info.get("last_price"),
            lambda: t.fast_info.get("previous_close"),
            lambda: float(t.history(period="5d")["Close"].iloc[-1])):
    try:
      v = g()
      if v:
        return float(v)
    except Exception:
      pass
  raise RuntimeError("no spot")


def atm_iv(chain_df, S, T, r, is_call):
  """Invert BSM (q=0) at the strike nearest spot to get ATM IV from the mid."""
  best = None
  for _, row in chain_df.iterrows():
    K = float(row["strike"])
    d = abs(K - S)
    if best is None or d < best[0]:
      bid, ask = row.get("bid"), row.get("ask")
      last = row.get("lastPrice")
      try:
        bid, ask = float(bid), float(ask)
      except (TypeError, ValueError):
        bid = ask = 0.0
      mid = 0.5 * (bid + ask) if bid > 0 and ask > 0 else float(last or 0)
      yiv = row.get("impliedVolatility")
      best = (d, K, mid, float(yiv) if yiv else float("nan"))
  if not best or best[2] <= 0:
    return None
  _, K, mid, yiv = best
  o = OptionInputs(S, K, T, r, Q_SPCX, 0.5, is_call)
  iv = implied_vol_bsm(o, mid)
  return (K, mid, iv, yiv)


def realized_vol(t: yf.Ticker, window: int) -> float:
  h = t.history(period=f"{max(window + 10, 40)}d")
  closes = list(h["Close"].dropna())[-(window + 1):]
  if len(closes) < 5:
    return float("nan")
  rets = [math.log(closes[i] / closes[i - 1]) for i in range(1, len(closes))]
  return statistics.pstdev(rets) * math.sqrt(252)


def main(argv=None):
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--symbol", default="SPCX")
  args = ap.parse_args(argv)

  t = yf.Ticker(args.symbol)
  S = spot_price(t)
  today = dt.date.today()
  print(f"{args.symbol}  spot={S:.4f}  asof={today}  q={Q_SPCX}")
  print()
  print("1) IV TERM STRUCTURE  sigma_ATM(T)  [cross-sectional: many expiries, 1 snapshot]")
  print(f"   {'expiry':<12} {'days':>5} {'ATM K':>7} {'BSM IV':>8} {'Yahoo IV':>9}")
  for exp in t.options:
    T_days = (dt.date.fromisoformat(exp) - today).days
    if T_days <= 0:
      continue
    T = T_days / 365.0
    r = rate_for(T)
    try:
      ch = t.option_chain(exp)
    except Exception:
      continue
    res = atm_iv(ch.calls, S, T, r, True)
    if not res:
      continue
    K, mid, iv, yiv = res
    ivs = f"{iv:8.4f}" if iv and not math.isnan(iv) else "     n/a"
    yivs = f"{yiv:9.4f}" if not math.isnan(yiv) else "      n/a"
    print(f"   {exp:<12} {T_days:>5} {K:>7.1f} {ivs} {yivs}")

  print()
  print("2) REALIZED VOL  [time-series: needs price history, NOT what Barchart inverts]")
  for w in (20, 60, 120):
    rv = realized_vol(t, w)
    rvs = f"{rv:.4f}" if not math.isnan(rv) else "n/a"
    print(f"   {w}-day realized vol (annualized) = {rvs}")

  print()
  print("Takeaway: sigma_ATM(T) above is the maturity-dependence of vol, all from")
  print("ONE snapshot's expiration cross-section. Realized vol is the only piece")
  print("that needs multiple calendar dates -- and it is a different quantity from")
  print("the implied vol Barchart displays.")


if __name__ == "__main__":
  main()
