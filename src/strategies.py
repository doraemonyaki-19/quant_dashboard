"""Concrete term-structure trades from the live chain, priced with real quotes.

Builds the structures proposed off the dashboard's vol term structure -- the
August event hump vs the backwardated belly -- using actual bid/ask/mid from
Yahoo, then reports net premium, net greeks, breakevens and a payoff curve
(evaluated at the earliest leg expiry, longer legs revalued by BSM with IV held
constant). `analyze()` is pure (no I/O) so the dashboard reuses it; the CLI
prints a report and saves an audit bundle.

    python src/strategies.py --symbol SPCX

Per share; a listed contract is x100. Greeks in dashboard display units
(delta per $1, theta per calendar day, vega per 1 vol point). NOT advice.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from pricing import OptionInputs, bsm_greeks, bsm_price, implied_vol_bsm  # noqa: E402
from surfaces import spot_price, div_yield, rate_for  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402  (surfaces set the Agg backend)


# --------------------------------------------------------------------------- #
# leg pricing
# --------------------------------------------------------------------------- #
def _mid(row):
  try:
    b, a = float(row["bid"]), float(row["ask"])
  except (TypeError, ValueError, KeyError):
    b = a = 0.0
  if b > 0 and a > 0:
    return 0.5 * (b + a), b, a
  last = float(row.get("lastPrice") or 0.0)
  return last, b, a


def leg_quote(t, expiration, strike, is_call, S, q, today, cache=None):
  """One option leg: nearest listed strike, mid/bid/ask, implied vol, greeks."""
  if cache is None:
    cache = {}
  if expiration not in cache:
    cache[expiration] = t.option_chain(expiration)
  ch = cache[expiration]
  df = ch.calls if is_call else ch.puts
  strikes = df["strike"].astype(float).tolist()
  K = min(strikes, key=lambda k: abs(k - strike))
  row = df[df["strike"].astype(float) == K].iloc[0].to_dict()
  mid, bid, ask = _mid(row)
  t_days = (dt.date.fromisoformat(expiration) - today).days
  T = t_days / 365.0
  r = rate_for(T)
  iv = implied_vol_bsm(OptionInputs(S, K, T, r, q, 0.5, is_call), mid) if mid > 0 \
      else float("nan")
  g = bsm_greeks(OptionInputs(S, K, T, r, q, iv, is_call)) if iv == iv else None
  return dict(expiration=expiration, t_days=t_days, strike=K, is_call=is_call,
              kind="C" if is_call else "P", mid=mid, bid=bid, ask=ask, iv=iv,
              T=T, r=r,
              delta=g.delta if g else float("nan"),
              gamma=g.gamma if g else float("nan"),
              theta=g.theta if g else float("nan"),
              vega=g.vega if g else float("nan"))


def net_greeks(legs):
  agg = dict(premium=0.0, delta=0.0, gamma=0.0, theta=0.0, vega=0.0)
  for L in legs:
    n = L["qty"]
    agg["premium"] += n * L["mid"]     # >0 = net debit paid
    for gk in ("delta", "gamma", "theta", "vega"):
      if L[gk] == L[gk]:
        agg[gk] += n * L[gk]
  return agg


# --------------------------------------------------------------------------- #
# payoff at the earliest leg expiry (longer legs revalued, IV held constant)
# --------------------------------------------------------------------------- #
def strategy_payoff(legs, net_premium, S, q, n=600):
  horizon = min(legs, key=lambda L: L["T"])
  hT = horizon["T"]
  grid = np.linspace(0.35 * S, 1.9 * S, n)
  vals = np.zeros(n)
  for L in legs:
    resid = L["T"] - hT
    for i, Sx in enumerate(grid):
      if resid <= 1e-9:
        v = max((Sx - L["strike"]) if L["is_call"] else (L["strike"] - Sx), 0.0)
      else:
        v = bsm_price(OptionInputs(Sx, L["strike"], resid, L["r"], q,
                                   L["iv"], L["is_call"]))
      vals[i] += L["qty"] * v
  pnl = vals - net_premium
  return grid, pnl, horizon["t_days"], horizon["expiration"]


def breakevens_from(grid, pnl):
  bes = []
  for i in range(1, len(pnl)):
    if pnl[i - 1] == 0 or pnl[i - 1] * pnl[i] < 0:
      x0, x1, y0, y1 = grid[i - 1], grid[i], pnl[i - 1], pnl[i]
      bes.append(round(float(x0 - y0 * (x1 - x0) / (y1 - y0)), 2))
  return bes


# --------------------------------------------------------------------------- #
# analysis (pure): symbol -> priced structures with payoff + greeks
# --------------------------------------------------------------------------- #
def analyze(symbol: str, ticker=None, spot=None, q=None):
  import yfinance as yf
  t = ticker or yf.Ticker(symbol)
  S = spot if spot is not None else spot_price(t)
  q = q if q is not None else div_yield(t, S)
  today = dt.date.today()
  exps = [e for e in t.options
          if (dt.date.fromisoformat(e) - today).days > 0]

  def pick(days):
    return min(exps, key=lambda e:
               abs((dt.date.fromisoformat(e) - today).days - days))

  e_near, e_hump, e_mid, e_long = pick(10), pick(28), pick(59), pick(150)
  Katm = round(S)
  cache = {}

  def lq(exp, k, call=True):
    return leg_quote(t, exp, k, call, S, q, today, cache)

  defs = []

  la = lq(e_near, Katm); la["qty"] = 1
  sa = lq(e_hump, Katm); sa["qty"] = -2
  ma = lq(e_mid, Katm); ma["qty"] = 1
  defs.append(("A) Call time butterfly (sell the Aug hump)",
               "Long near + mid shoulders, short 2x the event-window tenor. "
               "Sells the vol hump; near delta/vega-neutral, long gamma. "
               "Wins if the hump flattens.", [la, sa, ma]))

  sb = lq(e_hump, Katm); sb["qty"] = -1
  lb = lq(e_long, Katm); lb["qty"] = 1
  defs.append(("B) ATM call calendar (sell Aug / buy long)",
               "Sell rich event-window vol, own cheap long-dated vol. "
               "Long back-vega, short front-gamma; harvests the backwardation.",
               [sb, lb]))

  lc = lq(e_long, Katm); lc["qty"] = 1
  sc = lq(e_hump, round(S * 1.08)); sc["qty"] = -1
  defs.append(("C) Bullish call diagonal (long far / short near OTM hump)",
               "Own a long-dated ATM call; sell a near OTM call into the rich "
               "event-window vol to finance it. Net long delta + vega carry.",
               [lc, sc]))

  strategies = []
  for name, thesis, legs in defs:
    g = net_greeks(legs)
    grid, pnl, h_days, h_exp = strategy_payoff(legs, g["premium"], S, q)
    strategies.append(dict(
        name=name, thesis=thesis, legs=legs, net=g,
        premium=g["premium"], ptype="debit" if g["premium"] > 0 else "credit",
        breakevens=breakevens_from(grid, pnl), horizon_days=h_days,
        horizon_exp=h_exp, grid=grid, pnl=pnl,
        max_profit=round(float(np.max(pnl)), 4),
        max_loss=round(float(np.min(pnl)), 4)))

  return dict(symbol=symbol, spot=S, q=q, today=today, strikeATM=Katm,
              expiries=dict(near=e_near, hump=e_hump, mid=e_mid, long=e_long),
              strategies=strategies)


# --------------------------------------------------------------------------- #
# payoff figure (matplotlib) -- used by the dashboard Strategies tab
# --------------------------------------------------------------------------- #
def fig_payoff(strat, S):
  grid, pnl = strat["grid"], strat["pnl"]
  fig, ax = plt.subplots(figsize=(6.4, 3.5))
  ax.axhline(0, color="#8A95A2", lw=0.8)
  ax.fill_between(grid, pnl, 0, where=(pnl >= 0), color="#0E9E92", alpha=0.22)
  ax.fill_between(grid, pnl, 0, where=(pnl < 0), color="#C0392B", alpha=0.20)
  ax.plot(grid, pnl, color="#0E9E92", lw=1.8)
  ax.axvline(S, color="#C6791E", ls="--", lw=1.0)
  ax.annotate(f"spot ${S:.0f}", xy=(S, ax.get_ylim()[1]),
              xytext=(3, -12), textcoords="offset points",
              color="#C6791E", fontsize=8)
  for be in strat["breakevens"]:
    ax.plot([be], [0], "o", color="#10161D", ms=4)
    ax.annotate(f"${be:g}", xy=(be, 0), xytext=(0, 6),
                textcoords="offset points", ha="center", fontsize=7.5)
  ax.set_xlabel("underlying at horizon")
  ax.set_ylabel("P&L / share ($)")
  ax.set_title(f"Payoff @ {strat['horizon_exp']} ({strat['horizon_days']}d)  "
               f"{strat['ptype']} ${abs(strat['premium']):.2f}", fontsize=10)
  ax.grid(alpha=0.25)
  fig.tight_layout()
  return fig


# --------------------------------------------------------------------------- #
# CLI: report + audit bundle
# --------------------------------------------------------------------------- #
def report_and_save(res):
  S, q = res["spot"], res["q"]
  print("=" * 74)
  print(f"TERM-STRUCTURE TRADES  {res['symbol']}  spot=${S:.2f}  q={q:.3%}"
        f"  ATM~{res['strikeATM']}  ({res['today']})")
  print(f"  expiries: near={res['expiries']['near']} "
        f"hump={res['expiries']['hump']} mid={res['expiries']['mid']} "
        f"long={res['expiries']['long']}")
  print("=" * 74)
  for s in res["strategies"]:
    print(f"\n{s['name']}\n  {s['thesis']}")
    print(f"  {'leg':<7}{'exp':<12}{'d':>4} {'K':>6} {'mid':>7} {'IV%':>6} "
          f"{'delta':>7} {'theta/d':>8} {'vega':>6}")
    for L in s["legs"]:
      side = f"{'+' if L['qty']>0 else ''}{L['qty']}{L['kind']}"
      print(f"  {side:<7}{L['expiration']:<12}{L['t_days']:>4} {L['strike']:>6.0f} "
            f"{L['mid']:>7.2f} {L['iv']*100:>6.1f} {L['delta']:>7.3f} "
            f"{L['theta']:>8.3f} {L['vega']:>6.3f}")
    g = s["net"]
    print(f"  NET {s['ptype']} ${abs(s['premium']):.2f}/sh "
          f"(${abs(s['premium'])*100:.0f}/contract)  delta {g['delta']:+.3f}  "
          f"gamma {g['gamma']:+.4f}  theta {g['theta']:+.3f}/d  vega {g['vega']:+.3f}")
    print(f"  breakevens @ {s['horizon_exp']}: {s['breakevens']}  "
          f"max +${s['max_profit']:.2f}/sh, min ${s['max_loss']:.2f}/sh "
          f"(long legs' IV held const)")
  _save_audit(res)
  print("\nIllustrative only -- NOT investment advice. Check live fills, not mid.")


def _save_audit(res):
  now = dt.datetime.now()
  base = ROOT / "data" / "audit" / f"strategies_{res['symbol']}_{now:%Y%m%d-%H%M%S}"
  base.mkdir(parents=True, exist_ok=True)
  leg_rows, strat_rows = [], []
  for s in res["strategies"]:
    for L in s["legs"]:
      leg_rows.append(dict(strategy=s["name"], qty=L["qty"], kind=L["kind"],
                           expiration=L["expiration"], t_days=L["t_days"],
                           strike=L["strike"], bid=L["bid"], ask=L["ask"],
                           mid=round(L["mid"], 4), iv=round(L["iv"], 6),
                           delta=round(L["delta"], 5), gamma=round(L["gamma"], 6),
                           theta=round(L["theta"], 5), vega=round(L["vega"], 5)))
    g = s["net"]
    strat_rows.append(dict(strategy=s["name"], net_premium=round(s["premium"], 4),
                           type=s["ptype"], net_delta=round(g["delta"], 4),
                           net_gamma=round(g["gamma"], 5), net_theta=round(g["theta"], 4),
                           net_vega=round(g["vega"], 4),
                           breakevens="|".join(map(str, s["breakevens"])),
                           max_profit=s["max_profit"], max_loss=s["max_loss"],
                           horizon=s["horizon_exp"]))
  write_strategy_csvs(base, leg_rows, strat_rows)
  manifest = dict(symbol=res["symbol"], generated_local=now.isoformat(timespec="seconds"),
                  data_source="Yahoo Finance option chain (yfinance)",
                  spot=round(res["spot"], 4), dividend_yield_q=res["q"],
                  atm_strike=res["strikeATM"], expiries=res["expiries"],
                  assumptions=["mid=(bid+ask)/2", "greeks BSM at leg IV",
                               "payoff at earliest expiry; longer legs BSM, IV held",
                               "per share; contract x100; excl. commissions/slippage"],
                  disclaimer="Illustrative analysis, NOT investment advice.")
  (base / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
  idx = ROOT / "data" / "audit" / "strategies_index.csv"
  fresh = not idx.exists()
  with open(idx, "a", newline="", encoding="utf-8") as fh:
    w = csv.writer(fh)
    if fresh:
      w.writerow(["run_local", "symbol", "spot", "q", "atm", "dir"])
    w.writerow([now.isoformat(timespec="seconds"), res["symbol"],
                round(res["spot"], 4), res["q"], res["strikeATM"],
                base.relative_to(ROOT).as_posix()])
  print(f"\naudit -> {base.relative_to(ROOT).as_posix()}/")


def write_strategy_csvs(base: Path, leg_rows, strat_rows):
  """Write strategy_legs.csv + strategies.csv into an existing audit dir."""
  base.mkdir(parents=True, exist_ok=True)
  with open(base / "strategy_legs.csv", "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=list(leg_rows[0].keys()))
    w.writeheader(); w.writerows(leg_rows)
  with open(base / "strategies.csv", "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=list(strat_rows[0].keys()))
    w.writeheader(); w.writerows(strat_rows)


def main(argv=None):
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--symbol", default="SPCX")
  args = ap.parse_args(argv)
  report_and_save(analyze(args.symbol))


if __name__ == "__main__":
  main()
