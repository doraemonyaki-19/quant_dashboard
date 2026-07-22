"""Concrete term-structure trades from the live chain, priced with real quotes.

Reads the ticker's ATM vol term structure, classifies its shape (interior hump /
backwardation / upward / flat), and builds structures that sell the genuinely
richest tenor for that shape against cheaper long-dated vol -- a butterfly when
there is an interior hump to short, otherwise calendars/diagonals. Priced from
actual bid/ask/mid from Yahoo, with net premium, net greeks, breakevens and a
payoff curve (evaluated at the earliest leg expiry, longer legs revalued by BSM
with IV held constant). `analyze()` is pure (no I/O) so the dashboard reuses it;
the CLI prints a report and saves an audit bundle.

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


def option_requirement(legs):
  """Minimum options-approval level to place a structure, from its legs alone.

  Returns (level, reason). 'spread' = every short contract is covered by a long
  option of the same type held at least as long and at a no-worse strike (calls:
  long strike <= short; puts: long strike >= short) -> defined risk, tradeable
  without uncovered-option approval. 'naked' = some short leg is uncovered ->
  needs uncovered/naked approval. 'long' = no short legs at all. Stock is not
  considered (these structures are option-only)."""
  def covers(lo, sh):
    if lo["is_call"] != sh["is_call"]:
      return False
    if lo["T"] < sh["T"] - 1e-9:          # long must outlive the short
      return False
    return (lo["strike"] <= sh["strike"] + 1e-9 if sh["is_call"]
            else lo["strike"] >= sh["strike"] - 1e-9)

  longs, shorts = [], []
  for L in legs:
    n = int(round(abs(L["qty"])))
    if L["qty"] > 0:
      longs += [L] * n
    elif L["qty"] < 0:
      shorts += [L] * n
  if not shorts:
    return ("long", "no short legs")
  used = [False] * len(longs)
  for sh in shorts:                        # greedy match each short to a cover
    hit = next((i for i, lo in enumerate(longs)
                if not used[i] and covers(lo, sh)), None)
    if hit is None:
      return ("naked", "a short leg is not covered by a long option (uncovered)")
    used[hit] = True
  return ("spread", "every short leg is covered by a long option (defined risk)")


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

  Katm = round(S)
  cache = {}

  def lq(exp, k, call=True):
    return leg_quote(t, exp, k, call, S, q, today, cache)

  def put_equiv(L):
    """The same leg (strike/expiry/qty) expressed as a put instead of a call."""
    p = leg_quote(t, L["expiration"], L["strike"], False, S, q, today, cache)
    p["qty"] = L["qty"]
    return p

  # Distinct tenors across the front-to-mid-to-long span, deduped by expiry.
  tenors = []
  for target in (12, 35, 70, 150):
    e = pick(target)
    if e not in tenors:
      tenors.append(e)
  tenors.sort(key=lambda e: (dt.date.fromisoformat(e) - today).days)
  if len(tenors) < 2:
    return dict(symbol=symbol, spot=S, q=q, today=today, strikeATM=Katm,
                shape="insufficient expiries", term=[], strategies=[])

  # ATM leg (hence ATM IV) for each tenor -> the real term structure.
  atm = {e: lq(e, Katm) for e in tenors}
  term = [dict(expiry=e, t_days=atm[e]["t_days"], iv=atm[e]["iv"]) for e in tenors]
  ivs = [atm[e]["iv"] for e in tenors]

  # Shape: is the richest vol at the front, in an interior hump, or the back?
  sellable = tenors[:-1]                       # never sell the longest (own it)
  sell_e = max(sellable, key=lambda e: atm[e]["iv"])
  long_e = tenors[-1]
  hump_i = None
  for i in range(1, len(tenors) - 1):
    if ivs[i] > ivs[i - 1] + 0.01 and ivs[i] > ivs[i + 1] + 0.01:
      if hump_i is None or ivs[i] > ivs[hump_i]:
        hump_i = i
  if hump_i is not None:
    shape = f"a vol hump at {atm[tenors[hump_i]]['t_days']}d"
  elif ivs[0] - ivs[-1] > 0.01:
    shape = "backwardation (rich front, cheap back)"
  elif ivs[-1] - ivs[0] > 0.01:
    shape = "upward term structure (cheap front, rich back)"
  else:
    shape = "a roughly flat term structure"
  seq = " -> ".join(f"{atm[e]['t_days']}d {atm[e]['iv']*100:.0f}%" for e in tenors)

  def pct(e):
    return f"{atm[e]['iv']*100:.0f}%"

  defs = []

  # A) Primary vol sale: butterfly if there's an interior hump to short,
  #    otherwise a calendar that sells the genuinely richest tenor.
  if hump_i is not None:
    lo, hi = tenors[hump_i - 1], tenors[hump_i + 1]
    h = tenors[hump_i]
    la = dict(atm[lo], qty=1); sa = dict(atm[h], qty=-2); ma = dict(atm[hi], qty=1)
    defs.append((f"A) Term-vol butterfly — sell the {atm[h]['t_days']}d peak",
                 f"Term structure shows {shape} ({seq}). Long the "
                 f"{atm[lo]['t_days']}d/{atm[hi]['t_days']}d shoulders, short 2x "
                 f"the {atm[h]['t_days']}d peak ({pct(h)}); near delta/vega-"
                 "neutral, long gamma. Wins if the peak flattens.",
                 [la, sa, ma]))
  else:
    s = dict(atm[sell_e], qty=-1); l = dict(atm[long_e], qty=1)
    defs.append((f"A) Calendar — sell rich {atm[sell_e]['t_days']}d vol",
                 f"Term structure shows {shape} ({seq}). Sell the richest tenor "
                 f"({atm[sell_e]['t_days']}d @ {pct(sell_e)}) and own cheaper "
                 f"long-dated vol ({atm[long_e]['t_days']}d @ {pct(long_e)}). "
                 "Long back-vega, short front-gamma, positive carry.",
                 [s, l]))

  # B) A second calendar at a different short tenor (distinct from A's short).
  used_short = sell_e if hump_i is None else tenors[hump_i]
  alt = [e for e in sellable if e != used_short]
  if alt:
    alt_e = max(alt, key=lambda e: atm[e]["iv"])
    s = dict(atm[alt_e], qty=-1); l = dict(atm[long_e], qty=1)
    defs.append((f"B) Calendar — sell {atm[alt_e]['t_days']}d vs {atm[long_e]['t_days']}d",
                 f"Second calendar at the {atm[alt_e]['t_days']}d tenor "
                 f"({pct(alt_e)}) against the {atm[long_e]['t_days']}d "
                 f"({pct(long_e)}). Same long-vega / positive-carry idea at a "
                 "different point on the curve.", [s, l]))

  # C) Bullish diagonal: own long-dated ATM, sell a near OTM at the rich tenor.
  lc = dict(atm[long_e], qty=1)
  sc = lq(used_short, round(S * 1.08)); sc["qty"] = -1
  defs.append((f"C) Bullish diagonal — long {atm[long_e]['t_days']}d / short {atm[used_short]['t_days']}d OTM",
               f"Own a {atm[long_e]['t_days']}d ATM call; sell a "
               f"{atm[used_short]['t_days']}d OTM call ({round(S*1.08)} strike, "
               f"{sc['iv']*100:.0f}% IV) to finance it. Net long delta + vega "
               "carry; bullish tilt.", [lc, sc]))

  strategies = []
  for name, thesis, legs in defs:
    g = net_greeks(legs)
    grid, pnl, h_days, h_exp = strategy_payoff(legs, g["premium"], S, q)
    # Static delta hedge: hold `hedge_shares` of the underlying (per option unit)
    # to zero net delta at inception, held to horizon (NOT rehedged). Stock P&L
    # over the horizon is hedge_shares*(S_x - S); adding it isolates the vol/
    # term-structure edge from directional drift and reveals the gamma curvature.
    hedge_shares = -g["delta"]
    pnl_h = pnl + hedge_shares * (grid - S)
    # Put-equivalent legs: same strikes/expiries/signs, puts instead of calls.
    # By put-call parity the structure's vega/gamma/theta and payoff shape are
    # nearly identical; premium, net delta and assignment/margin differ, so it's
    # offered so you can trade whichever side fills better.
    legs_put = [put_equiv(L) for L in legs]
    gp = net_greeks(legs_put)
    req_lvl, req_reason = option_requirement(legs)
    reqp_lvl, reqp_reason = option_requirement(legs_put)
    grid_p, pnl_p, _, _ = strategy_payoff(legs_put, gp["premium"], S, q)
    hedge_shares_put = -gp["delta"]
    pnl_ph = pnl_p + hedge_shares_put * (grid_p - S)
    strategies.append(dict(
        name=name, thesis=thesis, legs=legs, net=g,
        premium=g["premium"], ptype="debit" if g["premium"] > 0 else "credit",
        breakevens=breakevens_from(grid, pnl), horizon_days=h_days,
        horizon_exp=h_exp, grid=grid, pnl=pnl,
        max_profit=round(float(np.max(pnl)), 4),
        max_loss=round(float(np.min(pnl)), 4),
        hedge_shares=round(float(hedge_shares), 4), pnl_hedged=pnl_h,
        breakevens_hedged=breakevens_from(grid, pnl_h),
        max_profit_hedged=round(float(np.max(pnl_h)), 4),
        max_loss_hedged=round(float(np.min(pnl_h)), 4),
        legs_put=legs_put, net_put=gp, premium_put=gp["premium"],
        ptype_put="debit" if gp["premium"] > 0 else "credit",
        hedge_shares_put=round(float(hedge_shares_put), 4),
        grid_put=grid_p, pnl_put=pnl_p, pnl_hedged_put=pnl_ph,
        breakevens_put=breakevens_from(grid_p, pnl_p),
        breakevens_hedged_put=breakevens_from(grid_p, pnl_ph),
        max_profit_put=round(float(np.max(pnl_p)), 4),
        max_loss_put=round(float(np.min(pnl_p)), 4),
        max_profit_hedged_put=round(float(np.max(pnl_ph)), 4),
        max_loss_hedged_put=round(float(np.min(pnl_ph)), 4),
        requires=req_lvl, requires_reason=req_reason,
        requires_put=reqp_lvl, requires_put_reason=reqp_reason))

  return dict(symbol=symbol, spot=S, q=q, today=today, strikeATM=Katm,
              shape=shape, term=term, sell_expiry=sell_e, long_expiry=long_e,
              strategies=strategies)


# --------------------------------------------------------------------------- #
# payoff figure (matplotlib) -- used by the dashboard Strategies tab
# --------------------------------------------------------------------------- #
def fig_payoff(strat, S, variant="call"):
  """Payoff curve for one structure. variant='call' (default) or 'put' selects
  which set of legs to plot; both share horizon/spot, so put-equivalent trades
  get their own curve (parity => similar shape, different breakevens/premium)."""
  sfx = "_put" if variant == "put" else ""
  grid = strat["grid" + sfx]
  pnl = strat["pnl" + sfx]
  fig, ax = plt.subplots(figsize=(6.4, 3.5))
  ax.axhline(0, color="#8A95A2", lw=0.8)
  ax.fill_between(grid, pnl, 0, where=(pnl >= 0), color="#0E9E92", alpha=0.22)
  ax.fill_between(grid, pnl, 0, where=(pnl < 0), color="#C0392B", alpha=0.20)
  ax.plot(grid, pnl, color="#0E9E92", lw=1.8, label="unhedged")
  if ("pnl_hedged" + sfx) in strat:
    ax.plot(grid, strat["pnl_hedged" + sfx], color="#10161D", lw=1.5, ls="--",
            label=f"delta-hedged ({strat['hedge_shares' + sfx]:+.2f} sh/unit)")
    ax.legend(loc="best", fontsize=7.5, framealpha=0.85)
  ax.axvline(S, color="#C6791E", ls="--", lw=1.0)
  ax.annotate(f"spot ${S:.0f}", xy=(S, ax.get_ylim()[1]),
              xytext=(3, -12), textcoords="offset points",
              color="#C6791E", fontsize=8)
  for be in strat["breakevens" + sfx]:
    ax.plot([be], [0], "o", color="#10161D", ms=4)
    ax.annotate(f"${be:g}", xy=(be, 0), xytext=(0, 6),
                textcoords="offset points", ha="center", fontsize=7.5)
  ax.set_xlabel("underlying at horizon")
  ax.set_ylabel("P&L / share ($)")
  legs_lbl = "put legs" if variant == "put" else "call legs"
  ax.set_title(f"Payoff @ {strat['horizon_exp']} ({strat['horizon_days']}d)  "
               f"{strat['ptype' + sfx]} ${abs(strat['premium' + sfx]):.2f}  "
               f"[{legs_lbl}]", fontsize=10)
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
  print(f"  shape: {res.get('shape', 'n/a')}")
  if res.get("term"):
    print("  ATM term structure: "
          + " -> ".join(f"{d['t_days']}d {d['iv']*100:.0f}%" for d in res["term"]))
  print("=" * 74)
  if not res["strategies"]:
    print("No strategies (insufficient distinct expiries)."); return
  req_lbl = {"spread": "SPREAD (defined risk)", "naked": "NAKED short (uncovered)",
             "long": "long only"}
  for s in res["strategies"]:
    print(f"\n{s['name']}\n  {s['thesis']}")
    print(f"  requires: {req_lbl.get(s['requires'], s['requires'])}"
          f"  |  put version: {req_lbl.get(s['requires_put'], s['requires_put'])}")
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
    print("  put-equivalent legs (same strikes; parity -> ~same vol trade):")
    for L in s["legs_put"]:
      side = f"{'+' if L['qty']>0 else ''}{L['qty']}{L['kind']}"
      print(f"  {side:<7}{L['expiration']:<12}{L['t_days']:>4} {L['strike']:>6.0f} "
            f"{L['mid']:>7.2f} {L['iv']*100:>6.1f} {L['delta']:>7.3f} "
            f"{L['theta']:>8.3f} {L['vega']:>6.3f}")
    gp = s["net_put"]
    print(f"  NET {s['ptype_put']} ${abs(s['premium_put']):.2f}/sh "
          f"(${abs(s['premium_put'])*100:.0f}/contract)  delta {gp['delta']:+.3f}  "
          f"gamma {gp['gamma']:+.4f}  theta {gp['theta']:+.3f}/d  vega {gp['vega']:+.3f}"
          f"  hedge {s['hedge_shares_put']:+.3f} sh/unit")
    print(f"    put breakevens @ {s['horizon_exp']}: {s['breakevens_put']}  "
          f"max +${s['max_profit_put']:.2f}/sh, min ${s['max_loss_put']:.2f}/sh")
    print(f"  breakevens @ {s['horizon_exp']}: {s['breakevens']}  "
          f"max +${s['max_profit']:.2f}/sh, min ${s['max_loss']:.2f}/sh "
          f"(long legs' IV held const)")
    hs = s["hedge_shares"]
    side = "long" if hs > 0 else "short"
    print(f"  delta hedge: {side} {abs(hs):.3f} sh/unit "
          f"({abs(hs)*100:.0f} sh per contract) -> delta-neutral at spot; "
          f"hedged @ horizon: max +${s['max_profit_hedged']:.2f}/sh, "
          f"min ${s['max_loss_hedged']:.2f}/sh, "
          f"BE {s['breakevens_hedged'] or 'none'}")
  _save_audit(res)
  print("\nIllustrative only -- NOT investment advice. Check live fills, not mid.")


def _save_audit(res):
  now = dt.datetime.now()
  base = ROOT / "data" / "audit" / f"strategies_{res['symbol']}_{now:%Y%m%d-%H%M%S}"
  base.mkdir(parents=True, exist_ok=True)
  leg_rows, strat_rows = [], []
  for s in res["strategies"]:
    for variant, legs in (("call", s["legs"]), ("put", s["legs_put"])):
      for L in legs:
        leg_rows.append(strategy_leg_row(s["name"], variant, L))
    g, gp = s["net"], s["net_put"]
    strat_rows.append(dict(strategy=s["name"], net_premium=round(s["premium"], 4),
                           type=s["ptype"], net_delta=round(g["delta"], 4),
                           net_gamma=round(g["gamma"], 5), net_theta=round(g["theta"], 4),
                           net_vega=round(g["vega"], 4),
                           requires=s["requires"], put_requires=s["requires_put"],
                           hedge_shares=s["hedge_shares"],
                           breakevens="|".join(map(str, s["breakevens"])),
                           max_profit=s["max_profit"], max_loss=s["max_loss"],
                           breakevens_hedged="|".join(map(str, s["breakevens_hedged"])),
                           max_profit_hedged=s["max_profit_hedged"],
                           max_loss_hedged=s["max_loss_hedged"],
                           put_net_premium=round(s["premium_put"], 4),
                           put_type=s["ptype_put"], put_net_delta=round(gp["delta"], 4),
                           put_net_vega=round(gp["vega"], 4),
                           put_hedge_shares=s["hedge_shares_put"],
                           put_breakevens="|".join(map(str, s["breakevens_put"])),
                           put_max_profit=s["max_profit_put"],
                           put_max_loss=s["max_loss_put"],
                           horizon=s["horizon_exp"]))
  write_strategy_csvs(base, leg_rows, strat_rows)
  manifest = dict(symbol=res["symbol"], generated_local=now.isoformat(timespec="seconds"),
                  data_source="Yahoo Finance option chain (yfinance)",
                  spot=round(res["spot"], 4), dividend_yield_q=res["q"],
                  atm_strike=res["strikeATM"], term_structure_shape=res.get("shape"),
                  atm_term_structure=res.get("term"),
                  assumptions=["mid=(bid+ask)/2", "greeks BSM at leg IV",
                               "payoff at earliest expiry; longer legs BSM, IV held",
                               "delta hedge = -net_delta shares/unit, static (set at "
                               "inception, held to horizon, not rehedged)",
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


def strategy_leg_row(strategy: str, variant: str, L: dict) -> dict:
  """One audit row for a strategy leg; `variant` is 'call' or 'put'."""
  return dict(strategy=strategy, variant=variant, qty=L["qty"], kind=L["kind"],
              expiration=L["expiration"], t_days=L["t_days"], strike=L["strike"],
              bid=L["bid"], ask=L["ask"], mid=round(L["mid"], 4), iv=round(L["iv"], 6),
              delta=round(L["delta"], 5), gamma=round(L["gamma"], 6),
              theta=round(L["theta"], 5), vega=round(L["vega"], 5))


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
