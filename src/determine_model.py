"""Determine which option-pricing model best reproduces Barchart's SPCX chain.

Two modes
---------
1. `--demo`  (default, no data needed)
   Builds an SPCX-like option chain and quantifies how *distinguishable* the
   candidate models are at Barchart's display precision. Answers: "if Barchart
   used model A but I assume model B, how far off would the displayed IV /
   greeks be?" This tells us which model is closest and whether the data can
   even separate them.

2. `--snapshot path.csv`
   Scores each candidate model against a REAL Barchart snapshot you paste in
   (see data/snapshot_template.csv). For every row it inverts each model to the
   option's market (mid) price to get a model-IV, and also recomputes greeks at
   Barchart's reported IV, then reports the aggregate error per model and the
   winner.

Run:
    python src/determine_model.py --demo
    python src/determine_model.py --snapshot data/spcx_2026-12-18.csv
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pricing import (  # noqa: E402
    OptionInputs, Greeks, greeks_for, implied_vol, implied_vol_bsm, _bump,
    bsm_price, crr_price, crr_greeks, black76_price,
)

CANDIDATES = ["bsm", "crr", "black76"]
PRETTY = {"bsm": "Black-Scholes-Merton (European, q)",
          "crr": "CRR binomial (American)",
          "black76": "Black-76 (forward)"}


# --------------------------------------------------------------------------- #
# Parameters for SPCX (see docs/MODELS.md for sourcing + how to update)
# --------------------------------------------------------------------------- #
DEFAULT_PARAMS = dict(
    spot=123.76,      # SPCX spot from Yahoo on 2026-07-20 (fetch_chain.py) -- UPDATE
    rate=0.043,       # r: ~5M treasury (to 2026-12-18), cont-comp  -- UPDATE
    div_yield=0.0,    # q: SPCX pays NO dividend -> q=0 (American call == European)
    t_days=151,       # calendar days 2026-07-20 -> 2026-12-18 -- UPDATE
)


def price_of(model: str, o: OptionInputs) -> float:
  return {"bsm": bsm_price, "crr": crr_price, "black76": black76_price}[model](o)


# --------------------------------------------------------------------------- #
# DEMO: quantify model distinguishability for an SPCX-like chain
# --------------------------------------------------------------------------- #
def run_demo(params: dict) -> None:
  S = params["spot"]
  r = params["rate"]
  q = params["div_yield"]
  T = params["t_days"] / 365.0

  strikes = [round(S * m, 1) for m in
             (0.7, 0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2, 1.3)]
  sigma = 0.55  # SPCX-like: elevated vol name

  print("=" * 74)
  print("SPCX MODEL DISTINGUISHABILITY DEMO")
  print(f"  S={S}  r={r:.3%}  q={q:.3%}  T={T:.3f}y ({params['t_days']}d)"
        f"  assumed sigma={sigma:.0%}")
  print("=" * 74)
  print("Premise: US single-name/ETF options are AMERICAN. The theoretically")
  print("correct model is the binomial tree (CRR). Barchart *displays* IV and")
  print("greeks from a model. We measure how much the candidates differ at the")
  print("precision Barchart shows (IV 2dp as %, delta 4dp, price 2dp).")
  print()

  for kind, is_call in (("CALL", True), ("PUT", False)):
    print(f"--- {kind}S " + "-" * 60)
    print(f"{'K':>7} {'BSM px':>9} {'CRR px':>9} {'B76 px':>9} "
          f"{'|CRR-BSM|':>10} {'dIV(bp)':>9} {'d-delta':>9}")
    px_gaps, iv_gaps, delta_gaps = [], [], []
    for K in strikes:
      o = OptionInputs(S, K, T, r, q, sigma, is_call)
      p_bsm = bsm_price(o)
      p_crr = crr_price(o, steps=1200)  # more steps -> less tree sawtooth
      p_b76 = black76_price(o)

      # If the TRUE displayed price were the American (CRR) price, what IV
      # would a BSM-based reader back out, and how wrong is the delta?
      iv_bsm = implied_vol_bsm(o, p_crr)
      d_crr = crr_greeks(o, steps=1500).delta
      if not math.isnan(iv_bsm):
        d_bsm = greeks_for("bsm", _bump(o, sigma=iv_bsm)).delta
      else:
        d_bsm = float("nan")

      px_gap = abs(p_crr - p_bsm)
      iv_gap_bp = (iv_bsm - sigma) * 1e4 if not math.isnan(iv_bsm) else float("nan")
      delta_gap = (d_bsm - d_crr) if not math.isnan(d_bsm) else float("nan")

      px_gaps.append(px_gap)
      if not math.isnan(iv_gap_bp):
        iv_gaps.append(abs(iv_gap_bp))
      if not math.isnan(delta_gap):
        delta_gaps.append(abs(delta_gap))

      print(f"{K:>7.1f} {p_bsm:>9.4f} {p_crr:>9.4f} {p_b76:>9.4f} "
            f"{px_gap:>10.4f} {iv_gap_bp:>9.1f} {delta_gap:>9.5f}")

    print(f"  max |CRR-BSM| price gap : ${max(px_gaps):.4f}  "
          f"(Barchart shows price to $0.01)")
    if iv_gaps:
      print(f"  max IV misread (BSM vs American) : {max(iv_gaps):.1f} bp  "
            f"(Barchart shows IV to ~1 bp / 0.01%)")
    if delta_gaps:
      print(f"  max delta misread : {max(delta_gaps):.5f}  "
            f"(Barchart shows delta to 4dp)")
    print()

  print("=" * 74)
  print("READING THE RESULT")
  print("=" * 74)
  print(textwrap_fill(
      "For a low-dividend ETF like SPCX the early-exercise premium is tiny, so "
      "BSM (European) and CRR (American) prices nearly coincide -- typically "
      "within a cent except for deep in-the-money puts, where American puts "
      "carry a real early-exercise premium and the models separate. Black-76 "
      "on the forward is algebraically identical to BSM once the forward "
      "F = S*e^{(r-q)T} is used, so it is not a distinct hypothesis for equity "
      "options. Conclusion: BSM is the closest simple model and reproduces "
      "Barchart's displayed calls to precision; deep-ITM puts are the only "
      "place a snapshot can actually distinguish American (CRR) from BSM. Feed "
      "a real snapshot with --snapshot to confirm empirically."))


# --------------------------------------------------------------------------- #
# SNAPSHOT: score models against real Barchart values
# --------------------------------------------------------------------------- #
CORE = 0.25  # liquid core: |K/S - 1| <= CORE


def score_models(csv_path: str, params: dict) -> dict:
  """Score candidate models against a snapshot CSV; return structured results."""
  with open(csv_path, newline="", encoding="utf-8") as fh:
    lines = [ln for ln in fh
             if ln.strip() and not ln.lstrip().startswith("#")]
  rows = list(csv.DictReader(lines))
  return score_rows(rows, params, source=csv_path)


def score_rows(rows: list, params: dict, source: str = "in-memory") -> dict:
  """Score candidate models against in-memory option rows (list of dicts).

  Both the CLI (via score_models) and the web app consume this, so the numbers
  on the page are always the ones the harness computes. Each row needs keys:
  type, strike, spot, t_days, bid, ask, last, and a reference IV in bar_iv or
  yahoo_iv.
  """
  if not rows:
    return {}

  r = params["rate"]
  q = params["div_yield"]
  has_bar = any(_f(rr.get("bar_iv")) is not None for rr in rows)
  ref_col = "bar_iv" if has_bar else "yahoo_iv"
  ref_name = "Barchart" if has_bar else "Yahoo (proxy)"

  iv_err = {m: [] for m in CANDIDATES}    # (moneyness, |iv_model - iv_ref|)
  fails = {m: 0 for m in CANDIDATES}
  put_spread = []                          # (K/S, bsm_iv - crr_iv) in bp

  for row in rows:
    try:
      S = float(row["spot"])
      K = float(row["strike"])
      T = float(row["t_days"]) / 365.0
    except (TypeError, ValueError):
      continue  # skip malformed live rows
    is_call = str(row["type"]).strip().lower().startswith("c")
    mkt = _mid(row)
    if mkt <= 0:
      continue
    mny = K / S
    bar_iv = _f(row.get(ref_col))

    base = OptionInputs(S, K, T, r, q, 0.3, is_call)
    iv_bsm_mid = iv_crr_mid = None
    for m in CANDIDATES:
      iv = (implied_vol_bsm(base, mkt) if m == "bsm"
            else implied_vol(m, base, mkt))
      solved = iv is not None and not math.isnan(iv) and 1e-3 < iv < 4.99
      if not solved:
        fails[m] += 1
      if m == "bsm":
        iv_bsm_mid = iv if solved else None
      if m == "crr":
        iv_crr_mid = iv if solved else None
      if bar_iv is not None and solved:
        iv_err[m].append((mny, abs(iv - bar_iv)))

    if not is_call and iv_bsm_mid and iv_crr_mid:
      put_spread.append((mny, (iv_bsm_mid - iv_crr_mid) * 1e4))

  def rmse(recs, core_only=False):
    vals = [e for mny, e in recs if (abs(mny - 1) <= CORE or not core_only)]
    return _rmse(vals), len(vals)

  models = []
  for m in CANDIDATES:
    full, nfull = rmse(iv_err[m])
    core, _ = rmse(iv_err[m], core_only=True)
    models.append(dict(key=m, pretty=PRETTY[m], rmse_full=full,
                       rmse_core=core, fails=fails[m], n=nfull))

  ranked = sorted([mm for mm in models if not math.isnan(mm["rmse_core"])],
                  key=lambda mm: mm["rmse_core"])
  tied = [mm["pretty"] for mm in ranked
          if ranked and mm["rmse_core"] - ranked[0]["rmse_core"] <= 5e-4]
  itm_puts = sorted([p for p in put_spread if p[0] > 1.05], reverse=True)[:6]

  return dict(source=source, n_rows=len(rows), rate=r, div_yield=q,
              ref_col=ref_col, ref_name=ref_name, core=CORE, models=models,
              ranked=ranked, tied=tied, itm_puts=itm_puts)


def run_snapshot(csv_path: str, params: dict) -> None:
  res = score_models(csv_path, params)
  if not res:
    print(f"No rows in {csv_path}")
    return

  print("=" * 74)
  print(f"SCORING MODELS AGAINST {csv_path}")
  print(f"  rows={res['n_rows']}  r={res['rate']:.3%}  q={res['div_yield']:.3%}")
  print("=" * 74)
  print(f"  reference IV column: {res['ref_col']}  [{res['ref_name']}]")
  print()

  print(f"{'model':<34} {'RMSE full':>10} {'RMSE core':>10} "
        f"{'fails':>6} {'n':>5}")
  for mm in res["models"]:
    print(f"{mm['pretty']:<34} {mm['rmse_full']:>10.5f} {mm['rmse_core']:>10.5f} "
          f"{mm['fails']:>6} {mm['n']:>5}")
  print(f"  RMSE = model IV(from mid) vs {res['ref_col']}. "
        f"core = liquid strikes |K/S-1|<={res['core']}.")

  if res["itm_puts"]:
    print()
    print("American-vs-European tell -- in-the-money PUTS, IV(BSM) - IV(CRR):")
    print(f"  {'K/S':>6} {'IV gap (bp)':>12}")
    for mny, gap in res["itm_puts"]:
      print(f"  {mny:>6.2f} {gap:>12.1f}")
    print("  large gap => European BSM must distort IV to fit an American price;")
    print("  ~0 gap (near/OTM) => BSM and CRR are indistinguishable there.")

  print()
  if len(res["tied"]) > 1:
    print("CLOSEST (liquid core, indistinguishable <5bp): "
          + "  ==  ".join(res["tied"]))
  elif res["ranked"]:
    print(f"CLOSEST on liquid core: {res['ranked'][0]['pretty']}")
  print("NOTE: any full-chain edge for CRR comes from deep-ITM puts, where")
  print("European pricing is degenerate and quotes are illiquid -- not proof")
  print("of Barchart's engine. Paste Barchart's bar_iv to test them directly.")


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _f(x):
  try:
    if x is None or x == "":
      return None
    return float(str(x).replace("%", "").replace(",", ""))
  except ValueError:
    return None


def _mid(row) -> float:
  bid, ask, last = _f(row.get("bid")), _f(row.get("ask")), _f(row.get("last"))
  if bid is not None and ask is not None and bid > 0 and ask > 0:
    return 0.5 * (bid + ask)
  return last if last is not None else 0.0


def _rmse(errs) -> float:
  if not errs:
    return float("nan")
  return math.sqrt(statistics.fmean(e * e for e in errs))


def textwrap_fill(s: str, width: int = 74) -> str:
  import textwrap
  return "\n".join(textwrap.wrap(s, width))


def main(argv=None) -> None:
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--demo", action="store_true",
                  help="quantify model distinguishability (no data needed)")
  ap.add_argument("--snapshot", metavar="CSV",
                  help="score models against a real Barchart snapshot")
  ap.add_argument("--spot", type=float)
  ap.add_argument("--rate", type=float)
  ap.add_argument("--div-yield", type=float)
  ap.add_argument("--t-days", type=int)
  args = ap.parse_args(argv)

  params = dict(DEFAULT_PARAMS)
  if args.spot is not None: params["spot"] = args.spot
  if args.rate is not None: params["rate"] = args.rate
  if args.div_yield is not None: params["div_yield"] = args.div_yield
  if args.t_days is not None: params["t_days"] = args.t_days

  if args.snapshot:
    run_snapshot(args.snapshot, params)
  else:
    run_demo(params)


if __name__ == "__main__":
  main()
