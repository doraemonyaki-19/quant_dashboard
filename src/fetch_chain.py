"""Download the SPCX option chain from Yahoo (via yfinance) into data/.

Yahoo is used because Barchart's own endpoints require session tokens/cookies.
Yahoo gives the same market quotes (bid/ask/last, volume, OI) plus Yahoo's OWN
implied vol. That lets the harness (a) reconstruct IV from market mid under each
candidate model, and (b) cross-check against a second vendor's IV.

NOTE: Yahoo's `impliedVolatility` is Yahoo's calc, NOT Barchart's. To score
against Barchart specifically, paste Barchart's displayed IV/delta into the
`bar_iv` / `bar_delta` columns of the saved CSV (they start blank here).

Usage:
    python src/fetch_chain.py                       # default exp 2026-12-18
    python src/fetch_chain.py --expiration 2026-12-18 --symbol SPCX
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
from pathlib import Path

import yfinance as yf

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def spot_price(t: yf.Ticker) -> float:
  for getter in (
      lambda: t.fast_info.get("last_price"),
      lambda: t.fast_info.get("previous_close"),
      lambda: float(t.history(period="5d")["Close"].iloc[-1]),
  ):
    try:
      v = getter()
      if v:
        return float(v)
    except Exception:
      pass
  raise RuntimeError("could not determine spot price")


def main(argv=None) -> None:
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--symbol", default="SPCX")
  ap.add_argument("--expiration", default="2026-12-18")
  ap.add_argument("--asof", default=None,
                  help="snapshot date YYYY-MM-DD for t_days (default: today)")
  args = ap.parse_args(argv)

  t = yf.Ticker(args.symbol)
  exps = t.options
  if args.expiration not in exps:
    raise SystemExit(f"{args.expiration} not in expirations: {exps}")

  S = spot_price(t)
  asof = (dt.date.fromisoformat(args.asof) if args.asof else dt.date.today())
  expd = dt.date.fromisoformat(args.expiration)
  t_days = (expd - asof).days

  chain = t.option_chain(args.expiration)
  out = DATA_DIR / f"{args.symbol.lower()}_{args.expiration}.csv"

  cols = ["type", "strike", "spot", "t_days", "bid", "ask", "last",
          "volume", "open_interest", "yahoo_iv", "bar_iv", "bar_delta",
          "bar_theta", "bar_gamma", "bar_vega"]
  n = 0
  with open(out, "w", newline="", encoding="utf-8") as fh:
    w = csv.writer(fh)
    w.writerow(cols)
    for kind, df in (("call", chain.calls), ("put", chain.puts)):
      for _, r in df.iterrows():
        w.writerow([
            kind,
            _num(r.get("strike")),
            round(S, 4),
            t_days,
            _num(r.get("bid")),
            _num(r.get("ask")),
            _num(r.get("lastPrice")),
            _num(r.get("volume")),
            _num(r.get("openInterest")),
            _num(r.get("impliedVolatility")),  # Yahoo's IV (decimal)
            "", "", "", "", "",                # Barchart cols: paste later
        ])
        n += 1

  print(f"symbol={args.symbol}  spot={S:.4f}  expiration={args.expiration}"
        f"  asof={asof}  t_days={t_days}")
  print(f"wrote {n} rows ({len(chain.calls)} calls, {len(chain.puts)} puts) -> {out}")


def chain_rows(t: yf.Ticker, expiration: str, S: float, t_days: int) -> list:
  """Return the option chain for one expiration as row dicts (no file written).

  Same schema the scorer expects (score_rows): type/strike/spot/t_days/bid/ask/
  last plus yahoo_iv as the reference IV. Used by the web app.
  """
  chain = t.option_chain(expiration)
  rows = []
  for kind, df in (("call", chain.calls), ("put", chain.puts)):
    for _, r in df.iterrows():
      rows.append(dict(
          type=kind, strike=_num(r.get("strike")), spot=round(S, 4),
          t_days=t_days, bid=_num(r.get("bid")), ask=_num(r.get("ask")),
          last=_num(r.get("lastPrice")),
          yahoo_iv=_num(r.get("impliedVolatility")), bar_iv="", bar_delta=""))
  return rows


def _num(x):
  try:
    if x is None:
      return ""
    f = float(x)
    return f
  except (TypeError, ValueError):
    return ""


if __name__ == "__main__":
  main()
