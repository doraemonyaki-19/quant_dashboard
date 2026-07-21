"""SPCX vol surface, greeks(T) and parameter term structure -- up to 12 months.

Everything here is CROSS-SECTIONAL from one snapshot: it pulls every SPCX
expiration within `--months` (default 12) from Yahoo and, per option, inverts
Black-Scholes (q=0) on the market mid to get implied vol. From that it builds:

  * vol_surface        sigma(K, T)  -- heatmap + 3D, saved to data/ + figures/
  * greeks(T)          ATM delta/gamma/theta/vega/rho vs maturity
  * param term struct  ATM IV(T), forward F(T), rate r(T) vs maturity

Outputs:
  data/vol_surface.csv
  data/term_structure_params.csv
  figures/vol_surface_heatmap.png
  figures/vol_surface_3d.png
  figures/greeks_vs_maturity.png
  figures/params_vs_maturity.png

Usage:
    python src/surfaces.py --symbol SPCX --months 12
"""

from __future__ import annotations

import argparse
import base64
import csv
import datetime as dt
import io
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
from mpl_toolkits.mplot3d import Axes3D   # noqa: F401,E402
import yfinance as yf                     # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from pricing import OptionInputs, implied_vol_bsm, bsm_greeks, _bump  # noqa: E402

Q = 0.0  # default dividend yield; per-ticker value fetched by div_yield()


# Flat rate placeholder. Replace with a live Treasury curve r(T) if desired.
def rate_for(_t_years: float) -> float:
  return 0.043


def div_yield(t: yf.Ticker, spot: float) -> float:
  """Trailing-12-month dividend yield (fraction). 0.0 if none / unavailable.

  Computed from actual paid dividends (t.dividends) rather than yfinance's
  info['dividendYield'], which is inconsistent across versions (it reports
  wildly wrong values like 32% for AAPL). Sanity-capped at 50%.
  """
  try:
    div = t.dividends
    if div is None or len(div) == 0 or not spot:
      return 0.0
    idx = div.index
    try:
      idx = idx.tz_localize(None)
    except (TypeError, AttributeError):
      pass
    cutoff = pd.Timestamp.today().normalize() - pd.Timedelta(days=365)
    annual = float(div[idx >= cutoff].sum())
    y = annual / float(spot)
    return y if 0.0 <= y < 0.5 else 0.0
  except Exception:
    return 0.0


# --------------------------------------------------------------------------- #
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


def option_mid(row):
  try:
    bid, ask = float(row.get("bid")), float(row.get("ask"))
  except (TypeError, ValueError):
    bid = ask = 0.0
  if bid > 0 and ask > 0:
    return 0.5 * (bid + ask)
  try:
    return float(row.get("lastPrice"))
  except (TypeError, ValueError):
    return 0.0


def collect(symbol: str, months: int, q: float = 0.0, ticker: yf.Ticker = None):
  """Collect the surface + per-expiry ATM data.

  Returns (S, asof, rows_surface, per_expiry, stats). `stats` makes the run's
  data coverage honest: how many expiries were within the horizon, how many
  produced a full-enough smile (>=4 OTM points) to enter the term structure,
  and which were dropped (thin smile or fetch failure). This is why
  `len(per_expiry)` can be far smaller than the ticker's true expiry count.
  """
  MIN_SMILE = 4                      # OTM points needed for a term-structure row
  t = ticker or yf.Ticker(symbol)
  S = spot_price(t)
  asof = dt.date.today()
  horizon = asof + dt.timedelta(days=int(months * 30.44))
  rows_surface = []  # flat records for CSV
  per_expiry = []
  n_in_horizon = 0
  dropped = []       # (expiry, reason)

  for exp in t.options:
    ed = dt.date.fromisoformat(exp)
    T_days = (ed - asof).days
    if T_days <= 0 or ed > horizon:
      continue
    n_in_horizon += 1
    T = T_days / 365.0
    r = rate_for(T)
    F = S * math.exp((r - q) * T)
    try:
      ch = t.option_chain(exp)
    except Exception:
      dropped.append((exp, "fetch_failed"))
      continue

    smile = []  # (moneyness, iv)
    for is_call, df in ((True, ch.calls), (False, ch.puts)):
      for _, row in df.iterrows():
        K = float(row["strike"])
        # Use OTM side only for a clean surface (tight quotes, no early-ex noise)
        if (is_call and K < S) or (not is_call and K >= S):
          continue
        mid = option_mid(row)
        intrinsic = max((S - K) if not is_call else (K - S), 0.0)
        if mid <= max(intrinsic, 0.01):
          continue
        o = OptionInputs(S, K, T, r, q, 0.5, is_call)
        iv = implied_vol_bsm(o, mid)
        if iv is None or math.isnan(iv) or not (0.02 < iv < 4.0):
          continue
        mny = K / S
        if 0.5 <= mny <= 2.0:
          smile.append((mny, iv))
          rows_surface.append(dict(expiry=exp, t_days=T_days, strike=K,
                                   moneyness=round(mny, 4),
                                   side="call" if is_call else "put",
                                   mid=round(mid, 4), iv=round(iv, 6)))
    if len(smile) < MIN_SMILE:
      dropped.append((exp, f"thin_smile({len(smile)})"))
      continue
    smile.sort()

    # ATM: strike nearest the forward; IV by interpolating the smile at K=F
    m_atm = F / S
    ms = np.array([m for m, _ in smile])
    ivs = np.array([v for _, v in smile])
    atm_iv = float(np.interp(m_atm, ms, ivs))
    Kx = min((K for K in ms * S), key=lambda k: abs(k - F))
    g = bsm_greeks(OptionInputs(S, float(Kx), T, r, q, atm_iv, True))

    per_expiry.append(dict(
        expiry=exp, T_days=T_days, T=T, r=r, F=F, atm_iv=atm_iv,
        atm_strike=float(Kx), smile=smile,
        delta=g.delta, gamma=g.gamma, theta=g.theta, vega=g.vega, rho=g.rho))

  per_expiry.sort(key=lambda d: d["T_days"])
  stats = dict(
      n_in_horizon=n_in_horizon,
      n_full_smile=len(per_expiry),
      n_dropped_thin=sum(1 for _, r in dropped if r.startswith("thin")),
      n_dropped_fetch=sum(1 for _, r in dropped if r == "fetch_failed"),
      dropped=dropped, min_smile_points=MIN_SMILE)
  return S, asof, rows_surface, per_expiry, stats


# --------------------------------------------------------------------------- #
def write_csvs(rows_surface, per_expiry, data_dir: Path):
  s = data_dir / "vol_surface.csv"
  with open(s, "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=["expiry", "t_days", "strike",
                                       "moneyness", "side", "mid", "iv"])
    w.writeheader()
    w.writerows(rows_surface)

  p = data_dir / "term_structure_params.csv"
  with open(p, "w", newline="", encoding="utf-8") as fh:
    w = csv.writer(fh)
    w.writerow(["expiry", "t_days", "T_years", "atm_strike", "atm_iv",
                "rate", "forward", "delta", "gamma", "theta_day", "vega_pt",
                "rho_pct"])
    for d in per_expiry:
      w.writerow([d["expiry"], d["T_days"], round(d["T"], 5),
                  round(d["atm_strike"], 2), round(d["atm_iv"], 6),
                  d["r"], round(d["F"], 4), round(d["delta"], 5),
                  round(d["gamma"], 6), round(d["theta"], 5),
                  round(d["vega"], 5), round(d["rho"], 5)])
  return s, p


# --------------------------------------------------------------------------- #
def _nice_step(x: float) -> float:
  """Round a raw bin width up to a human-friendly moneyness step."""
  for s in (0.0025, 0.005, 0.01, 0.02, 0.025, 0.05, 0.1):
    if x <= s:
      return s
  return 0.1


def data_moneyness_grid(per_expiry, target_bins: int = 48):
  """Derive the heatmap bin RANGE and bin SIZE from the observed moneyness.

  Instead of a hardcoded 0.6-1.6 window, span exactly the strikes that trade:
  range = [min, max] observed moneyness (snapped to a nice step), and bin size
  = nice_step(range / target_bins). Returns (grid, step, m_min, m_max).
  """
  allm = [m for d in per_expiry for m, _ in d["smile"]]
  if not allm:
    return np.linspace(0.6, 1.6, 41), 0.025, 0.6, 1.6
  m_min, m_max = min(allm), max(allm)
  step = _nice_step((m_max - m_min) / max(target_bins, 1))
  lo = math.floor(m_min / step) * step
  hi = math.ceil(m_max / step) * step
  grid = np.arange(lo, hi + step / 2, step)
  return grid, step, m_min, m_max


def build_grid(per_expiry, m_lo=0.6, m_hi=1.6, n=41, grid=None):
  """Interpolate each expiry's smile onto a common moneyness grid.

  Pass an explicit `grid` (e.g. from data_moneyness_grid) to bin over the
  observed moneyness range; otherwise a linspace(m_lo, m_hi, n) is used.
  """
  m_grid = grid if grid is not None else np.linspace(m_lo, m_hi, n)
  T_axis = np.array([d["T_days"] for d in per_expiry], float)
  Z = np.full((len(per_expiry), len(m_grid)), np.nan)
  for i, d in enumerate(per_expiry):
    ms = np.array([m for m, _ in d["smile"]])
    ivs = np.array([v for _, v in d["smile"]])
    z = np.interp(m_grid, ms, ivs, left=np.nan, right=np.nan)
    Z[i] = z
  return m_grid, T_axis, Z


def fig_surface(per_expiry, symbol="SPCX"):
  """Heatmap of sigma(K, T); bins span the observed moneyness range."""
  grid, step, m_min, m_max = data_moneyness_grid(per_expiry)
  m_grid, T_axis, Z = build_grid(per_expiry, grid=grid)
  fig, ax = plt.subplots(figsize=(9, 6))
  Zm = np.ma.masked_invalid(Z) * 100.0
  pc = ax.pcolormesh(m_grid, T_axis, Zm, shading="auto", cmap="viridis")
  ax.set_xlabel("moneyness  K / S")
  ax.set_ylabel("days to expiry  T")
  ax.set_xlim(grid[0], grid[-1])
  ax.set_title(f"{symbol} implied-vol surface  σ(K, T)   "
               f"bins {m_min:.2f}–{m_max:.2f}, Δm={step:g}")
  cb = fig.colorbar(pc, ax=ax)
  cb.set_label("implied vol  (%)")
  ax.axvline(1.0, color="w", lw=0.8, ls="--", alpha=0.7)
  fig.tight_layout()
  return fig


def fig_surface3d(per_expiry, symbol="SPCX"):
  """3D surface over a tighter moneyness band (0.85-1.20); interior gaps filled."""
  m_grid, T_axis, Z = build_grid(per_expiry, 0.85, 1.20, 31)
  Zf = Z.copy()
  for i in range(Zf.shape[0]):
    row = Zf[i]
    if np.isnan(row).all():
      continue
    idx = np.arange(len(row))
    good = ~np.isnan(row)
    row[~good] = np.interp(idx[~good], idx[good], row[good])
    Zf[i] = row
  M, Tg = np.meshgrid(m_grid, T_axis)
  fig = plt.figure(figsize=(9, 6.5))
  ax = fig.add_subplot(111, projection="3d")
  ax.plot_surface(M, Tg, Zf * 100.0, cmap="viridis", edgecolor="none",
                  antialiased=True)
  ax.set_xlabel("moneyness K/S")
  ax.set_ylabel("days to expiry T")
  ax.set_zlabel("IV (%)")
  ax.set_title(f"{symbol} implied-vol surface (3D)")
  ax.view_init(elev=28, azim=-125)
  fig.tight_layout()
  return fig


def fig_greeks(per_expiry, symbol="SPCX"):
  T = [d["T_days"] for d in per_expiry]
  specs = [("delta", "ATM delta"), ("gamma", "ATM gamma (per $)"),
           ("vega", "ATM vega (per 1% vol)"), ("theta", "ATM theta (per day)"),
           ("rho", "ATM rho (per 1% rate)")]
  fig, axes = plt.subplots(2, 3, figsize=(13, 7.5))
  axes = axes.ravel()
  for ax, (key, title) in zip(axes, specs):
    ax.plot(T, [d[key] for d in per_expiry], "o-", color="#2b6cb0")
    ax.set_title(title)
    ax.set_xlabel("days to expiry T")
    ax.grid(alpha=0.3)
  axes[5].plot(T, [d["atm_iv"] * 100 for d in per_expiry], "o-", color="#c05621")
  axes[5].set_title("ATM implied vol (%)")
  axes[5].set_xlabel("days to expiry T")
  axes[5].grid(alpha=0.3)
  fig.suptitle(f"{symbol} ATM greeks as a function of maturity (q=0, up to 12m)",
               fontsize=13)
  fig.tight_layout(rect=(0, 0, 1, 0.97))
  return fig


def fig_params(per_expiry, symbol="SPCX"):
  T = [d["T_days"] for d in per_expiry]
  fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
  axes[0].plot(T, [d["atm_iv"] * 100 for d in per_expiry], "o-", color="#c05621")
  axes[0].set_title("σ_ATM(T)  — implied-vol term structure")
  axes[0].set_ylabel("IV (%)")
  axes[1].plot(T, [d["r"] * 100 for d in per_expiry], "o-", color="#2f855a")
  axes[1].set_title("r(T)  — risk-free (placeholder, edit rate_for)")
  axes[1].set_ylabel("rate (%)")
  axes[2].plot(T, [d["F"] for d in per_expiry], "o-", color="#553c9a")
  axes[2].set_title("F(T) = S·e^{(r−q)T}  — forward")
  axes[2].set_ylabel("forward price ($)")
  for ax in axes:
    ax.set_xlabel("days to expiry T")
    ax.grid(alpha=0.3)
  fig.suptitle(f"{symbol} pricing parameters as a function of maturity (up to 12m)",
               fontsize=13)
  fig.tight_layout(rect=(0, 0, 1, 0.94))
  return fig


def _save(fig, path: Path):
  fig.savefig(path, dpi=130)
  plt.close(fig)


def _b64(fig) -> str:
  buf = io.BytesIO()
  fig.savefig(buf, format="png", dpi=130)
  plt.close(fig)
  return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def figures_b64(per_expiry, symbol="SPCX") -> dict:
  """Render all four figures to base64 data URIs (in-memory, for the web app)."""
  return {
      "vol_surface_heatmap": _b64(fig_surface(per_expiry, symbol)),
      "vol_surface_3d": _b64(fig_surface3d(per_expiry, symbol)),
      "greeks_vs_maturity": _b64(fig_greeks(per_expiry, symbol)),
      "params_vs_maturity": _b64(fig_params(per_expiry, symbol)),
  }


_PLOTLYJS = None  # cached inline plotly.js bundle (shared across renders)


def plotly_js() -> str:
  """The full plotly.js source, bundled from the local package (no CDN)."""
  global _PLOTLYJS
  if _PLOTLYJS is None:
    from plotly.offline import get_plotlyjs
    _PLOTLYJS = get_plotlyjs()
  return _PLOTLYJS


def plotly_surface_html(per_expiry, symbol="SPCX", embed_js: bool = False) -> str:
  """Interactive, drag-to-rotate 3D IV surface as an HTML fragment.

  Plotly is bundled locally (no CDN). embed_js=True inlines the full library
  into this fragment (self-contained, for the static file / Artifact);
  embed_js=False emits only the plot + a reference to plotly.js that the page
  must provide once (the web app serves it at /plotly.js and caches it).
  """
  import plotly.graph_objects as go
  grid, step, m_min, m_max = data_moneyness_grid(per_expiry, target_bins=32)
  _, T_axis, Z = build_grid(per_expiry, grid=grid)
  Zf = Z.copy()
  for i in range(Zf.shape[0]):                # fill gaps so the sheet is smooth
    row = Zf[i]
    if np.isnan(row).all():
      continue
    idx = np.arange(len(row))
    good = ~np.isnan(row)
    row[~good] = np.interp(idx[~good], idx[good], row[good])
    Zf[i] = row
  fig = go.Figure(data=[go.Surface(
      x=grid, y=T_axis, z=Zf * 100.0, colorscale="Viridis",
      colorbar=dict(title="IV %"),
      hovertemplate="K/S %{x:.2f}<br>%{y:.0f}d<br>IV %{z:.1f}%<extra></extra>")])
  fig.update_layout(
      scene=dict(xaxis_title="moneyness K/S", yaxis_title="days to expiry T",
                 zaxis_title="IV (%)",
                 camera=dict(eye=dict(x=-1.6, y=-1.6, z=0.7))),
      template="plotly_white", paper_bgcolor="white",
      font=dict(size=11), margin=dict(l=0, r=0, t=4, b=0), height=460,
      title=dict(text=f"{symbol} IV surface", x=0.5, y=0.98, font=dict(size=13)))
  return fig.to_html(full_html=False,
                     include_plotlyjs=(True if embed_js else False),
                     default_height="460px",
                     config={"displayModeBar": True, "responsive": True,
                             "displaylogo": False,
                             "modeBarButtonsToRemove": ["toImage"]})


# --------------------------------------------------------------------------- #
def main(argv=None):
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--symbol", default="SPCX")
  ap.add_argument("--months", type=int, default=12)
  args = ap.parse_args(argv)

  data_dir = ROOT / "data"
  fig_dir = ROOT / "figures"
  fig_dir.mkdir(exist_ok=True)

  t = yf.Ticker(args.symbol)
  S0 = spot_price(t)
  q = div_yield(t, S0)
  S, asof, rows_surface, per_expiry, stats = collect(args.symbol, args.months, q, t)
  if not per_expiry:
    raise SystemExit("no expirations collected")
  print(f"{args.symbol}  spot={S:.4f}  asof={asof}  q={q:.4f}  "
        f"expiries in horizon: {stats['n_in_horizon']}  "
        f"full-smile: {stats['n_full_smile']}  "
        f"(dropped thin={stats['n_dropped_thin']}, fetch={stats['n_dropped_fetch']})  "
        f"surface points: {len(rows_surface)}")
  if stats["n_full_smile"] < 0.6 * stats["n_in_horizon"]:
    print(f"  WARNING: only {stats['n_full_smile']}/{stats['n_in_horizon']} "
          "expiries had a full smile -- term structure may be sparse "
          "(thin/stale quotes).")

  s_csv, p_csv = write_csvs(rows_surface, per_expiry, data_dir)
  _save(fig_surface(per_expiry, args.symbol), fig_dir / "vol_surface_heatmap.png")
  _save(fig_surface3d(per_expiry, args.symbol), fig_dir / "vol_surface_3d.png")
  _save(fig_greeks(per_expiry, args.symbol), fig_dir / "greeks_vs_maturity.png")
  _save(fig_params(per_expiry, args.symbol), fig_dir / "params_vs_maturity.png")

  print("wrote:")
  for f in (s_csv, p_csv,
            fig_dir / "vol_surface_heatmap.png",
            fig_dir / "vol_surface_3d.png",
            fig_dir / "greeks_vs_maturity.png",
            fig_dir / "params_vs_maturity.png"):
    print("  ", f.relative_to(ROOT))

  print("\nATM term structure (T days | IV | delta | gamma | vega | theta/day | rho):")
  for d in per_expiry:
    print(f"  {d['T_days']:>4}  iv={d['atm_iv']*100:5.1f}%  delta={d['delta']:.3f}  "
          f"gamma={d['gamma']:.4f}  vega={d['vega']:.3f}  theta={d['theta']:.3f}  "
          f"rho={d['rho']:.3f}")


if __name__ == "__main__":
  main()
