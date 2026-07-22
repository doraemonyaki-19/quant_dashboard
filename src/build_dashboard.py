"""Render the options model dashboard for ANY ticker (shared by CLI + web app).

  compute_context(symbol)  -> pulls the chain, builds figures + scoring live
  render_dashboard(context) -> fills the HTML template
  main()                    -> writes a static dashboard.html for one --symbol

The web app (src/app.py) imports compute_context + render_dashboard so the page
and the CLI produce identical output. Data comes from Yahoo (yfinance); a
published Artifact cannot fetch, so live ticker entry needs the local app.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import yfinance as yf                                          # noqa: E402
from determine_model import score_rows                        # noqa: E402
from surfaces import (collect, div_yield, rate_for, spot_price,  # noqa: E402
                      fig_surface, fig_greeks, fig_params, _b64,
                      plotly_surface_html)
from strategies import (analyze, fig_payoff, write_strategy_csvs,  # noqa: E402
                        strategy_leg_row)
from fetch_chain import chain_rows                             # noqa: E402


GREEKS = [
    ("&Delta;", "Delta", "&part;V / &part;S",
     "Price move per $1 change in the underlying.", "per $1, range -1..1"),
    ("&Gamma;", "Gamma", "&part;&sup2;V / &part;S&sup2;",
     "How fast delta itself moves; largest near-the-money and short-dated.",
     "delta per $1"),
    ("&Theta;", "Theta", "&part;V / &part;t",
     "Time decay &mdash; value lost as one calendar day passes (negative for a long option).",
     "price per day"),
    ("&nu;", "Vega", "&part;V / &part;&sigma;",
     "Price move per 1 percentage-point change in implied volatility.",
     "per 1 vol point"),
    ("&rho;", "Rho", "&part;V / &part;r",
     "Price move per 1 percentage-point change in the risk-free rate.",
     "per 1% rate"),
]


# --------------------------------------------------------------------------- #
# Pipeline: symbol -> everything the template needs (all live from Yahoo)
# --------------------------------------------------------------------------- #
def compute_context(symbol: str, months: int = 12,
                    embed_plotly: bool = False) -> dict:
  symbol = symbol.upper().strip()
  if not symbol or len(symbol) > 8 or not symbol.replace(".", "").isalnum():
    raise ValueError(f"'{symbol}' is not a valid ticker symbol.")
  t = yf.Ticker(symbol)
  try:
    q = div_yield(t, spot_price(t))
    S, asof, rows_surface, per_expiry, cov = collect(symbol, months, q, t)
  except Exception as e:
    raise ValueError(f"Could not load option data for '{symbol}' ({e}).")
  if not per_expiry:
    raise ValueError(
        f"No listed options within {months} months for '{symbol}'. "
        "Try a symbol with an active options market.")

  figs = {
      "vol_surface_heatmap": _b64(fig_surface(per_expiry, symbol)),
      "greeks_vs_maturity": _b64(fig_greeks(per_expiry, symbol)),
      "params_vs_maturity": _b64(fig_params(per_expiry, symbol)),
  }
  surf3d_html = plotly_surface_html(per_expiry, symbol, embed_js=embed_plotly)
  feat = min(per_expiry, key=lambda d: abs(d["T_days"] - 150))  # ~5-month expiry
  feat_rate = rate_for(feat["T"])
  rows = chain_rows(t, feat["expiry"], S, feat["T_days"])
  score = score_rows(rows, {"rate": feat_rate, "div_yield": q},
                     source=feat["expiry"])

  # Term-structure trade ideas + payoff curves (guarded: never break the page)
  strat = None
  try:
    strat = analyze(symbol, ticker=t, spot=S, q=q)
    for s in strat["strategies"]:
      s["payoff_png"] = _b64(fig_payoff(s, S))
      s["payoff_png_put"] = _b64(fig_payoff(s, S, variant="put"))
  except Exception as e:
    sys.stderr.write(f"[strategies] warning: {e}\n")

  ivs = [d["atm_iv"] for d in per_expiry]
  ctx = dict(symbol=symbol, spot=S, asof=asof, q=q, months=months, strat=strat,
             coverage=cov,
             n_exp=len(per_expiry), per_expiry=per_expiry, figs=figs,
             surf3d_html=surf3d_html, plotly_external=not embed_plotly,
             feat=feat, feat_rate=feat_rate, score=score,
             rows_surface=rows_surface, scoring_rows=rows,
             iv_hi=max(ivs) * 100, iv_lo=min(ivs) * 100,
             feat_iv=feat["atm_iv"] * 100)
  try:
    ctx["audit_dir"] = save_audit(ctx)  # persist inputs + results for audit
  except Exception as e:  # never let an audit-write error break the analysis
    sys.stderr.write(f"[audit] warning: could not save audit bundle: {e}\n")
    ctx["audit_dir"] = None
  return ctx


# --------------------------------------------------------------------------- #
# Audit trail: persist the raw inputs + calculation results under data/audit/
# --------------------------------------------------------------------------- #
def _nan_to_none(x):
  return None if isinstance(x, float) and math.isnan(x) else x


def save_audit(ctx: dict) -> Path:
  """Write the data used and the results to data/audit/<SYMBOL>_<stamp>/.

  Bundle (all plain text, re-openable independently):
    manifest.json       run metadata + verdict summary (machine-readable)
    surface_points.csv  every OTM strike used: mid (input) -> IV (output)
    term_structure.csv  per-expiry ATM IV, greeks, rate, forward
    scoring_chain.csv    raw featured-expiry chain fed to the model scoring
    scoring_results.csv  per-model RMSE table
  Also appends one row to data/audit/index.csv (the audit log).
  """
  now = dt.datetime.now()
  base = ROOT / "data" / "audit" / f"{ctx['symbol']}_{now:%Y%m%d-%H%M%S}"
  base.mkdir(parents=True, exist_ok=True)
  score = ctx["score"]
  best = score["ranked"][0]["pretty"] if score["ranked"] else None

  manifest = {
      "symbol": ctx["symbol"],
      "generated_local": now.isoformat(timespec="seconds"),
      "data_source": "Yahoo Finance option chain (yfinance)",
      "asof": str(ctx["asof"]),
      "spot": round(ctx["spot"], 4),
      "dividend_yield_q": ctx["q"],
      "rate_used": ctx["feat_rate"],
      "rate_note": "flat placeholder (rate_for); swap for a live Treasury curve",
      "months": ctx["months"],
      "expiry_coverage": {
          "in_horizon": ctx["coverage"]["n_in_horizon"],
          "with_full_smile": ctx["coverage"]["n_full_smile"],
          "dropped_thin_smile": ctx["coverage"]["n_dropped_thin"],
          "dropped_fetch_failed": ctx["coverage"]["n_dropped_fetch"],
          "min_smile_points": ctx["coverage"]["min_smile_points"],
          "dropped": [{"expiry": e, "reason": r}
                      for e, r in ctx["coverage"]["dropped"]],
          "note": ("n_expiries counts only expiries with a full-enough smile "
                   "(>= min_smile_points OTM strikes); the term structure uses "
                   "these. Others were dropped (thin/stale quotes or fetch "
                   "failure). The ticker may list many more expiries."),
      },
      "n_expiries": ctx["n_exp"],
      "expiries": [d["expiry"] for d in ctx["per_expiry"]],
      "featured_expiry": ctx["feat"]["expiry"],
      "featured_t_days": ctx["feat"]["T_days"],
      "reference_iv_column": score["ref_col"],
      "reference_iv_source": score["ref_name"],
      "core_moneyness_band": score["core"],
      "scoring": {
          "models": [{"model": m["pretty"],
                      "rmse_core": _nan_to_none(round(m["rmse_core"], 6)),
                      "rmse_full": _nan_to_none(round(m["rmse_full"], 6)),
                      "unpriced": m["fails"], "n": m["n"]}
                     for m in score["models"]],
          "lowest_rmse_core_model": best,
          "itm_put_iv_gap_bp": [[round(mny, 3), round(gap, 1)]
                                for mny, gap in score["itm_puts"]],
      },
      "files": ["surface_points.csv", "term_structure.csv",
                "scoring_chain.csv", "scoring_results.csv"],
  }
  (base / "manifest.json").write_text(json.dumps(manifest, indent=2),
                                      encoding="utf-8")

  with open(base / "surface_points.csv", "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, extrasaction="ignore",
                       fieldnames=["expiry", "t_days", "strike", "moneyness",
                                   "side", "mid", "iv"])
    w.writeheader()
    w.writerows(ctx["rows_surface"])

  with open(base / "term_structure.csv", "w", newline="", encoding="utf-8") as fh:
    w = csv.writer(fh)
    w.writerow(["expiry", "t_days", "T_years", "atm_strike", "atm_iv", "rate",
                "forward", "delta", "gamma", "theta_day", "vega_pt", "rho_pct"])
    for d in ctx["per_expiry"]:
      w.writerow([d["expiry"], d["T_days"], round(d["T"], 5),
                  round(d["atm_strike"], 2), round(d["atm_iv"], 6), d["r"],
                  round(d["F"], 4), round(d["delta"], 5), round(d["gamma"], 6),
                  round(d["theta"], 5), round(d["vega"], 5), round(d["rho"], 5)])

  with open(base / "scoring_chain.csv", "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, extrasaction="ignore",
                       fieldnames=["type", "strike", "spot", "t_days", "bid",
                                   "ask", "last", "yahoo_iv", "bar_iv",
                                   "bar_delta"])
    w.writeheader()
    w.writerows(ctx["scoring_rows"])

  with open(base / "scoring_results.csv", "w", newline="", encoding="utf-8") as fh:
    w = csv.writer(fh)
    w.writerow(["model", "rmse_core", "rmse_full", "unpriced", "n"])
    for m in score["models"]:
      w.writerow([m["pretty"], _nan_to_none(round(m["rmse_core"], 6)),
                  _nan_to_none(round(m["rmse_full"], 6)), m["fails"], m["n"]])

  # strategy legs + summary (same bundle, if strategies were computed)
  if ctx.get("strat"):
    leg_rows, strat_rows = [], []
    for s in ctx["strat"]["strategies"]:
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

  idx = ROOT / "data" / "audit" / "index.csv"
  fresh = not idx.exists()
  with open(idx, "a", newline="", encoding="utf-8") as fh:
    w = csv.writer(fh)
    if fresh:
      w.writerow(["run_local", "symbol", "spot", "q", "months", "n_expiries",
                  "featured_expiry", "ref_iv", "lowest_rmse_model", "dir"])
    w.writerow([now.isoformat(timespec="seconds"), ctx["symbol"],
                round(ctx["spot"], 4), ctx["q"], ctx["months"], ctx["n_exp"],
                ctx["feat"]["expiry"], score["ref_col"], best or "",
                base.relative_to(ROOT).as_posix()])
  return base


# --------------------------------------------------------------------------- #
# HTML fragments
# --------------------------------------------------------------------------- #
def _term_rows(per_expiry) -> str:
  return "".join(
      f"<tr><td>{d['expiry']}</td><td>{d['T_days']}</td>"
      f"<td>{d['atm_iv']*100:.1f}</td><td>{d['delta']:.3f}</td>"
      f"<td>{d['gamma']:.4f}</td><td>{d['vega']:.3f}</td>"
      f"<td>{d['theta']:.3f}</td><td>{d['rho']:.3f}</td></tr>"
      for d in per_expiry)


def _score_fragments(score) -> dict:
  core_vals = [m["rmse_core"] for m in score["models"]
               if not math.isnan(m["rmse_core"])]
  spread_vp = (max(core_vals) - min(core_vals)) * 100 if core_vals else 0.0
  best_key = score["ranked"][0]["key"] if score["ranked"] else None
  rows_html = ""
  for m in score["models"]:
    low = m["key"] == best_key
    tag = '<span class="tick">lowest RMSE</span>' if low else ""
    rows_html += (
        f'<tr class="{"win" if low else ""}"><td class="ml">{m["pretty"]}{tag}</td>'
        f'<td>{m["rmse_core"]:.4f}</td><td>{m["rmse_full"]:.4f}</td>'
        f'<td>{m["fails"]}</td><td>{m["n"]}</td></tr>')
  if score["itm_puts"]:
    mny, gap = score["itm_puts"][0]
    itm_tell = f"+{gap:,.0f}&nbsp;bp at K/S&nbsp;{mny:.1f}"
  else:
    itm_tell = "n/a (no deep-ITM puts quoted in this expiry)"
  verdict_line = (f"Statistically indistinguishable on the core &mdash; all "
                  f"within ~{spread_vp:.1f} vol-points, dominated by the "
                  f"last-vs-mid reference noise")
  return dict(score_rows=rows_html, itm_tell=itm_tell, verdict_line=verdict_line,
              ref_name=score["ref_name"], core_pct=f"{score['core']*100:.0f}")


def _pill_note(q: float) -> str:
  if q < 1e-4:
    return ("q&nbsp;=&nbsp;0, so an American call equals the European call (no "
            "early exercise). Indistinguishable from the American binomial on "
            "liquid quotes &mdash; they separate only in illiquid deep-ITM puts.")
  return (f"q&nbsp;=&nbsp;{q*100:.2f}%. Indistinguishable from the American "
          "binomial on liquid quotes; American exercise bites mainly for ITM "
          "puts and calls just before an ex-dividend date.")


def _shell(searchbar: str, body: str) -> str:
  return (STYLE + '\n<div class="wrap"><div class="inner">\n'
          + searchbar + "\n" + body + "\n</div></div>\n" + SPINNER + TABS_JS)


def render_dashboard(ctx: dict) -> str:
  sym = ctx["symbol"]
  sf = _score_fragments(ctx["score"])
  gdefs = "".join(
      f'<div class="gdef"><div class="gsym">{s}</div>'
      f'<div class="gname">{n} <span class="gder">{d}</span></div>'
      f'<div class="gtxt">{txt}</div><div class="gunit eyebrow">{u}</div></div>'
      for s, n, d, txt, u in GREEKS)
  q_disp = "0.00" if ctx["q"] < 1e-4 else f"{ctx['q']*100:.2f}%"
  audit_rel = (Path(ctx["audit_dir"]).relative_to(ROOT).as_posix()
               if ctx.get("audit_dir") else "")
  cov = ctx.get("coverage", {})
  coverage_banner = ""
  if cov and cov["n_full_smile"] < 0.6 * max(cov["n_in_horizon"], 1):
    coverage_banner = (
        '<div class="flag" style="margin:6px 0 0">'
        f'<b>Sparse data.</b> Only {cov["n_full_smile"]} of '
        f'{cov["n_in_horizon"]} expiries in the 12-month window had a '
        f'full-enough smile (&ge;{cov["min_smile_points"]} out-of-the-money '
        'strikes); the rest were dropped for thin/stale quotes. The surface and '
        'term structure below cover only those expiries &mdash; treat the shape '
        'as partial, and prefer re-running during market hours.</div>')
  body = BODY.format(audit_rel=audit_rel, coverage_banner=coverage_banner,
      symbol=sym, spot=f"{ctx['spot']:,.2f}", asof=ctx["asof"],
      q_disp=q_disp, pill_note=_pill_note(ctx["q"]),
      n_exp=ctx["n_exp"], months=ctx["months"],
      feat_date=ctx["feat"]["expiry"], feat_days=ctx["feat"]["T_days"],
      feat_iv=f"{ctx['feat_iv']:.1f}",
      iv_hi=f"{ctx['iv_hi']:.0f}", iv_lo=f"{ctx['iv_lo']:.0f}",
      heatmap=ctx["figs"]["vol_surface_heatmap"],
      greeks=ctx["figs"]["greeks_vs_maturity"],
      params=ctx["figs"]["params_vs_maturity"],
      trows=_term_rows(ctx["per_expiry"]), gdefs=gdefs, **sf)
  # Plotly HTML holds literal { } (JS/JSON) -> inject after .format via sentinel.
  body = body.replace("%%SURF3D%%", ctx["surf3d_html"])

  n_strat = len(ctx["strat"]["strategies"]) if ctx.get("strat") else 0
  tabbar = (
      '<div class="tabbar" role="tablist">'
      '<button class="tabbtn" role="tab" aria-selected="true" '
      'data-tab="tab-analysis">Analysis</button>'
      '<button class="tabbtn" role="tab" aria-selected="false" '
      f'data-tab="tab-strategies">Strategies<span class="badge">{n_strat}</span>'
      '</button></div>')
  tabbed = (tabbar
            + '<div class="tabpanel active" id="tab-analysis" role="tabpanel">'
            + body + '</div>'
            + '<div class="tabpanel" id="tab-strategies" role="tabpanel">'
            + _strategy_panel(ctx) + '</div>')

  # App mode: load the locally-bundled plotly.js once (served at /plotly.js,
  # browser-cached) before the plot's inline init script runs.
  if ctx.get("plotly_external"):
    tabbed = '<script src="/plotly.js"></script>\n' + tabbed
  return _shell(searchbar_html(sym), tabbed)


def _hedge_howto(strat: dict) -> str:
  """A concrete how-to for delta-hedging, worked with the first trade's numbers."""
  S = strat["spot"]
  ex = None
  for s in strat["strategies"]:
    if abs(s["hedge_shares"]) > 1e-9:
      ex = s
      break
  if ex is None:
    return ""
  hs = ex["hedge_shares"]
  # Per one listed contract (x100 shares); a desk trades N contracts.
  sh_ctr = hs * 100
  side = "buy" if hs > 0 else "sell (short)"
  opp = "sold" if hs > 0 else "bought"
  worked = (
      f'<b>Worked example &mdash; &ldquo;{ex["name"].split("—")[0].strip()}&rdquo;.</b> '
      f'Its net delta is {ex["net"]["delta"]:+.3f}/share, so hedge&nbsp;=&nbsp;'
      f'&minus;({ex["net"]["delta"]:+.3f}) = {hs:+.3f} shares per option unit. '
      f'For <b>one contract</b> (&times;100) you {side} '
      f'<b>{abs(sh_ctr):.0f} shares</b> of {strat["symbol"]} at the ${S:,.2f} spot; '
      f'for 10 contracts, {abs(sh_ctr)*10:.0f} shares. Because the options were net '
      f'{"long" if hs < 0 else "short"} delta, the hedge is the {opp} stock that '
      f'offsets it &mdash; leaving a position whose P&amp;L comes from vol and time, '
      f'not from which way {strat["symbol"]} drifts.')
  return (
      '<div class="strat-howto">'
      '<h3>How to delta-hedge these trades</h3>'
      '<p>Each structure below carries a small residual <b>net delta</b> (its '
      'directional exposure &mdash; dollars of P&amp;L per $1 move in the '
      'underlying). Delta-hedging cancels that so the trade is a clean bet on '
      '<b>vol / term structure</b>, which is the actual thesis. The dashed line on '
      'every payoff plot is this hedged P&amp;L.</p>'
      '<ol>'
      '<li><b>Read the net delta</b> from the card&rsquo;s &Delta; chip (per '
      'share, for one option unit).</li>'
      '<li><b>Trade the opposite in stock.</b> Hold <b>&minus;net&Delta; shares '
      'per option unit</b> &mdash; i.e. <b>&minus;net&Delta;&times;100 shares per '
      'contract</b>. Positive net delta &rArr; short that many shares; negative '
      '&rArr; buy them. This zeroes total delta at the current spot.</li>'
      '<li><b>Hold, then re-hedge as spot moves.</b> The hedge is exact only at '
      'today&rsquo;s price; <b>gamma</b> makes delta drift as the underlying moves '
      '(the dashed curve still bends away from flat). A desk re-hedges back to '
      'neutral periodically &mdash; the profit/loss of that rehedging <em>is</em> '
      'the long/short-gamma P&amp;L. The dashed curve here assumes a <b>static</b> '
      'hedge set once and held to the horizon.</li>'
      '<li><b>Watch the costs.</b> Each re-hedge crosses the stock bid/ask and '
      'pays financing on the shares (and borrow, if short). Frequent rehedging of '
      'a low-gamma position can eat the edge.</li>'
      '</ol>'
      f'<p>{worked}</p>'
      '</div>')


def _strategy_panel(ctx: dict) -> str:
  """Strategies tab: term-structure trade cards with payoff curves."""
  strat = ctx.get("strat")
  if not strat or not strat["strategies"]:
    return ('<div class="errbox" style="border-left-color:var(--warn)">'
            '<h2>No strategy set available</h2><p>The chain for this ticker '
            'did not yield the expiries needed to build the term-structure '
            'trades.</p></div>')
  q_disp = "0" if strat["q"] < 1e-4 else f"{strat['q']*100:.2f}%"
  term_seq = " → ".join(f"{d['t_days']}d {d['iv']*100:.0f}%"
                        for d in strat.get("term", []))
  shape = strat.get("shape", "the term structure")
  out = [
      f'<p class="strat-intro"><b>Term-structure trades for {strat["symbol"]}</b> '
      f'(spot ${strat["spot"]:,.2f}, q={q_disp}, ATM~{strat["strikeATM"]}). '
      f'This ticker&rsquo;s ATM term structure shows <b>{shape}</b> '
      f'(<span class="mono">{term_seq}</span>). The trades below sell the '
      'genuinely richest tenor for this shape against cheaper long-dated vol. '
      'Priced from live mids; payoff is evaluated at the <b>earliest leg '
      'expiry</b>, longer legs revalued by BSM with IV held constant. Each payoff '
      'plot also shows a <b>delta-hedged</b> curve (dashed): a static stock '
      'position of &minus;net&Delta; shares/unit set at inception and held to '
      'horizon, which zeroes directional exposure at spot and isolates the '
      'vol/gamma edge. Per share (contract &times;100).</p>']

  out.append(_hedge_howto(strat))

  out.append(
      '<div class="strat-filter" role="group" aria-label="account permission filter">'
      '<span class="sf-label">My account can trade:</span>'
      '<button type="button" class="sf-btn active" data-strat-filter="naked">'
      'Naked shorts &amp; spreads</button>'
      '<button type="button" class="sf-btn" data-strat-filter="spread">'
      'Spreads only (no naked)</button>'
      '<button type="button" class="sf-btn" data-strat-filter="long">'
      'Long options only</button>'
      '<span class="sf-hint" id="strat-count"></span></div>'
      '<div id="strat-empty" class="flag" style="display:none">No trade here fits '
      'that permission level. These are all short-vol / term-structure harvests, '
      'so every one sells a leg &mdash; a strictly long-only account can&rsquo;t '
      'run them as constructed.</div>')

  req_badge = {
      "spread": ('req-spread', 'Spread', 'Defined-risk: every short leg is covered '
                 'by a long option. Tradeable with spread approval, no naked short.'),
      "naked": ('req-naked', 'Naked short', 'Contains an uncovered short leg &mdash; '
                'needs uncovered/naked-option approval.'),
      "long": ('req-long', 'Long only', 'No short legs.')}

  def leg_rows_html(legs):
    return "".join(
        f'<tr><td>{"+" if L["qty"]>0 else ""}{L["qty"]}{L["kind"]}</td>'
        f'<td>{L["expiration"]}</td><td>{L["t_days"]}</td>'
        f'<td>{L["strike"]:.0f}</td><td>{L["mid"]:.2f}</td>'
        f'<td>{L["iv"]*100:.1f}</td><td>{L["delta"]:.3f}</td>'
        f'<td>{L["theta"]:.3f}</td><td>{L["vega"]:.3f}</td></tr>'
        for L in legs)

  head = ('<thead><tr><th>Leg</th><th>Expiry</th><th>d</th><th>K</th><th>Mid</th>'
          '<th>IV%</th><th>&Delta;</th><th>&Theta;/d</th><th>Vega</th></tr></thead>')

  for s in strat["strategies"]:
    legs = leg_rows_html(s["legs"])
    legs_put = leg_rows_html(s["legs_put"])
    gp, hsp = s["net_put"], s["hedge_shares_put"]
    g = s["net"]
    prem_chip = (f'<span class="chip {s["ptype"]}">'
                 f'{s["ptype"]} ${abs(s["premium"]):.2f}/sh'
                 f' (${abs(s["premium"])*100:.0f})</span>')
    hs = s["hedge_shares"]
    hedge_chip = (f'<span class="chip">hedge {"+" if hs > 0 else ""}{hs:.2f} sh'
                  f' ({abs(hs)*100:.0f}/contract)</span>')
    chips = (prem_chip
             + f'<span class="chip">Δ {g["delta"]:+.3f}</span>'
             + f'<span class="chip">Γ {g["gamma"]:+.4f}</span>'
             + f'<span class="chip">Θ {g["theta"]:+.3f}/d</span>'
             + f'<span class="chip">vega {g["vega"]:+.3f}</span>'
             + hedge_chip)
    be = (", ".join(f"${b:g}" for b in s["breakevens"])
          if s["breakevens"] else "none in range")
    be_h = (", ".join(f"${b:g}" for b in s["breakevens_hedged"])
            if s["breakevens_hedged"] else "none in range")
    be_p = (", ".join(f"${b:g}" for b in s["breakevens_put"])
            if s["breakevens_put"] else "none in range")
    be_ph = (", ".join(f"${b:g}" for b in s["breakevens_hedged_put"])
             if s["breakevens_hedged_put"] else "none in range")
    rq_cls, rq_lbl, rq_tip = req_badge.get(s["requires"], req_badge["naked"])
    out.append(
        f'<div class="strat" data-requires="{s["requires"]}">'
        f'<div class="strat-hd"><h3>{s["name"]}'
        f'<span class="req {rq_cls}" title="{rq_tip}">{rq_lbl}</span></h3>'
        f'<div class="th">{s["thesis"]}</div></div>'
        '<div class="strat-body">'
        '<div class="strat-legs"><div class="tablewrap"><table>'
        f'{head}<tbody>{legs}</tbody></table></div>'
        f'<div class="chips">{chips}</div>'
        f'<div class="strat-be">Breakevens @ {s["horizon_exp"]}: {be} '
        f'&middot; max +${s["max_profit"]:.2f} / min ${s["max_loss"]:.2f} per sh</div>'
        f'<div class="strat-be">Delta-hedged ({"+" if hs > 0 else ""}{hs:.2f} sh/unit, '
        f'static): BE {be_h} &middot; max +${s["max_profit_hedged"]:.2f} / '
        f'min ${s["max_loss_hedged"]:.2f} per sh</div>'
        '<details class="strat-alt"><summary>Put-equivalent structure (same '
        'strikes &mdash; trade whichever fills better)</summary>'
        '<div class="tablewrap"><table>'
        f'{head}<tbody>{legs_put}</tbody></table></div>'
        f'<div class="strat-be">NET {s["ptype_put"]} '
        f'${abs(s["premium_put"]):.2f}/sh &middot; &Delta; {gp["delta"]:+.3f} '
        f'&middot; vega {gp["vega"]:+.3f} &middot; hedge '
        f'{"+" if hsp > 0 else ""}{hsp:.2f} sh/unit</div>'
        f'<div class="strat-be">Breakevens @ {s["horizon_exp"]}: {be_p} '
        f'&middot; max +${s["max_profit_put"]:.2f} / '
        f'min ${s["max_loss_put"]:.2f} per sh &middot; hedged BE {be_ph}</div>'
        f'<div class="strat-plot"><img src="{s["payoff_png_put"]}" '
        f'alt="put-equivalent payoff curve for {s["name"]}"></div>'
        f'<div class="strat-be">Put version requires: '
        f'<span class="req {req_badge.get(s["requires_put"], req_badge["naked"])[0]}">'
        f'{req_badge.get(s["requires_put"], req_badge["naked"])[1]}</span>'
        f'{" &mdash; same as the call version" if s["requires_put"] == s["requires"] else " &mdash; differs from the call version (strike coverage flips for puts)"}'
        '. Put-call parity keeps the vol trade &amp; payoff shape ~identical; '
        'premium, delta &amp; assignment differ.</div>'
        '</details>'
        '</div>'
        f'<div class="strat-plot"><img src="{s["payoff_png"]}" '
        f'alt="payoff curve for {s["name"]}"></div>'
        '</div></div>')

  out.append(
      '<div class="flag" style="margin-top:6px">'
      '<b>Read before trading.</b> Illustrative, <b>not investment advice</b>. '
      'Priced at mid &mdash; SPCX spreads are wide, so much of the apparent edge '
      'is inside the bid/ask; check your real fill. Payoff holds long-leg IV '
      'constant (it won&rsquo;t be). Multi-expiry structures settle in stages. '
      'q=0 &rArr; short ITM American puts risk early assignment. '
      f'Snapshot {ctx["asof"]}.</div>')
  return "\n".join(out)


def searchbar_html(symbol: str, message: str = "") -> str:
  msg = f'<span class="sb-msg">{message}</span>' if message else ""
  return (
      '<form class="searchbar" method="get" action="/">'
      '<span class="sb-brand">OPTIONS&nbsp;MODEL</span>'
      '<label class="eyebrow" for="sym">Ticker</label>'
      f'<input id="sym" name="symbol" value="{symbol}" autocomplete="off" '
      'spellcheck="false" maxlength="8" aria-label="ticker symbol">'
      '<button type="submit">Analyze</button>'
      '<span class="sb-hint">any US ticker with listed options '
      '&middot; ~30&ndash;60s</span>' + msg + '</form>')


def landing_page() -> str:
  examples = "".join(
      f'<a class="ex" href="/?symbol={s}">{s}</a>' for s in
      ("SPCX", "AAPL", "SPY", "TSLA", "NVDA", "GME"))
  body = f'''
    <div class="errbox" style="border-left-color:var(--accent); max-width:720px">
      <div class="eyebrow">Options pricing-model reconstruction</div>
      <h2>Enter a ticker to rebuild its option-pricing model.</h2>
      <p>Pulls the live option chain, inverts Black-Scholes per option, and
      builds the implied-vol surface &sigma;(K,T), the ATM greeks and pricing
      parameters across every expiration within 12 months, then scores which
      model (BSM / American binomial / Black-76) reprices the chain. ~30&ndash;60s.</p>
      <div class="examples">{examples}</div>
    </div>
    <style>
      .examples {{ display:flex; gap:10px; flex-wrap:wrap; margin-top:16px; }}
      .ex {{ font-family:var(--mono); font-weight:700; letter-spacing:.06em;
        border:1px solid var(--line); border-radius:8px; padding:7px 13px;
        background:var(--surface-2); color:var(--ink); }}
      .ex:hover {{ border-color:var(--accent); text-decoration:none; }}
    </style>'''
  return _shell(searchbar_html("SPCX"), body)


def error_page(symbol: str, message: str) -> str:
  body = f'''
    <div class="errbox">
      <div class="eyebrow">Couldn&rsquo;t analyze {symbol or "&mdash;"}</div>
      <h2>{message}</h2>
      <p>Enter a ticker with an active options market &mdash; e.g.
      <span class="mono">SPCX</span>, <span class="mono">AAPL</span>,
      <span class="mono">SPY</span>, <span class="mono">TSLA</span>.</p>
    </div>'''
  return _shell(searchbar_html(symbol or "", ""), body)


# --------------------------------------------------------------------------- #
def main(argv=None):
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--symbol", default="SPCX")
  ap.add_argument("--months", type=int, default=12)
  ap.add_argument("--out", default=str(ROOT / "dashboard.html"))
  args = ap.parse_args(argv)
  # Static file must stand alone -> inline the plotly bundle.
  ctx = compute_context(args.symbol, args.months, embed_plotly=True)
  html = render_dashboard(ctx)
  Path(args.out).write_text(html, encoding="utf-8")
  kb = Path(args.out).stat().st_size / 1024
  print(f"{args.symbol}: wrote {args.out} ({kb:.0f} KB), "
        f"spot={ctx['spot']:.2f} q={ctx['q']:.4f} expiries={ctx['n_exp']}")


# --------------------------------------------------------------------------- #
# Page shell (CSS) + full template. {{ }} are literal braces for .format().
# --------------------------------------------------------------------------- #
STYLE = r"""<style>
  :root {
    --paper:#F6F7F9; --surface:#FFFFFF; --surface-2:#EEF1F4; --ink:#10161D;
    --muted:#5A6673; --faint:#8A95A2; --line:#E1E5EA; --mount:#FFFFFF;
    --accent:#0E9E92; --accent-2:#C6791E; --good:#128A5B; --warn:#B7791F;
    --shadow:0 1px 2px rgba(16,22,29,.06),0 8px 24px rgba(16,22,29,.06);
  }
  @media (prefers-color-scheme: dark) {
    :root {
      --paper:#0E141B; --surface:#161E27; --surface-2:#1E2833; --ink:#E7ECF1;
      --muted:#93A0AD; --faint:#63707D; --line:#26313C; --mount:#F4F6F8;
      --accent:#2DD4BF; --accent-2:#E0912F; --good:#34D399; --warn:#E0B341;
      --shadow:0 1px 2px rgba(0,0,0,.4),0 12px 30px rgba(0,0,0,.35);
    }
  }
  :root[data-theme="light"] {
    --paper:#F6F7F9; --surface:#FFFFFF; --surface-2:#EEF1F4; --ink:#10161D;
    --muted:#5A6673; --faint:#8A95A2; --line:#E1E5EA; --mount:#FFFFFF;
    --accent:#0E9E92; --accent-2:#C6791E; --good:#128A5B; --warn:#B7791F;
    --shadow:0 1px 2px rgba(16,22,29,.06),0 8px 24px rgba(16,22,29,.06);
  }
  :root[data-theme="dark"] {
    --paper:#0E141B; --surface:#161E27; --surface-2:#1E2833; --ink:#E7ECF1;
    --muted:#93A0AD; --faint:#63707D; --line:#26313C; --mount:#F4F6F8;
    --accent:#2DD4BF; --accent-2:#E0912F; --good:#34D399; --warn:#E0B341;
    --shadow:0 1px 2px rgba(0,0,0,.4),0 12px 30px rgba(0,0,0,.35);
  }
  * { box-sizing:border-box; }
  body { margin:0; }
  .wrap {
    --sans:"Helvetica Neue",system-ui,-apple-system,"Segoe UI",Roboto,Arial,sans-serif;
    --mono:ui-monospace,"SF Mono","Cascadia Code","JetBrains Mono",Menlo,Consolas,monospace;
    background:var(--paper); color:var(--ink); font-family:var(--sans);
    line-height:1.55; -webkit-font-smoothing:antialiased;
    padding:clamp(14px,3vw,36px); min-height:100vh;
  }
  .inner { max-width:1160px; margin:0 auto; }
  .mono { font-family:var(--mono); font-variant-numeric:tabular-nums; }
  .eyebrow { font-family:var(--mono); font-size:11px; letter-spacing:.14em;
    text-transform:uppercase; color:var(--faint); }

  /* search bar */
  .searchbar {
    display:flex; align-items:center; gap:12px; flex-wrap:wrap;
    background:var(--surface); border:1px solid var(--line); border-radius:12px;
    padding:12px 16px; box-shadow:var(--shadow); margin-bottom:22px;
  }
  .sb-brand { font-family:var(--mono); font-weight:700; font-size:12px;
    letter-spacing:.16em; color:var(--accent); }
  .searchbar label { margin-left:6px; }
  .searchbar input {
    font-family:var(--mono); font-size:18px; font-weight:700; letter-spacing:.06em;
    text-transform:uppercase; width:120px; padding:7px 11px; color:var(--ink);
    background:var(--paper); border:1px solid var(--line); border-radius:8px;
  }
  .searchbar input:focus { outline:2px solid var(--accent); outline-offset:1px; }
  .searchbar button {
    font-family:var(--sans); font-weight:700; font-size:14px; cursor:pointer;
    color:#fff; background:var(--accent); border:0; border-radius:8px; padding:9px 18px;
  }
  .searchbar button:hover { filter:brightness(1.06); }
  .sb-hint { color:var(--faint); font-size:12px; }
  .sb-msg { color:var(--warn); font-size:12.5px; font-family:var(--mono); }

  /* tabs */
  .tabbar { display:flex; gap:6px; margin-bottom:20px; border-bottom:1px solid var(--line); }
  .tabbtn { font-family:var(--mono); font-size:12px; font-weight:700; letter-spacing:.1em;
    text-transform:uppercase; color:var(--muted); background:none; border:0; cursor:pointer;
    padding:11px 16px; border-bottom:2px solid transparent; margin-bottom:-1px; }
  .tabbtn:hover { color:var(--ink); }
  .tabbtn[aria-selected="true"] { color:var(--accent); border-bottom-color:var(--accent); }
  .tabpanel { display:none; }
  .tabpanel.active { display:block; }
  .tabbtn .badge { color:var(--faint); font-weight:600; margin-left:6px; }

  /* strategy cards */
  .strat-intro { color:var(--muted); font-size:13.5px; max-width:74ch; margin:0 0 18px; }
  .strat-howto { background:var(--surface); border:1px solid var(--line);
    border-left:3px solid var(--accent); border-radius:12px; box-shadow:var(--shadow);
    padding:16px 20px; margin:0 0 22px; max-width:82ch; }
  .strat-howto h3 { margin:0 0 8px; font-size:15px; font-weight:800; letter-spacing:-.01em; }
  .strat-howto p { color:var(--muted); font-size:13px; margin:8px 0; }
  .strat-howto ol { color:var(--muted); font-size:13px; margin:8px 0 8px; padding-left:20px; }
  .strat-howto li { margin:5px 0; }
  .strat-howto b { color:var(--ink); }
  .strat { background:var(--surface); border:1px solid var(--line); border-radius:12px;
    box-shadow:var(--shadow); margin-bottom:18px; overflow:hidden; }
  .strat-hd { padding:16px 18px 6px; }
  .strat-hd h3 { margin:0 0 5px; font-size:16px; font-weight:800; letter-spacing:-.01em; }
  .strat-hd .th { color:var(--muted); font-size:13px; max-width:80ch; }
  .strat-body { display:grid; grid-template-columns:1.1fr 1fr; gap:8px 18px; padding:6px 18px 16px; align-items:start; }
  .strat-legs table { margin-top:6px; }
  .chips { display:flex; flex-wrap:wrap; gap:8px; margin-top:12px; }
  .chip { font-family:var(--mono); font-size:11.5px; font-weight:600; border-radius:7px;
    padding:4px 9px; border:1px solid var(--line); background:var(--surface-2); }
  .chip.debit { color:var(--warn); } .chip.credit { color:var(--good); }
  .chip.pos { color:var(--good); } .chip.neg { color:var(--warn); }
  .strat-plot { background:var(--mount); border:1px solid var(--line); border-radius:8px;
    padding:6px; overflow-x:auto; }
  .strat-plot img { display:block; width:100%; height:auto; }
  .strat-be { font-family:var(--mono); font-size:12px; color:var(--muted); margin-top:8px; }
  .req { font-family:var(--mono); font-size:10.5px; font-weight:700; letter-spacing:.04em;
    text-transform:uppercase; border-radius:6px; padding:2px 7px; margin-left:10px;
    vertical-align:middle; border:1px solid transparent; white-space:nowrap; }
  .req-spread { color:var(--good); background:color-mix(in srgb, var(--good) 12%, transparent);
    border-color:color-mix(in srgb, var(--good) 35%, transparent); }
  .req-naked { color:var(--warn); background:color-mix(in srgb, var(--warn) 12%, transparent);
    border-color:color-mix(in srgb, var(--warn) 35%, transparent); }
  .req-long { color:var(--muted); background:var(--surface-2); border-color:var(--line); }
  .strat-filter { display:flex; flex-wrap:wrap; align-items:center; gap:8px;
    margin:0 0 18px; }
  .sf-label { font-size:12.5px; color:var(--muted); font-weight:600; margin-right:2px; }
  .sf-btn { font-family:var(--mono); font-size:11.5px; font-weight:700; cursor:pointer;
    color:var(--muted); background:var(--surface); border:1px solid var(--line);
    border-radius:8px; padding:6px 11px; }
  .sf-btn:hover { color:var(--ink); border-color:var(--accent); }
  .sf-btn.active { color:#fff; background:var(--accent); border-color:var(--accent); }
  .sf-hint { font-size:12px; color:var(--faint); font-family:var(--mono); margin-left:2px; }
  .strat-alt { margin-top:10px; border-top:1px dashed var(--line); padding-top:8px; }
  .strat-alt summary { cursor:pointer; font-size:12px; font-weight:700; color:var(--accent); }
  .strat-alt summary:hover { filter:brightness(1.1); }
  .strat-alt .tablewrap { margin-top:8px; }
  @media (max-width:760px) { .strat-body { grid-template-columns:1fr; } }

  header.top { display:flex; flex-wrap:wrap; gap:18px 24px; align-items:flex-end;
    justify-content:space-between; border-bottom:1px solid var(--line); padding-bottom:22px; }
  .ticker { display:flex; align-items:center; gap:12px; }
  .ticker .sym { font-family:var(--mono); font-weight:700; font-size:13px;
    letter-spacing:.1em; color:var(--paper); background:var(--accent);
    padding:5px 9px; border-radius:5px; }
  h1 { font-size:clamp(22px,3vw,30px); line-height:1.1; margin:10px 0 6px;
    font-weight:800; letter-spacing:-.02em; text-wrap:balance; }
  .sub { color:var(--muted); font-size:14px; max-width:62ch; }
  .meta { display:flex; gap:20px; flex-wrap:wrap; margin-top:12px; }
  .meta > div { display:flex; flex-direction:column; gap:2px; }
  .meta .k { font-family:var(--mono); font-size:15px; font-weight:600; }

  .verdict { background:var(--surface); border:1px solid var(--line);
    border-left:3px solid var(--good); border-radius:10px; padding:14px 18px;
    box-shadow:var(--shadow); min-width:260px; max-width:340px; }
  .verdict .pill { display:inline-block; font-family:var(--mono); font-size:12px;
    font-weight:700; color:var(--good); letter-spacing:.04em;
    border:1px solid color-mix(in srgb,var(--good) 40%,transparent);
    background:color-mix(in srgb,var(--good) 12%,transparent);
    padding:3px 8px; border-radius:999px; margin-bottom:8px; }
  .verdict .model { font-weight:800; font-size:17px; letter-spacing:-.01em; }
  .verdict .note { color:var(--muted); font-size:12.5px; margin-top:5px; }

  .kpis { display:grid; grid-template-columns:repeat(4,1fr); gap:14px; margin:26px 0; }
  .kpi { background:var(--surface); border:1px solid var(--line); border-radius:10px;
    padding:15px 16px; box-shadow:var(--shadow); }
  .kpi .val { font-family:var(--mono); font-variant-numeric:tabular-nums;
    font-size:25px; font-weight:700; letter-spacing:-.01em; display:flex;
    align-items:baseline; gap:4px; }
  .kpi .val .u { font-size:14px; color:var(--faint); font-weight:600; }
  .kpi .lab { margin-top:4px; }

  .grid { display:grid; grid-template-columns:1fr 1fr; gap:18px; }
  .card { background:var(--surface); border:1px solid var(--line); border-radius:12px;
    box-shadow:var(--shadow); overflow:hidden; display:flex; flex-direction:column; }
  .card.full { grid-column:1 / -1; }
  .card .head { padding:16px 18px 4px; }
  .card h3 { margin:6px 0 0; font-size:16px; font-weight:700; letter-spacing:-.01em; }
  .card .read { color:var(--muted); font-size:13px; padding:6px 18px 0; }
  .mount { margin:14px; margin-top:12px; background:var(--mount); border:1px solid var(--line);
    border-radius:8px; padding:8px; overflow-x:auto; }
  .mount img { display:block; width:100%; height:auto; }
  .mount.plot3d { padding:4px; min-height:460px; }
  .mount.plot3d .plotly-graph-div { width:100% !important; }

  .findings { display:grid; grid-template-columns:1fr 1fr; gap:18px; margin-top:26px; }
  .panel { background:var(--surface); border:1px solid var(--line); border-radius:12px;
    padding:20px 22px; box-shadow:var(--shadow); }
  .panel h2 { font-size:15px; margin:8px 0 12px; font-weight:800; letter-spacing:-.01em; }
  .panel p { font-size:13.5px; color:var(--ink); margin:0 0 11px; }
  .panel p .lead { color:var(--accent); font-family:var(--mono); font-weight:600; }
  .panel ul { margin:0; padding-left:18px; font-size:13.5px; color:var(--ink); }
  .panel li { margin-bottom:7px; }
  .flag { border-left:3px solid var(--warn);
    background:color-mix(in srgb,var(--warn) 9%,transparent);
    padding:11px 14px; border-radius:0 8px 8px 0; font-size:13px; color:var(--ink); margin-top:4px; }

  .tablewrap { overflow-x:auto; margin-top:12px; }
  table { border-collapse:collapse; width:100%; font-family:var(--mono);
    font-size:12.5px; font-variant-numeric:tabular-nums; }
  th, td { text-align:right; padding:6px 10px; border-bottom:1px solid var(--line); white-space:nowrap; }
  th:first-child, td:first-child { text-align:left; }
  thead th { color:var(--faint); font-weight:600; font-size:11px; letter-spacing:.06em; text-transform:uppercase; }
  tbody tr:hover { background:var(--surface-2); }
  .scoretable td.ml, .scoretable th.ml { text-align:left; }
  .scoretable tr.win td { background:color-mix(in srgb,var(--good) 10%,transparent); }
  .tick { font-family:var(--mono); font-size:10px; letter-spacing:.06em; text-transform:uppercase;
    color:var(--good); margin-left:8px; border:1px solid color-mix(in srgb,var(--good) 45%,transparent);
    border-radius:999px; padding:1px 7px; vertical-align:middle; }
  .scorenote { display:flex; flex-wrap:wrap; gap:10px 20px; margin-top:14px; font-size:12.5px;
    color:var(--muted); align-items:center; }
  .tell { font-family:var(--mono); color:var(--accent-2); font-weight:600; }

  .gdefs { display:grid; gap:12px; grid-template-columns:repeat(auto-fit,minmax(185px,1fr)); margin-top:6px; }
  .gdef { background:var(--surface-2); border:1px solid var(--line); border-radius:10px; padding:13px 15px; }
  .gsym { font-family:var(--mono); font-size:24px; font-weight:700; color:var(--accent); line-height:1; }
  .gname { font-weight:700; font-size:14px; margin-top:7px; }
  .gder { font-family:var(--mono); font-weight:500; font-size:11.5px; color:var(--faint); margin-left:4px; }
  .gtxt { font-size:12.5px; color:var(--muted); margin-top:5px; }
  .gunit { margin-top:8px; }

  .errbox { background:var(--surface); border:1px solid var(--line); border-left:3px solid var(--warn);
    border-radius:12px; padding:26px 28px; box-shadow:var(--shadow); max-width:640px; }
  .errbox h2 { margin:8px 0 10px; font-size:19px; font-weight:800; }
  .errbox p { color:var(--muted); font-size:14px; }

  footer { margin-top:28px; padding-top:18px; border-top:1px solid var(--line); color:var(--muted); font-size:12.5px; }
  .code { font-family:var(--mono); font-size:12px; background:var(--surface-2); border:1px solid var(--line);
    border-radius:8px; padding:12px 14px; overflow-x:auto; white-space:pre; color:var(--ink); margin-top:10px; }
  a { color:var(--accent); text-decoration:none; }
  a:hover { text-decoration:underline; }

  /* analyzing spinner overlay */
  .spin-overlay { position:fixed; inset:0; display:none; z-index:9999;
    flex-direction:column; align-items:center; justify-content:center; gap:18px;
    background:color-mix(in srgb,var(--paper) 82%,transparent);
    -webkit-backdrop-filter:blur(3px); backdrop-filter:blur(3px); }
  .spin-overlay.on { display:flex; }
  .spin-ring { width:54px; height:54px; border-radius:50%;
    border:4px solid color-mix(in srgb,var(--accent) 22%,transparent);
    border-top-color:var(--accent); animation:spin .9s linear infinite; }
  @keyframes spin { to { transform:rotate(360deg); } }
  .spin-title { font-family:var(--mono); font-weight:700; font-size:15px; color:var(--ink); }
  .spin-title .spin-tick { color:var(--accent); }
  .spin-elapsed { color:var(--muted); font-weight:600; font-variant-numeric:tabular-nums; }
  .spin-sub { font-size:12.5px; color:var(--muted); }
  @media (prefers-reduced-motion: reduce) {
    .spin-ring { animation:none; opacity:.85; }
  }

  @media (max-width:860px) {
    .kpis { grid-template-columns:repeat(2,1fr); }
    .grid, .findings { grid-template-columns:1fr; }
    .card:not(.full) { grid-column:auto; }
  }
</style>"""

SPINNER = r"""
<div class="spin-overlay" id="spin" role="status" aria-live="polite">
  <div class="spin-ring"></div>
  <div class="spin-title">Analyzing <span class="spin-tick">&hellip;</span>
    <span class="spin-elapsed" aria-hidden="true">0s</span></div>
  <div class="spin-sub">Pulling the chain, building the surface &amp; scoring models &middot; ~30&ndash;60s</div>
</div>
<script>
(function () {
  var ov = document.getElementById("spin");
  if (!ov) return;
  var el = ov.querySelector(".spin-elapsed");
  var timer = null, t0 = 0;
  function tick() { if (el) el.textContent = Math.floor((Date.now() - t0) / 1000) + "s"; }
  function stop() { if (timer) { clearInterval(timer); timer = null; } ov.classList.remove("on"); }
  function show(sym) {
    var t = ov.querySelector(".spin-tick");
    if (t && sym) t.textContent = String(sym).toUpperCase();
    t0 = Date.now();
    if (el) el.textContent = "0s";
    ov.classList.add("on");
    if (timer) clearInterval(timer);
    timer = setInterval(tick, 250);
  }
  var form = document.querySelector(".searchbar");
  if (form) form.addEventListener("submit", function () {
    var inp = form.querySelector("#sym");
    if (inp && inp.value.trim()) show(inp.value.trim());
  });
  document.querySelectorAll('a[href*="symbol="]').forEach(function (a) {
    a.addEventListener("click", function () {
      var m = a.getAttribute("href").match(/symbol=([^&]+)/);
      show(m ? decodeURIComponent(m[1]) : "");
    });
  });
  // stop + hide if the page is restored from bfcache with the overlay still on
  window.addEventListener("pageshow", stop);
})();
</script>"""

TABS_JS = r"""
<script>
(function () {
  var bar = document.querySelector(".tabbar");
  if (!bar) return;
  var btns = bar.querySelectorAll(".tabbtn");
  btns.forEach(function (b) {
    b.addEventListener("click", function () {
      btns.forEach(function (x) { x.setAttribute("aria-selected", "false"); });
      document.querySelectorAll(".tabpanel").forEach(function (p) {
        p.classList.remove("active");
      });
      b.setAttribute("aria-selected", "true");
      var panel = document.getElementById(b.dataset.tab);
      if (panel) panel.classList.add("active");
      window.scrollTo({ top: 0, behavior: "instant" in window ? "instant" : "auto" });
    });
  });
})();
(function () {
  var btns = document.querySelectorAll("[data-strat-filter]");
  if (!btns.length) return;
  var cards = document.querySelectorAll(".strat[data-requires]");
  var empty = document.getElementById("strat-empty");
  var count = document.getElementById("strat-count");
  // level -> set of card requirements it may trade
  var allow = { naked: ["spread", "naked", "long"], spread: ["spread", "long"],
                long: ["long"] };
  function apply(level) {
    var ok = allow[level] || allow.naked, shown = 0;
    cards.forEach(function (c) {
      var vis = ok.indexOf(c.dataset.requires) !== -1;
      c.style.display = vis ? "" : "none";
      if (vis) shown++;
    });
    if (empty) empty.style.display = shown ? "none" : "";
    if (count) count.textContent = shown + " of " + cards.length + " shown";
  }
  btns.forEach(function (b) {
    b.addEventListener("click", function () {
      btns.forEach(function (x) { x.classList.remove("active"); });
      b.classList.add("active");
      apply(b.dataset.stratFilter);
    });
  });
  apply("naked");
})();
</script>"""

BODY = r"""
  <header class="top">
    <div class="brand">
      <div class="ticker">
        <span class="sym">{symbol}</span>
        <span class="eyebrow">options pricing-model reconstruction</span>
      </div>
      <h1>What model prices the {symbol} option chain?</h1>
      <p class="sub">Black-Scholes inverted per option on live mid quotes, with
      the vol surface, greeks and pricing parameters rebuilt across every
      expiration inside {months}&nbsp;months &mdash; all from one snapshot.</p>
      <div class="meta">
        <div><span class="eyebrow">Underlying</span><span class="k">${spot}</span></div>
        <div><span class="eyebrow">As of</span><span class="k">{asof}</span></div>
        <div><span class="eyebrow">Dividend q</span><span class="k">{q_disp}</span></div>
        <div><span class="eyebrow">Source</span><span class="k">Yahoo chain</span></div>
      </div>
    </div>
    <div class="verdict">
      <span class="pill">CLOSEST MODEL</span>
      <div class="model">Black&ndash;Scholes&ndash;Merton</div>
      <div class="note">{pill_note}</div>
    </div>
  </header>

  <section class="kpis">
    <div class="kpi"><div class="val">${spot}</div><div class="lab eyebrow">{symbol} spot</div></div>
    <div class="kpi"><div class="val">{n_exp}</div><div class="lab eyebrow">Expiries &le; {months}m</div></div>
    <div class="kpi"><div class="val">{feat_iv}<span class="u">%</span></div><div class="lab eyebrow">{feat_date} ATM IV</div></div>
    <div class="kpi"><div class="val">{iv_hi}&ndash;{iv_lo}<span class="u">%</span></div><div class="lab eyebrow">ATM IV term range</div></div>
  </section>
  {coverage_banner}

  <section class="grid">
    <div class="card">
      <div class="head"><span class="eyebrow">Implied-vol surface</span>
        <h3>&sigma;(K, T) &mdash; heatmap</h3></div>
      <p class="read">Implied vol across strikes (moneyness) and maturities,
      each point inverted from a live option mid.</p>
      <div class="mount"><img src="{heatmap}" alt="{symbol} implied vol surface heatmap"></div>
    </div>
    <div class="card">
      <div class="head"><span class="eyebrow">Implied-vol surface</span>
        <h3>&sigma;(K, T) &mdash; 3D, interactive</h3></div>
      <p class="read"><b>Drag to rotate</b> &middot; scroll to zoom &middot;
      double-click to reset. The smile curls across moneyness; the ridge shifts
      with time-to-expiry.</p>
      <div class="mount plot3d">%%SURF3D%%</div>
    </div>
    <div class="card full">
      <div class="head"><span class="eyebrow">Greeks(T)</span>
        <h3>ATM greeks as a function of maturity</h3></div>
      <p class="read">Black-Scholes sensitivities of the at-the-money option
      across the {months}-month expiration cross-section.</p>
      <div class="mount"><img src="{greeks}" alt="ATM greeks vs maturity"></div>
    </div>
    <div class="card full">
      <div class="head"><span class="eyebrow">Parameters(T)</span>
        <h3>Pricing parameters as a function of maturity</h3></div>
      <p class="read">&sigma;<sub>ATM</sub>(T) is the data-driven vol term
      structure; F(T)=S&middot;e<sup>(r&minus;q)T</sup> is the forward; r(T) is a
      flat placeholder (swap in a live Treasury curve).</p>
      <div class="mount"><img src="{params}" alt="Pricing parameters vs maturity"></div>
    </div>
  </section>

  <section class="grid" style="margin-top:18px">
    <div class="card full">
      <div class="head"><span class="eyebrow">Model scoring</span>
        <h3>Which model reprices the chain? &mdash; IV RMSE vs {ref_name} IV</h3></div>
      <p class="read">Each candidate is inverted to the market mid per option on
      the {feat_date} expiry; the table is the error vs the reference IV. Judge
      on the <b>liquid core</b> (|K/S&minus;1|&nbsp;&le;&nbsp;{core_pct}%). Lower is closer.</p>
      <div class="mount" style="background:var(--surface); padding:2px 6px">
        <div class="tablewrap">
          <table class="scoretable">
            <thead><tr><th class="ml">Model</th><th>RMSE core</th>
              <th>RMSE full</th><th>Unpriced</th><th>n</th></tr></thead>
            <tbody>{score_rows}</tbody>
          </table>
        </div>
      </div>
      <div class="scorenote" style="padding:0 18px 16px">
        <span>&#10003; {verdict_line}.</span>
        <span>Black-76 = BSM for an equity (identical RMSE).</span>
        <span>The RMSE gap is within the last-vs-mid reference noise; the
          <b>Unpriced</b> column shows where each model's inversion breaks down
          on illiquid deep-ITM strikes (American puts near intrinsic) &mdash;
          which is where the models genuinely differ.</span>
        <span>American tell (deep-ITM puts): IV(BSM)&minus;IV(CRR) reaches
          <span class="tell">{itm_tell}</span>.</span>
      </div>
    </div>
  </section>

  <section class="grid" style="margin-top:18px">
    <div class="card full">
      <div class="head"><span class="eyebrow">Reference</span>
        <h3>The greeks, defined</h3></div>
      <p class="read">Sensitivities of the option price V to each input, in the
      display units used in the greeks(T) panel (BSM, per share).</p>
      <div style="padding:2px 16px 18px"><div class="gdefs">{gdefs}</div></div>
    </div>
  </section>

  <section class="findings">
    <div class="panel">
      <h2>How the model was determined</h2>
      <p><span class="lead">IV is implied, not looked up.</span> Each option's
      volatility is the number that makes the model reprice it to the market mid.
      Quote vendors (Barchart, Yahoo) showing a per-option IV column are
      inverting a constant-parameter Black-Scholes, option by option.</p>
      <p>Three candidates are scored against the live chain &mdash; BSM
      (European), CRR binomial (American), and Black-76 (forward). Black-76
      collapses to BSM for an equity; American vs European can only diverge in
      the puts, and materially only in the illiquid deep-ITM tail.</p>
      <div class="flag"><b>Caveat.</b> Reference IV here is Yahoo's, a proxy. To
      test a specific vendor's engine, score against their displayed IV on the
      deep-ITM puts.</div>
    </div>
    <div class="panel">
      <h2>Calendar time <span style="color:var(--faint)">t</span> vs maturity <span style="color:var(--faint)">T</span></h2>
      <ul>
        <li><b>Implied vol needs no history</b> &mdash; it comes from today's
        price. One snapshot reconstructs and verifies the numbers.</li>
        <li><b>The surface &sigma;(K,T) and rate r(T) are cross-sectional</b>
        &mdash; from many strikes and the {n_exp} live expirations at a single
        instant, not from a time series.</li>
        <li><b>Only realized vol / IV dynamics need many dates</b> &mdash; and
        that is a different quantity from the implied vol shown here.</li>
      </ul>
      <div class="tablewrap">
        <table>
          <thead><tr><th>Expiry</th><th>Days</th><th>IV%</th><th>&Delta;</th>
            <th>&Gamma;</th><th>Vega</th><th>&Theta;/d</th><th>&rho;</th></tr></thead>
          <tbody>{trows}</tbody>
        </table>
      </div>
    </div>
  </section>

  <footer>
    <span class="eyebrow">Reproduce (CLI)</span>
    <div class="code">python src/build_dashboard.py --symbol {symbol}
python src/surfaces.py       --symbol {symbol} --months {months}
python src/app.py            # interactive: any ticker in the box</div>
    <p style="margin-top:12px"><span class="eyebrow">Audit trail</span> &nbsp;
    inputs &amp; results saved to <span class="mono">{audit_rel}/</span> &mdash;
    manifest.json, surface_points.csv, term_structure.csv, scoring_chain.csv,
    scoring_results.csv.</p>
    <p style="margin-top:10px">Snapshot {asof} &middot; {symbol} ${spot} &middot;
    q&nbsp;=&nbsp;{q_disp} &middot; data from the Yahoo option chain &middot;
    analytics from <span class="mono">spcx_model/</span>.</p>
  </footer>
"""

if __name__ == "__main__":
  main()
