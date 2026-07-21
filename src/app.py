"""Local web app: type any ticker in a box, get its option-pricing dashboard.

A published Artifact runs under a strict CSP and cannot fetch market data, so
live ticker entry needs a real backend. This is a zero-dependency stdlib
http.server that runs the same pipeline as build_dashboard for whatever symbol
is submitted, and returns the rendered dashboard.

    python src/app.py                 # serves http://127.0.0.1:8000
    python src/app.py --port 8080

Then open the URL and type a ticker (SPCX, AAPL, SPY, ...). Each analysis pulls
the chain from Yahoo and takes ~30-60s.
"""

from __future__ import annotations

import argparse
import http.server
import sys
import urllib.parse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import build_dashboard as bd  # noqa: E402
from surfaces import plotly_js  # noqa: E402


def page(symbol: str, inner: str) -> bytes:
  title = f"{symbol} · options model" if symbol else "Options model dashboard"
  return (
      "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
      "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">"
      f"<title>{title}</title>"
      "<style>html,body{margin:0;background:#0E141B}</style></head><body>"
      + inner + "</body></html>").encode("utf-8")


class Handler(http.server.BaseHTTPRequestHandler):
  def do_GET(self):
    u = urllib.parse.urlparse(self.path)
    if u.path == "/plotly.js":
      self._serve_plotly()
      return
    if u.path not in ("/", "/index.html"):
      self.send_error(404, "Not found")
      return
    qs = urllib.parse.parse_qs(u.query)
    symbol = (qs.get("symbol", [""])[0] or "").strip().upper()
    try:
      if not symbol:
        inner = bd.landing_page()
      else:
        sys.stderr.write(f"[analyze] {symbol} ...\n")
        ctx = bd.compute_context(symbol)
        inner = bd.render_dashboard(ctx)
    except ValueError as e:
      inner = bd.error_page(symbol, str(e))
    except Exception as e:  # network hiccup, Yahoo shape change, etc.
      inner = bd.error_page(symbol, f"Unexpected error: {e}")

    data = page(symbol, inner)
    self.send_response(200)
    self.send_header("Content-Type", "text/html; charset=utf-8")
    self.send_header("Content-Length", str(len(data)))
    self.end_headers()
    self.wfile.write(data)

  def _serve_plotly(self):
    data = plotly_js().encode("utf-8")
    self.send_response(200)
    self.send_header("Content-Type", "application/javascript; charset=utf-8")
    self.send_header("Content-Length", str(len(data)))
    self.send_header("Cache-Control", "public, max-age=86400")  # browser-cache
    self.end_headers()
    self.wfile.write(data)

  def log_message(self, *args):  # keep the console quiet
    pass


def main(argv=None):
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--port", type=int, default=8000)
  ap.add_argument("--host", default="127.0.0.1")
  args = ap.parse_args(argv)
  srv = http.server.ThreadingHTTPServer((args.host, args.port), Handler)
  url = f"http://{args.host}:{args.port}"
  print(f"Options-model dashboard serving at {url}")
  print("Open it and type a ticker (SPCX, AAPL, SPY, ...). Ctrl-C to stop.")
  try:
    srv.serve_forever()
  except KeyboardInterrupt:
    print("\nstopped.")


if __name__ == "__main__":
  main()
