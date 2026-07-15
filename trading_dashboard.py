"""
Daily Stock Signal Dashboard
=============================
Pulls daily price history for a small watchlist, computes classic technical
indicators (SMA, RSI, MACD), derives a simple BUY / SELL / HOLD signal for
each stock, and writes everything to a single self-contained HTML dashboard
you can open in your browser.

This tool does NOT place trades. It only surfaces signals for you to review.

HOW TO RUN
----------
1. Install dependencies (one time):
       pip install yfinance pandas plotly

2. Edit the WATCHLIST list below (max ~5 tickers keeps it fast and readable).

3. Run it:
       python trading_dashboard.py

4. Open the generated file:
       dashboard.html

5. (Optional) Automate the daily check:
   - Mac/Linux: add a cron job, e.g. run at 9:00am every weekday:
       0 9 * * 1-5 /usr/bin/python3 /path/to/trading_dashboard.py
   - Windows: use Task Scheduler to run this script daily.

DISCLAIMER
----------
This is a technical-analysis educational tool, not financial advice.
Indicators are lagging by nature and can produce false signals, especially
in choppy or low-volume markets. Always do your own research and consider
your own risk tolerance before trading.
"""

import sys
from datetime import datetime

import pandas as pd

# ---------------------------------------------------------------------------
# CONFIG — edit this section
# ---------------------------------------------------------------------------
WATCHLIST = ["WFC", "CSCO", "XOM", "TGT", "SBUX"]   # Wells Fargo, Cisco, ExxonMobil, Target, Starbucks ($50-150 range)
LOOKBACK_PERIOD = "1y"      # how much history to pull (e.g. "6mo", "1y", "2y")
SMA_FAST = 50
SMA_SLOW = 200
RSI_PERIOD = 14
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
OUTPUT_FILE = "dashboard.html"
CURRENCY_SYMBOL = "$"

# Optional: Google sign-in + cloud sync for your buy price/qty entries, so
# they survive across devices and browser data clears instead of relying on
# browser local storage. Leave these as-is to skip cloud sync (the dashboard
# still works fine — entries just stay local-only). To enable it, create a
# free Firebase project (see README.md for exact steps) and paste your web
# app's config values in below.
FIREBASE_CONFIG = {
    apiKey: "AIzaSyD-abc123456789",
    authDomain: "stock-dashboard-12345.firebaseapp.com",
    projectId: "stock-dashboard-12345",
    storageBucket: "stock-dashboard-12345.appspot.com",
    messagingSenderId: "123456789012",
    appId: "1:123456789012:web:abcdef123456",
}
#-----------------------------------------------------------------------

def fetch_data(ticker: str, period: str) -> pd.DataFrame:
    """Download daily OHLCV data for a ticker."""
    import yfinance as yf
    df = yf.download(ticker, period=period, interval="1d", progress=False, auto_adjust=True)
    if df.empty:
        raise ValueError(f"No data returned for {ticker}. Check the ticker symbol.")
    # yfinance sometimes returns MultiIndex columns for a single ticker — flatten them
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    # Yahoo sometimes includes a placeholder row for "today" before that day has actually
    # traded (e.g. if this runs before market open), with empty Open/Close. Drop any such
    # incomplete trailing rows so "latest" always means the last fully-completed session.
    df = df.dropna(subset=["Open", "High", "Low", "Close"])
    if df.empty:
        raise ValueError(f"No complete trading data available for {ticker} after filtering.")
    return df


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add SMA, RSI, MACD columns to the dataframe."""
    df = df.copy()

    # Simple Moving Averages
    df[f"SMA{SMA_FAST}"] = df["Close"].rolling(SMA_FAST).mean()
    df[f"SMA{SMA_SLOW}"] = df["Close"].rolling(SMA_SLOW).mean()

    # RSI (Wilder's smoothing)
    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / RSI_PERIOD, min_periods=RSI_PERIOD, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / RSI_PERIOD, min_periods=RSI_PERIOD, adjust=False).mean()
    rs = avg_gain / avg_loss
    df["RSI"] = 100 - (100 / (1 + rs))

    # MACD
    ema_fast = df["Close"].ewm(span=MACD_FAST, adjust=False).mean()
    ema_slow = df["Close"].ewm(span=MACD_SLOW, adjust=False).mean()
    df["MACD"] = ema_fast - ema_slow
    df["MACD_signal"] = df["MACD"].ewm(span=MACD_SIGNAL, adjust=False).mean()
    df["MACD_hist"] = df["MACD"] - df["MACD_signal"]

    return df


def generate_signal(df: pd.DataFrame) -> dict:
    """
    Look at the most recent bar(s) and derive a simple BUY / SELL / HOLD
    signal plus the reasons behind it. This is intentionally simple and
    transparent so you can see exactly why it fired.
    """
    latest = df.iloc[-1]
    prev = df.iloc[-2]

    reasons = []
    score = 0  # positive = bullish, negative = bearish

    # 1. Trend: price vs long SMA
    if pd.notna(latest[f"SMA{SMA_SLOW}"]):
        if latest["Close"] > latest[f"SMA{SMA_SLOW}"]:
            score += 1
            reasons.append(f"Price above SMA{SMA_SLOW} (long-term uptrend)")
        else:
            score -= 1
            reasons.append(f"Price below SMA{SMA_SLOW} (long-term downtrend)")

    # 2. Golden cross / death cross (SMA_FAST vs SMA_SLOW)
    if pd.notna(latest[f"SMA{SMA_FAST}"]) and pd.notna(latest[f"SMA{SMA_SLOW}"]):
        fast_now, slow_now = latest[f"SMA{SMA_FAST}"], latest[f"SMA{SMA_SLOW}"]
        fast_prev, slow_prev = prev[f"SMA{SMA_FAST}"], prev[f"SMA{SMA_SLOW}"]
        if fast_prev <= slow_prev and fast_now > slow_now:
            score += 2
            reasons.append(f"Golden cross: SMA{SMA_FAST} just crossed above SMA{SMA_SLOW}")
        elif fast_prev >= slow_prev and fast_now < slow_now:
            score -= 2
            reasons.append(f"Death cross: SMA{SMA_FAST} just crossed below SMA{SMA_SLOW}")

    # 3. MACD crossover
    if pd.notna(latest["MACD"]) and pd.notna(latest["MACD_signal"]):
        if prev["MACD"] <= prev["MACD_signal"] and latest["MACD"] > latest["MACD_signal"]:
            score += 2
            reasons.append("MACD crossed above its signal line (bullish momentum shift)")
        elif prev["MACD"] >= prev["MACD_signal"] and latest["MACD"] < latest["MACD_signal"]:
            score -= 2
            reasons.append("MACD crossed below its signal line (bearish momentum shift)")
        elif latest["MACD"] > latest["MACD_signal"]:
            score += 1
            reasons.append("MACD above signal line (momentum still positive)")
        else:
            score -= 1
            reasons.append("MACD below signal line (momentum still negative)")

    # 4. RSI — overbought / oversold
    if pd.notna(latest["RSI"]):
        if latest["RSI"] < RSI_OVERSOLD:
            score += 1
            reasons.append(f"RSI at {latest['RSI']:.1f} — oversold, watch for a bounce")
        elif latest["RSI"] > RSI_OVERBOUGHT:
            score -= 1
            reasons.append(f"RSI at {latest['RSI']:.1f} — overbought, watch for a pullback")
        else:
            reasons.append(f"RSI at {latest['RSI']:.1f} — neutral zone")

    # Translate score to a signal label
    if score >= 3:
        label = "BUY"
    elif score <= -3:
        label = "SELL"
    else:
        label = "HOLD"

    return {
        "signal": label,
        "score": score,
        "reasons": reasons,
        "open": latest["Open"],
        "close": latest["Close"],
        "date": df.index[-1].strftime("%Y-%m-%d"),
    }


def build_chart_html(ticker: str, df: pd.DataFrame) -> str:
    """Build a Plotly price+indicator chart for one ticker, return as HTML div."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        row_heights=[0.55, 0.2, 0.25], vertical_spacing=0.03,
        subplot_titles=(f"{ticker} — Price & Moving Averages", "RSI", "MACD"),
    )

    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
        name="Price", showlegend=False,
    ), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df[f"SMA{SMA_FAST}"], name=f"SMA{SMA_FAST}",
                              line=dict(width=1.3)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df[f"SMA{SMA_SLOW}"], name=f"SMA{SMA_SLOW}",
                              line=dict(width=1.3)), row=1, col=1)

    fig.add_trace(go.Scatter(x=df.index, y=df["RSI"], name="RSI",
                              line=dict(width=1.3, color="#8e44ad")), row=2, col=1)
    fig.add_hline(y=RSI_OVERBOUGHT, line_dash="dot", line_color="red", row=2, col=1)
    fig.add_hline(y=RSI_OVERSOLD, line_dash="dot", line_color="green", row=2, col=1)

    fig.add_trace(go.Bar(x=df.index, y=df["MACD_hist"], name="MACD Hist",
                          marker_color="#95a5a6"), row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["MACD"], name="MACD",
                              line=dict(width=1.3, color="#2980b9")), row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["MACD_signal"], name="Signal",
                              line=dict(width=1.3, color="#e67e22")), row=3, col=1)

    fig.update_layout(
        height=700, margin=dict(l=40, r=20, t=40, b=20),
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", y=1.08),
        template="plotly_white",
        dragmode="pan",   # dragging pans the chart by default (no more accidental box/lasso select)
    )
    return fig.to_html(
        full_html=False,
        include_plotlyjs=False,
        config={
            "scrollZoom": True,                          # mouse wheel / trackpad pinch zooms in and out
            "modeBarButtonsToRemove": ["lasso2d", "select2d"],  # these caused the "stuck lasso" issue — removed entirely
            "displaylogo": False,
        },
    )


def build_dashboard(results: list[dict]) -> str:
    """Assemble the full HTML dashboard page from per-ticker results."""
    signal_colors = {"BUY": "#1e8e3e", "SELL": "#d93025", "HOLD": "#e8a33d"}

    summary_rows = ""
    for r in results:
        color = signal_colors[r["signal"]]
        t = r["ticker"]
        summary_rows += f"""
        <tr>
          <td><strong>{t}</strong></td>
          <td>{CURRENCY_SYMBOL}{r['open']:.2f}</td>
          <td>{CURRENCY_SYMBOL}{r['close']:.2f}</td>
          <td><span style="background:{color};color:white;padding:4px 10px;
              border-radius:12px;font-weight:600;">{r['signal']}</span></td>
          <td style="font-size:0.85em;color:#555;">{'; '.join(r['reasons'])}</td>
          <td><input type="number" step="0.01" min="0" class="pos-input" id="buy_{t}"
              placeholder="e.g. {r['close']:.2f}" oninput="updatePL('{t}')"></td>
          <td><input type="number" step="1" min="0" class="pos-input qty-input" id="qty_{t}"
              placeholder="shares" oninput="updatePL('{t}')"></td>
          <td id="pl_{t}"><span style="color:#999;">—</span></td>
        </tr>"""

    chart_sections = ""
    for r in results:
        chart_sections += f"""
        <div class="chart-card">
          {r['chart_html']}
        </div>"""

    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    current_prices_js = "{" + ", ".join(f'"{r["ticker"]}": {r["close"]}' for r in results) + "}"
    firebase_config_js = "{" + ", ".join(f'"{k}": "{v}"' for k, v in FIREBASE_CONFIG.items()) + "}"

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Stock Signal Dashboard</title>
<script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
<script src="https://www.gstatic.com/firebasejs/10.12.0/firebase-app-compat.js"></script>
<script src="https://www.gstatic.com/firebasejs/10.12.0/firebase-auth-compat.js"></script>
<script src="https://www.gstatic.com/firebasejs/10.12.0/firebase-firestore-compat.js"></script>
<style>
  body {{ font-family: -apple-system, Segoe UI, Roboto, Arial, sans-serif;
         background:#f5f6f8; margin:0; padding:24px; color:#1a1a1a; }}
  h1 {{ margin-bottom:4px; }}
  .timestamp {{ color:#777; margin-bottom:24px; font-size:0.9em; }}
  table {{ width:100%; border-collapse:collapse; background:white;
          box-shadow:0 1px 3px rgba(0,0,0,0.1); border-radius:8px; overflow:hidden; }}
  th, td {{ text-align:left; padding:12px 16px; border-bottom:1px solid #eee; }}
  th {{ background:#fafafa; font-size:0.8em; text-transform:uppercase; color:#666; }}
  .chart-card {{ background:white; border-radius:8px; padding:16px; margin-top:24px;
                box-shadow:0 1px 3px rgba(0,0,0,0.1); }}
  .disclaimer {{ margin-top:32px; padding:16px; background:#fff8e1; border-left:4px solid #e8a33d;
                font-size:0.85em; color:#555; border-radius:4px; }}
  .pos-input {{ width:90px; padding:6px 8px; border:1px solid #ddd; border-radius:4px; font-size:0.9em; }}
  .qty-input {{ width:70px; }}
  .position-note {{ margin-top:12px; padding:12px 16px; background:#eef4ff; border-left:4px solid #3b82f6;
                    font-size:0.82em; color:#444; border-radius:4px; }}
  .auth-bar {{ display:flex; align-items:center; gap:12px; margin-bottom:16px; }}
  .auth-btn {{ padding:8px 16px; border-radius:6px; border:1px solid #ddd; background:white;
              cursor:pointer; font-size:0.9em; font-weight:600; }}
  .auth-btn:hover {{ background:#f5f5f5; }}
  #userLabel {{ font-size:0.85em; color:#555; }}
</style>
</head>
<body>
  <h1>📊 Daily Stock Signal Dashboard</h1>
  <div class="timestamp">Generated {now}</div>

  <div class="auth-bar">
    <button id="signInBtn" class="auth-btn" onclick="signIn()" style="display:none;">Sign in with Google to sync across devices</button>
    <button id="signOutBtn" class="auth-btn" onclick="signOutUser()" style="display:none;">Sign out</button>
    <span id="userLabel"></span>
  </div>

  <table>
    <tr>
      <th>Ticker</th><th>Open</th><th>Last Close</th><th>Signal</th><th>Why</th>
      <th>Your Buy Price</th><th>Qty</th><th>P/L</th>
    </tr>
    {summary_rows}
  </table>

  <div class="position-note" id="positionNote">
    Enter your buy price when you buy (and optionally quantity) — leave it
    blank to clear it when you sell. Currently saved only in this browser;
    sign in with Google above to sync across devices instead.
  </div>

  {chart_sections}

  <div class="disclaimer">
    <strong>Not financial advice.</strong> These signals come from lagging technical
    indicators (SMA crossovers, RSI, MACD) applied mechanically to recent price
    history. They can and do produce false signals. Use this as one input among many,
    do your own research, and only invest what you can afford to lose.
  </div>

  <script>
    const currentPrices = {current_prices_js};
    const currencySymbol = "{CURRENCY_SYMBOL}";
    const firebaseConfig = {firebase_config_js};

    let currentUser = null;
    let userPositionsRef = null;
    let firebaseReady = false;

    // Only actually initialize Firebase if the config has been filled in —
    // otherwise silently fall back to local-only storage so the page never breaks.
    try {{
      if (firebaseConfig.apiKey && firebaseConfig.apiKey !== "YOUR_API_KEY") {{
        firebase.initializeApp(firebaseConfig);
        firebaseReady = true;
      }}
    }} catch (e) {{
      console.warn("Firebase not configured — using local browser storage only.", e);
    }}

    function signIn() {{
      const provider = new firebase.auth.GoogleAuthProvider();
      firebase.auth().signInWithPopup(provider).catch(function(err) {{
        alert("Sign-in failed: " + err.message);
      }});
    }}

    function signOutUser() {{
      firebase.auth().signOut();
    }}

    function updatePL(ticker, shouldSave) {{
      if (shouldSave === undefined) shouldSave = true;
      const buyInput = document.getElementById('buy_' + ticker);
      const qtyInput = document.getElementById('qty_' + ticker);
      const plCell = document.getElementById('pl_' + ticker);
      const buy = parseFloat(buyInput.value);
      const qty = parseFloat(qtyInput.value) || 0;
      const current = currentPrices[ticker];

      if (!buy || buy <= 0) {{
        plCell.innerHTML = '<span style="color:#999;">—</span>';
        if (shouldSave) clearPosition(ticker);
        return;
      }}

      const pctChange = ((current - buy) / buy) * 100;
      const sign = pctChange >= 0 ? '+' : '';
      const color = pctChange >= 0 ? '#1e8e3e' : '#d93025';

      let html = '<span style="color:' + color + ';font-weight:600;">' +
                 sign + pctChange.toFixed(2) + '%</span>';

      if (qty > 0) {{
        const dollarChange = (current - buy) * qty;
        const dSign = dollarChange >= 0 ? '+' : '-';
        html += '<br><span style="color:' + color + ';font-size:0.85em;">' +
                dSign + currencySymbol + Math.abs(dollarChange).toFixed(2) + '</span>';
      }}

      plCell.innerHTML = html;
      if (shouldSave) savePosition(ticker, buyInput.value, qtyInput.value);
    }}

    function savePosition(ticker, buy, qty) {{
      if (currentUser && userPositionsRef) {{
        const update = {{}};
        update[ticker] = {{ buy: buy, qty: qty }};
        userPositionsRef.set(update, {{ merge: true }}).catch(function(err) {{
          console.error("Cloud save failed, falling back to local storage:", err);
          localStorage.setItem('buyPrice_' + ticker, buy);
          localStorage.setItem('qty_' + ticker, qty);
        }});
      }} else {{
        localStorage.setItem('buyPrice_' + ticker, buy);
        localStorage.setItem('qty_' + ticker, qty);
      }}
    }}

    function clearPosition(ticker) {{
      if (currentUser && userPositionsRef) {{
        const update = {{}};
        update[ticker] = firebase.firestore.FieldValue.delete();
        userPositionsRef.set(update, {{ merge: true }}).catch(function(err) {{
          console.error("Cloud clear failed:", err);
        }});
      }} else {{
        localStorage.removeItem('buyPrice_' + ticker);
        localStorage.removeItem('qty_' + ticker);
      }}
    }}

    function loadPositionLocal(ticker) {{
      const savedBuy = localStorage.getItem('buyPrice_' + ticker);
      const savedQty = localStorage.getItem('qty_' + ticker);
      if (savedBuy) document.getElementById('buy_' + ticker).value = savedBuy;
      if (savedQty) document.getElementById('qty_' + ticker).value = savedQty;
      updatePL(ticker, false);
    }}

    function loadPositionsFromCloud() {{
      userPositionsRef.get().then(function(doc) {{
        const data = doc.exists ? doc.data() : {{}};
        Object.keys(currentPrices).forEach(function(ticker) {{
          const pos = data[ticker];
          if (pos) {{
            document.getElementById('buy_' + ticker).value = pos.buy;
            document.getElementById('qty_' + ticker).value = pos.qty;
          }}
          updatePL(ticker, false);
        }});
      }}).catch(function(err) {{
        console.error("Cloud load failed, falling back to local storage:", err);
        Object.keys(currentPrices).forEach(loadPositionLocal);
      }});
    }}

    document.addEventListener('DOMContentLoaded', function() {{
      if (firebaseReady) {{
        document.getElementById('signInBtn').style.display = 'inline-block';
        firebase.auth().onAuthStateChanged(function(user) {{
          currentUser = user;
          const signInBtn = document.getElementById('signInBtn');
          const signOutBtn = document.getElementById('signOutBtn');
          const userLabel = document.getElementById('userLabel');
          const note = document.getElementById('positionNote');
          if (user) {{
            signInBtn.style.display = 'none';
            signOutBtn.style.display = 'inline-block';
            userLabel.textContent = 'Synced as ' + user.displayName;
            note.textContent = 'Enter your buy price when you buy, and clear it when you sell — synced to your Google account across devices.';
            userPositionsRef = firebase.firestore().collection('positions').doc(user.uid);
            loadPositionsFromCloud();
          }} else {{
            signInBtn.style.display = 'inline-block';
            signOutBtn.style.display = 'none';
            userLabel.textContent = '';
            note.textContent = 'Enter your buy price when you buy (and optionally quantity) — leave it blank to clear it when you sell. Currently saved only in this browser; sign in with Google above to sync across devices instead.';
            userPositionsRef = null;
            Object.keys(currentPrices).forEach(loadPositionLocal);
          }}
        }});
      }} else {{
        Object.keys(currentPrices).forEach(loadPositionLocal);
      }}
    }});
  </script>
</body>
</html>"""


def main():
    results = []
    for ticker in WATCHLIST:
        print(f"Fetching {ticker}...")
        try:
            df = fetch_data(ticker, LOOKBACK_PERIOD)
            df = compute_indicators(df)
            sig = generate_signal(df)
            sig["ticker"] = ticker
            sig["chart_html"] = build_chart_html(ticker, df)
            results.append(sig)
            print(f"  -> {sig['signal']} at ${sig['close']:.2f}")
        except Exception as e:
            print(f"  !! Failed to process {ticker}: {e}", file=sys.stderr)

    if not results:
        print("No results generated — check your internet connection and ticker symbols.")
        sys.exit(1)

    html = build_dashboard(results)
    with open(OUTPUT_FILE, "w") as f:
        f.write(html)
    print(f"\nDashboard written to {OUTPUT_FILE} — open it in your browser.")


if __name__ == "__main__":
    main()
