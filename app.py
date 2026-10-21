import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="Signal Agent", page_icon="📡", layout="centered")
st.title("📡 Signal Agent")
st.caption("Real market data + math scoring. Sirf education/testing ke liye, financial advice nahi.")

# ---------------- Sidebar settings ----------------
symbol = st.sidebar.text_input("Symbol (Yahoo)", "GC=F")  # GC=F = Gold futures
tf = st.sidebar.selectbox("Timeframe", ["15m", "30m", "1h"], index=2)
lookback = st.sidebar.slider("Sweep lookback (bars)", 10, 50, 20)
atr_mult = st.sidebar.slider("SL = ATR x", 0.5, 3.0, 1.5, 0.1)
rr = st.sidebar.slider("Risk:Reward", 1.0, 4.0, 2.0, 0.5)
min_score = st.sidebar.slider("Minimum score", 40, 100, 70, 5)


@st.cache_data(ttl=120)
def load(sym, interval):
    period = "60d" if interval in ("15m", "30m") else "180d"
    df = yf.download(sym, period=period, interval=interval, auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df[["Open", "High", "Low", "Close"]].dropna()


def build(df, lb, mult, rr_, min_sc):
    d = df.copy()
    d["ema50"] = d.Close.ewm(span=50, adjust=False).mean()
    d["ema200"] = d.Close.ewm(span=200, adjust=False).mean()

    delta = d.Close.diff()
    up = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    dn = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    d["rsi"] = 100 - 100 / (1 + up / dn.replace(0, np.nan))

    tr = pd.concat(
        [d.High - d.Low, (d.High - d.Close.shift()).abs(), (d.Low - d.Close.shift()).abs()],
        axis=1,
    ).max(axis=1)
    d["atr"] = tr.ewm(alpha=1 / 14, adjust=False).mean()

    prev_low = d.Low.shift(1).rolling(lb).min()
    prev_high = d.High.shift(1).rolling(lb).max()
    bull_sweep = (d.Low < prev_low) & (d.Close > prev_low)
    bear_sweep = (d.High > prev_high) & (d.Close < prev_high)

    # Score (0-100): sweep 40 + trend 30 + price vs EMA50 15 + RSI zone 15
    buy = (
        40 * bull_sweep
        + 30 * (d.ema50 > d.ema200)
        + 15 * (d.Close > d.ema50)
        + 15 * d.rsi.between(40, 65)
    )
    sell = (
        40 * bear_sweep
        + 30 * (d.ema50 < d.ema200)
        + 15 * (d.Close < d.ema50)
        + 15 * d.rsi.between(35, 60)
    )
    d["buy_score"] = np.where(bull_sweep, buy, 0)
    d["sell_score"] = np.where(bear_sweep, sell, 0)

    d["side"] = ""
    d.loc[d.buy_score >= min_sc, "side"] = "BUY"
    d.loc[d.sell_score >= min_sc, "side"] = "SELL"
    d["score"] = np.where(d.side == "BUY", d.buy_score, np.where(d.side == "SELL", d.sell_score, 0))

    risk = d.atr * mult
    d["entry"] = d.Close
    d["sl"] = np.where(d.side == "BUY", d.Close - risk, d.Close + risk)
    d["tp"] = np.where(d.side == "BUY", d.Close + risk * rr_, d.Close - risk * rr_)
    return d.dropna(subset=["ema200", "atr"])


def backtest(d, horizon=30):
    results = []
    sig = d[d.side != ""]
    for ts, r in sig.iterrows():
        i = d.index.get_loc(ts)
        fut = d.iloc[i + 1 : i + 1 + horizon]
        out = None
        for _, f in fut.iterrows():
            if r.side == "BUY":
                if f.Low <= r.sl:
                    out = -1.0
                    break
                if f.High >= r.tp:
                    out = rr
                    break
            else:
                if f.High >= r.sl:
                    out = -1.0
                    break
                if f.Low <= r.tp:
                    out = rr
                    break
        if out is not None:
            results.append(out)
    return results


try:
    raw = load(symbol, tf)
except Exception as e:
    st.error(f"Data load nahi hua: {e}")
    st.stop()

if len(raw) < 250:
    st.warning("Data kam hai. Symbol ya timeframe change karke dekhein.")
    st.stop()

d = build(raw, lookback, atr_mult, rr, min_score)
sigs = d[d.side != ""]

# ---------------- Latest signal card ----------------
st.subheader("Latest signal")
if sigs.empty:
    st.info("Abhi koi signal nahi. Minimum score kam karke dekhein.")
else:
    last = sigs.iloc[-1]
    bars_ago = len(d) - 1 - d.index.get_loc(sigs.index[-1])
    color = "🟢" if last.side == "BUY" else "🔴"
    st.markdown(f"### {color} {last.side}  |  Score {int(last.score)}/100")
    c1, c2, c3 = st.columns(3)
    c1.metric("Entry", f"{last.entry:.2f}")
    c2.metric("Stop Loss", f"{last.sl:.2f}")
    c3.metric("Take Profit", f"{last.tp:.2f}")
    st.caption(
        f"{bars_ago} bar(s) pehle ({sigs.index[-1]}). RSI {last.rsi:.0f}, ATR {last.atr:.2f}. "
        "Reason: liquidity sweep + trend (EMA50/200) + RSI zone."
    )

# ---------------- Backtest ----------------
st.subheader("Backtest (is data par)")
res = backtest(d)
if res:
    wins = sum(1 for x in res if x > 0)
    total_r = sum(res)
    b1, b2, b3 = st.columns(3)
    b1.metric("Trades", len(res))
    b2.metric("Win rate", f"{wins / len(res) * 100:.0f}%")
    b3.metric("Total R", f"{total_r:.1f}")
    st.caption("R = risk units. Past result future ki guarantee nahi. Pehle demo par test karein.")
else:
    st.info("Backtest ke liye signals nahi mile.")

# ---------------- Chart ----------------
st.subheader("Chart")
view = d.tail(150)
fig = go.Figure(
    [
        go.Candlestick(
            x=view.index, open=view.Open, high=view.High, low=view.Low, close=view.Close, name="Price"
        ),
        go.Scatter(x=view.index, y=view.ema50, name="EMA50", line=dict(width=1)),
        go.Scatter(x=view.index, y=view.ema200, name="EMA200", line=dict(width=1)),
    ]
)
for side, sym_, col in (("BUY", "triangle-up", "green"), ("SELL", "triangle-down", "red")):
    s = view[view.side == side]
    fig.add_trace(
        go.Scatter(
            x=s.index, y=s.Close, mode="markers", name=side,
            marker=dict(symbol=sym_, size=12, color=col),
        )
    )
fig.update_layout(height=450, xaxis_rangeslider_visible=False, margin=dict(l=0, r=0, t=10, b=0))
st.plotly_chart(fig, use_container_width=True)

# ---------------- History ----------------
st.subheader("Recent signals")
if not sigs.empty:
    tbl = sigs.tail(10)[["side", "score", "entry", "sl", "tp"]].round(2).iloc[::-1]
    st.dataframe(tbl, use_container_width=True)
