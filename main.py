import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import yfinance as yf
from datetime import datetime, timedelta

# ─── PAGE CONFIG ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="AI Trading Backtester", layout="wide")
st.title("📈 AI Trading Strategy Backtester")
st.markdown("Build your own trading strategy and validate it against real historical data.")

# ─── SIDEBAR: STRATEGY BUILDER ──────────────────────────────────────────────────
st.sidebar.header("🛠️ Strategy Builder")

ticker = st.sidebar.text_input("Stock Ticker (e.g. AAPL, MSFT, TSLA)", value="AAPL").upper()

col1, col2 = st.sidebar.columns(2)
start_date = col1.date_input("Start Date", datetime.today() - timedelta(days=365*3))
end_date   = col2.date_input("End Date",   datetime.today() - timedelta(days=1))

initial_capital = st.sidebar.number_input("Starting Capital ($)", value=10000, step=1000)

st.sidebar.markdown("---")
st.sidebar.subheader("📐 Indicators")

use_ma     = st.sidebar.checkbox("Moving Average Crossover", value=True)
use_rsi    = st.sidebar.checkbox("RSI (Momentum)")
use_bb     = st.sidebar.checkbox("Bollinger Bands (Volatility)")
use_custom = st.sidebar.checkbox("Custom Price Rule")

# MA settings
if use_ma:
    st.sidebar.markdown("**Moving Averages**")
    short_ma = st.sidebar.slider("Short MA (days)", 5,  50,  20)
    long_ma  = st.sidebar.slider("Long MA (days)",  20, 200, 50)

# RSI settings
if use_rsi:
    st.sidebar.markdown("**RSI**")
    rsi_period    = st.sidebar.slider("RSI Period", 7, 21, 14)
    rsi_oversold  = st.sidebar.slider("Buy when RSI below", 20, 40, 30)
    rsi_overbought = st.sidebar.slider("Sell when RSI above", 60, 85, 70)

# Bollinger Band settings
if use_bb:
    st.sidebar.markdown("**Bollinger Bands**")
    bb_period = st.sidebar.slider("BB Period", 10, 30, 20)
    bb_std    = st.sidebar.slider("BB Std Dev", 1.0, 3.0, 2.0, step=0.5)

# Custom price rule
if use_custom:
    st.sidebar.markdown("**Custom Rule**")
    custom_buy_pct  = st.sidebar.slider("Buy if price drops by % in a day", 1, 10, 3)
    custom_sell_pct = st.sidebar.slider("Sell if price rises by % in a day", 1, 10, 3)

run_btn = st.sidebar.button("🚀 Run Backtest", use_container_width=True)

# ─── HELPER FUNCTIONS ───────────────────────────────────────────────────────────
def compute_rsi(series, period=14):
    delta = series.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def run_backtest(df):
    signals = pd.DataFrame(index=df.index)
    signals['price'] = df['Close']
    signals['buy']   = False
    signals['sell']  = False

    # Moving Average Crossover
    if use_ma:
        signals['short_ma'] = df['Close'].rolling(short_ma).mean()
        signals['long_ma']  = df['Close'].rolling(long_ma).mean()
        signals['buy']  |= (signals['short_ma'] > signals['long_ma']) & \
                           (signals['short_ma'].shift(1) <= signals['long_ma'].shift(1))
        signals['sell'] |= (signals['short_ma'] < signals['long_ma']) & \
                           (signals['short_ma'].shift(1) >= signals['long_ma'].shift(1))

    # RSI
    if use_rsi:
        signals['rsi']  = compute_rsi(df['Close'], rsi_period)
        signals['buy']  |= signals['rsi'] < rsi_oversold
        signals['sell'] |= signals['rsi'] > rsi_overbought

    # Bollinger Bands
    if use_bb:
        signals['bb_mid']   = df['Close'].rolling(bb_period).mean()
        signals['bb_upper'] = signals['bb_mid'] + bb_std * df['Close'].rolling(bb_period).std()
        signals['bb_lower'] = signals['bb_mid'] - bb_std * df['Close'].rolling(bb_period).std()
        signals['buy']  |= df['Close'] < signals['bb_lower']
        signals['sell'] |= df['Close'] > signals['bb_upper']

    # Custom rule
    if use_custom:
        daily_change = df['Close'].pct_change() * 100
        signals['buy']  |= daily_change < -custom_buy_pct
        signals['sell'] |= daily_change >  custom_sell_pct

    # ── Simulate trades ──
    cash, shares  = float(initial_capital), 0.0
    portfolio_val = []
    trade_log     = []

    for i, (idx, row) in enumerate(signals.iterrows()):
        price = row['price']
        if pd.isna(price):
            portfolio_val.append(cash)
            continue

        if row['buy'] and cash >= price:
            shares_to_buy = cash // price
            shares += shares_to_buy
            cost    = shares_to_buy * price
            cash   -= cost
            trade_log.append({'Date': idx, 'Action': 'BUY', 'Price': round(price,2),
                               'Shares': shares_to_buy, 'Value': round(cost,2)})

        elif row['sell'] and shares > 0:
            proceeds = shares * price
            cash    += proceeds
            trade_log.append({'Date': idx, 'Action': 'SELL', 'Price': round(price,2),
                               'Shares': shares, 'Value': round(proceeds,2)})
            shares   = 0

        portfolio_val.append(cash + shares * price)

    signals['portfolio'] = portfolio_val

    # Buy-and-hold benchmark
    shares_bh  = initial_capital / df['Close'].iloc[0]
    signals['benchmark'] = shares_bh * df['Close']

    return signals, pd.DataFrame(trade_log)

# ─── MAIN PANEL ─────────────────────────────────────────────────────────────────
if not run_btn:
    st.info("👈 Configure your strategy in the sidebar, then click **Run Backtest**.")
    st.markdown("""
    **How it works:**
    - Pick a stock ticker and date range
    - Choose one or more indicators to build your strategy
    - Hit Run Backtest to see how your strategy would have performed
    - Compare your strategy vs simply buying and holding the stock
    """)
else:
    if not (use_ma or use_rsi or use_bb or use_custom):
        st.error("Please select at least one indicator in the sidebar.")
        st.stop()

    with st.spinner(f"Fetching data for {ticker}..."):
        raw = yf.download(ticker, start=start_date, end=end_date, progress=False)

    if raw.empty:
        st.error(f"No data found for **{ticker}**. Check the ticker symbol and try again.")
        st.stop()

    # Flatten MultiIndex columns if present
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    signals, trades = run_backtest(raw)

    final_val   = signals['portfolio'].iloc[-1]
    benchmark_v = signals['benchmark'].iloc[-1]
    total_ret   = ((final_val - initial_capital) / initial_capital) * 100
    bench_ret   = ((benchmark_v - initial_capital) / initial_capital) * 100
    n_trades    = len(trades)

    # ── KPI Cards ──
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Final Portfolio Value", f"${final_val:,.0f}", f"{total_ret:+.1f}%")
    k2.metric("Buy & Hold Value",      f"${benchmark_v:,.0f}", f"{bench_ret:+.1f}%")
    k3.metric("Strategy vs B&H",       f"{total_ret - bench_ret:+.1f}%")
    k4.metric("Total Trades",          n_trades)

    st.markdown("---")

    # ── P&L Chart ──
    st.subheader("📊 Portfolio Value Over Time")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=signals.index, y=signals['portfolio'],
                             name="Your Strategy", line=dict(color="#00C896", width=2)))
    fig.add_trace(go.Scatter(x=signals.index, y=signals['benchmark'],
                             name="Buy & Hold", line=dict(color="#636EFA", width=2, dash='dash')))

    # Mark buy/sell on chart
    buys  = signals[signals['buy']]
    sells = signals[signals['sell']]
    fig.add_trace(go.Scatter(x=buys.index,  y=buys['portfolio'],
                             mode='markers', name='Buy Signal',
                             marker=dict(symbol='triangle-up', size=10, color='green')))
    fig.add_trace(go.Scatter(x=sells.index, y=sells['portfolio'],
                             mode='markers', name='Sell Signal',
                             marker=dict(symbol='triangle-down', size=10, color='red')))

    fig.update_layout(template='plotly_dark', height=500,
                      xaxis_title="Date", yaxis_title="Portfolio Value ($)",
                      legend=dict(orientation='h', yanchor='bottom', y=1.02))
    st.plotly_chart(fig, use_container_width=True)

    # ── Price + Indicators Chart ──
    st.subheader("🕯️ Stock Price & Indicators")
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=raw.index, y=raw['Close'], name="Close Price",
                              line=dict(color='white', width=1.5)))
    if use_ma and 'short_ma' in signals:
        fig2.add_trace(go.Scatter(x=signals.index, y=signals['short_ma'],
                                  name=f"MA {short_ma}", line=dict(color='orange', width=1.2)))
        fig2.add_trace(go.Scatter(x=signals.index, y=signals['long_ma'],
                                  name=f"MA {long_ma}", line=dict(color='cyan', width=1.2)))
    if use_bb and 'bb_upper' in signals:
        fig2.add_trace(go.Scatter(x=signals.index, y=signals['bb_upper'],
                                  name="BB Upper", line=dict(color='purple', dash='dot')))
        fig2.add_trace(go.Scatter(x=signals.index, y=signals['bb_lower'],
                                  name="BB Lower", line=dict(color='purple', dash='dot')))
    fig2.update_layout(template='plotly_dark', height=400,
                       xaxis_title="Date", yaxis_title="Price ($)")
    st.plotly_chart(fig2, use_container_width=True)

    # ── RSI Chart ──
    if use_rsi and 'rsi' in signals:
        st.subheader("📉 RSI Indicator")
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=signals.index, y=signals['rsi'],
                                  name="RSI", line=dict(color='yellow')))
        fig3.add_hline(y=rsi_overbought, line_dash="dash", line_color="red",   annotation_text="Overbought")
        fig3.add_hline(y=rsi_oversold,   line_dash="dash", line_color="green", annotation_text="Oversold")
        fig3.update_layout(template='plotly_dark', height=250,
                           yaxis=dict(range=[0,100]), xaxis_title="Date", yaxis_title="RSI")
        st.plotly_chart(fig3, use_container_width=True)

    # ── Trade Log ──
    if not trades.empty:
        st.subheader("📋 Trade Log")
        trades['Date'] = pd.to_datetime(trades['Date']).dt.strftime('%Y-%m-%d')
        st.dataframe(trades.style.applymap(
            lambda v: 'color: lightgreen' if v == 'BUY' else ('color: salmon' if v == 'SELL' else ''),
            subset=['Action']), use_container_width=True)
    else:
        st.warning("No trades were triggered. Try adjusting your strategy parameters.")
