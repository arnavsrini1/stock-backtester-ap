import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import yfinance as yf
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

# ─── PAGE CONFIG ────────────────────────────────────────────────────────────
st.set_page_config(page_title="AI Trading Signal Generator", layout="wide")
st.title("🤖 AI Trading Signal Generator")
st.markdown("Train a real AI model on historical stock data and generate Buy/Sell signals.")

# ─── SIDEBAR ────────────────────────────────────────────────────────────────
st.sidebar.header("⚙️ Configuration")

# Stock & Date
ticker = st.sidebar.text_input("Stock Ticker (e.g. AAPL, TSLA, NVDA)", value="AAPL").upper()
c1, c2 = st.sidebar.columns(2)
start_date = c1.date_input("Start Date", datetime.today() - timedelta(days=365*4))
end_date   = c2.date_input("End Date",   datetime.today() - timedelta(days=1))
initial_capital = st.sidebar.number_input("Starting Capital ($)", value=10000, step=1000)

st.sidebar.markdown("---")

# ─── AI MODEL SELECTION ─────────────────────────────────────────────────────
st.sidebar.subheader("🧠 Choose Your AI Model")
model_choice = st.sidebar.selectbox("AI Model", [
    "Random Forest",
    "Logistic Regression",
    "Gradient Boosting"
])

st.sidebar.markdown("---")
st.sidebar.subheader("🎛️ Model Settings")

# Model-specific tuning
if model_choice == "Random Forest":
    n_estimators  = st.sidebar.slider("Number of Trees", 10, 300, 100,
                                       help="More trees = more accurate but slower")
    max_depth     = st.sidebar.slider("Max Tree Depth", 2, 20, 5,
                                       help="Deeper = more complex patterns learned")
    min_samples   = st.sidebar.slider("Min Samples to Split", 2, 20, 5,
                                       help="Higher = less overfitting")

elif model_choice == "Logistic Regression":
    C_val         = st.sidebar.slider("Regularisation (C)", 0.01, 10.0, 1.0, step=0.01,
                                       help="Lower = simpler model, Higher = fits data more closely")
    max_iter      = st.sidebar.slider("Max Iterations", 100, 1000, 200,
                                       help="How long the model trains")

elif model_choice == "Gradient Boosting":
    n_estimators  = st.sidebar.slider("Number of Boosting Rounds", 10, 300, 100,
                                       help="More rounds = stronger model")
    learning_rate = st.sidebar.slider("Learning Rate", 0.01, 0.5, 0.1, step=0.01,
                                       help="How fast the model learns — lower is more careful")
    max_depth     = st.sidebar.slider("Max Tree Depth", 2, 10, 3,
                                       help="Complexity of each round")

st.sidebar.markdown("---")
st.sidebar.subheader("📐 Feature Engineering")
st.sidebar.markdown("*Select which signals to feed the AI:*")
use_ma     = st.sidebar.checkbox("Moving Averages",     value=True)
use_rsi    = st.sidebar.checkbox("RSI Momentum",        value=True)
use_bb     = st.sidebar.checkbox("Bollinger Bands",     value=True)
use_volume = st.sidebar.checkbox("Volume Signals",      value=True)
use_macd   = st.sidebar.checkbox("MACD",                value=True)

st.sidebar.markdown("---")
st.sidebar.subheader("🎯 Signal Settings")
lookahead  = st.sidebar.slider("Lookahead Days (prediction window)", 1, 10, 3,
                                help="AI predicts if stock goes up over this many days")
threshold  = st.sidebar.slider("Buy Signal Threshold (%)", 0.5, 5.0, 1.0, step=0.1,
                                help="Minimum % gain required to trigger a Buy signal")
test_split = st.sidebar.slider("Train/Test Split (%)", 60, 90, 80,
                                help="% of data used for training vs testing")

run_btn = st.sidebar.button("🚀 Train AI & Generate Signals", use_container_width=True)

# ─── FEATURE BUILDER ────────────────────────────────────────────────────────
def build_features(df):
    f = pd.DataFrame(index=df.index)
    c = df['Close']

    if use_ma:
        f['ma5']       = c.rolling(5).mean()
        f['ma20']      = c.rolling(20).mean()
        f['ma50']      = c.rolling(50).mean()
        f['ma_ratio']  = f['ma5'] / f['ma20']
        f['ma_cross']  = (f['ma5'] > f['ma20']).astype(int)

    if use_rsi:
        delta = c.diff()
        gain  = delta.clip(lower=0).rolling(14).mean()
        loss  = (-delta.clip(upper=0)).rolling(14).mean()
        f['rsi'] = 100 - (100 / (1 + gain / loss.replace(0, np.nan)))
        f['rsi_signal'] = (f['rsi'] < 30).astype(int)

    if use_bb:
        mid         = c.rolling(20).mean()
        std         = c.rolling(20).std()
        f['bb_pos'] = (c - (mid - 2*std)) / (4*std).replace(0, np.nan)

    if use_volume:
        f['vol_ratio'] = df['Volume'] / df['Volume'].rolling(20).mean()

    if use_macd:
        ema12        = c.ewm(span=12).mean()
        ema26        = c.ewm(span=26).mean()
        macd         = ema12 - ema26
        signal       = macd.ewm(span=9).mean()
        f['macd']    = macd
        f['macd_sig']= signal
        f['macd_hist']= macd - signal

    # Price-based features always included
    f['returns']   = c.pct_change()
    f['volatility']= f['returns'].rolling(10).std()
    f['momentum']  = c / c.shift(10) - 1
    f['high_low']  = (df['High'] - df['Low']) / c

    return f

# ─── MAIN ───────────────────────────────────────────────────────────────────
if not run_btn:
    st.info("👈 Configure your AI model in the sidebar, then click **Train AI & Generate Signals**.")
    col1, col2, col3 = st.columns(3)
    col1.metric("Models Available", "3")
    col2.metric("Features Available", "15+")
    col3.metric("Markets", "US Stocks")
    st.markdown("""
    ### How it works
    1. **Pick a stock** and date range
    2. **Choose an AI model** — Random Forest, Logistic Regression, or Gradient Boosting
    3. **Tune the settings** — adjust model parameters and which signals to feed it
    4. **Train & Test** — the AI learns on historical data and predicts Buy/Sell signals
    5. **See the results** — chart, performance vs buy-and-hold, and full trade log
    """)
else:
    if not any([use_ma, use_rsi, use_bb, use_volume, use_macd]):
        st.error("Please select at least one feature in the sidebar.")
        st.stop()

    with st.spinner(f"Downloading {ticker} data..."):
        raw = yf.download(ticker, start=start_date, end=end_date, progress=False)

    if raw.empty:
        st.error(f"No data found for **{ticker}**. Check the ticker and try again.")
        st.stop()

    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    # Build features
    feats = build_features(raw)

    # Build target: 1 = Buy (price up > threshold% in lookahead days), 0 = Sell/Hold
    future_ret = raw['Close'].shift(-lookahead) / raw['Close'] - 1
    target = (future_ret > threshold / 100).astype(int)

    # Combine and clean
    data = feats.copy()
    data['target'] = target
    data = data.dropna()

    if len(data) < 100:
        st.error("Not enough data to train. Try a longer date range.")
        st.stop()

    X = data.drop('target', axis=1)
    y = data['target']

    split = int(len(X) * test_split / 100)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    # Scale
    scaler  = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    # Train model
    with st.spinner(f"Training {model_choice} AI model..."):
        if model_choice == "Random Forest":
            model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                           min_samples_split=min_samples, random_state=42, n_jobs=-1)
        elif model_choice == "Logistic Regression":
            model = LogisticRegression(C=C_val, max_iter=max_iter, random_state=42)
        else:
            model = GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=learning_rate,
                                               max_depth=max_depth, random_state=42)
        model.fit(X_train_s, y_train)

    # Predict
    preds      = model.predict(X_test_s)
    proba      = model.predict_proba(X_test_s)[:, 1]
    accuracy   = accuracy_score(y_test, preds)

    # ── KPI Row 1 ──
    st.subheader("🎯 Model Performance")
    k1, k2, k3, k4 = st.columns(4)
    buy_signals  = preds.sum()
    sell_signals = (preds == 0).sum()
    k1.metric("Model Accuracy", f"{accuracy*100:.1f}%", help="% of signals correctly predicted on test data")
    k2.metric("Buy Signals",  int(buy_signals))
    k3.metric("Sell Signals", int(sell_signals))
    k4.metric("Test Period Rows", len(X_test))

    st.markdown("---")

    # ── Simulate trading on test set ──
    test_prices = raw['Close'].loc[X_test.index]
    cash, shares = float(initial_capital), 0.0
    portfolio, trade_log = [], []
    prev_signal = -1

    for i, (idx, price) in enumerate(test_prices.items()):
        sig = preds[i]
        confidence = proba[i]

        if sig == 1 and prev_signal != 1 and cash >= price:
            shares_to_buy = cash // price
            if shares_to_buy > 0:
                shares  += shares_to_buy
                cash    -= shares_to_buy * price
                trade_log.append({'Date': idx, 'Action': 'BUY', 'Price': round(float(price),2),
                                   'Shares': int(shares_to_buy), 'Confidence': f"{confidence*100:.1f}%"})

        elif sig == 0 and prev_signal != 0 and shares > 0:
            proceeds = shares * price
            cash    += proceeds
            trade_log.append({'Date': idx, 'Action': 'SELL', 'Price': round(float(price),2),
                               'Shares': int(shares), 'Confidence': f"{(1-confidence)*100:.1f}%"})
            shares   = 0

        portfolio.append(cash + shares * float(price))
        prev_signal = sig

    final_val  = portfolio[-1] if portfolio else initial_capital
    bh_val     = initial_capital / float(test_prices.iloc[0]) * float(test_prices.iloc[-1])
    strat_ret  = (final_val - initial_capital) / initial_capital * 100
    bh_ret     = (bh_val - initial_capital) / initial_capital * 100

    # ── KPI Row 2 ──
    st.subheader("💰 Trading Performance (Test Period)")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Final Portfolio", f"${final_val:,.0f}", f"{strat_ret:+.1f}%")
    m2.metric("Buy & Hold Value", f"${bh_val:,.0f}", f"{bh_ret:+.1f}%")
    m3.metric("AI vs Buy & Hold", f"{strat_ret - bh_ret:+.1f}%")
    m4.metric("Total Trades", len(trade_log))

    st.markdown("---")

    # ── Portfolio Chart ──
    st.subheader("📈 AI Portfolio vs Buy & Hold")
    bh_series = initial_capital / float(test_prices.iloc[0]) * test_prices

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=test_prices.index, y=portfolio,
                             name="AI Strategy", line=dict(color="#00C896", width=2)))
    fig.add_trace(go.Scatter(x=test_prices.index, y=bh_series,
                             name="Buy & Hold", line=dict(color="#636EFA", width=2, dash='dash')))

    trades_df = pd.DataFrame(trade_log)
    if not trades_df.empty:
        buys  = trades_df[trades_df['Action'] == 'BUY']
        sells = trades_df[trades_df['Action'] == 'SELL']
        if not buys.empty:
            buy_port = [portfolio[list(test_prices.index).index(d)] for d in buys['Date'] if d in list(test_prices.index)]
            fig.add_trace(go.Scatter(x=buys['Date'], y=buy_port, mode='markers', name='Buy',
                                     marker=dict(symbol='triangle-up', size=12, color='lime')))
        if not sells.empty:
            sell_port = [portfolio[list(test_prices.index).index(d)] for d in sells['Date'] if d in list(test_prices.index)]
            fig.add_trace(go.Scatter(x=sells['Date'], y=sell_port, mode='markers', name='Sell',
                                     marker=dict(symbol='triangle-down', size=12, color='red')))

    fig.update_layout(template='plotly_dark', height=450,
                      xaxis_title="Date", yaxis_title="Portfolio Value ($)",
                      legend=dict(orientation='h', yanchor='bottom', y=1.02))
    st.plotly_chart(fig, use_container_width=True)

    # ── Signal Confidence Chart ──
    st.subheader("🔬 AI Signal Confidence Over Time")
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=X_test.index, y=proba, name="Buy Probability",
                              line=dict(color='orange', width=1.5), fill='tozeroy',
                              fillcolor='rgba(255,165,0,0.1)'))
    fig2.add_hline(y=0.5, line_dash="dash", line_color="white", annotation_text="50% threshold")
    fig2.update_layout(template='plotly_dark', height=250,
                       yaxis=dict(range=[0,1], tickformat='.0%'),
                       xaxis_title="Date", yaxis_title="Buy Signal Confidence")
    st.plotly_chart(fig2, use_container_width=True)

    # ── Feature Importance ──
    if model_choice in ["Random Forest", "Gradient Boosting"]:
        st.subheader("🧩 Feature Importance (What the AI Learned)")
        importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=True).tail(10)
        fig3 = go.Figure(go.Bar(x=importances.values, y=importances.index,
                                orientation='h', marker_color='#00C896'))
        fig3.update_layout(template='plotly_dark', height=350,
                           xaxis_title="Importance", yaxis_title="Feature")
        st.plotly_chart(fig3, use_container_width=True)

    # ── Trade Log ──
    if not trades_df.empty:
        st.subheader("📋 Trade Log")
        st.dataframe(trades_df.style.applymap(
            lambda v: 'color: lightgreen' if v == 'BUY' else ('color: salmon' if v == 'SELL' else ''),
            subset=['Action']), use_container_width=True)
    else:
        st.warning("No trades triggered. Try adjusting the threshold or lookahead days.")
