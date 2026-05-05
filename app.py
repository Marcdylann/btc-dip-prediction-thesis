"""
BTC Dip Prediction Dashboard
University of San Carlos | BS Computer Science | Barria & Pilapil · 2025
Run: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import joblib
from pathlib import Path
import lightgbm as lgb
import tensorflow as tf
from datetime import datetime

from src.features import FEATURE_COLS, add_btc_features, add_eth_features, add_daily_features

MODELS_DIR = Path("models")

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="BTC Dip Predictor",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Design tokens ──────────────────────────────────────────────────────────────
BTC_CLR  = "#F7931A"
ETH_CLR  = "#627EEA"
DIP_CLR  = "#FF4B4B"
SAFE_CLR = "#00C896"
WARN_CLR = "#FFB830"
BG       = "#0e1117"
CARD_BG  = "#161b22"
BORDER   = "#30363d"
TEXT_MUT = "#8b949e"

CHART = dict(
    template="plotly_dark",
    paper_bgcolor=BG,
    plot_bgcolor=BG,
    font=dict(family="Inter, system-ui, sans-serif", size=13, color="#c9d1d9"),
    margin=dict(t=40, b=40, l=10, r=10),
    hovermode="x unified",
    hoverlabel=dict(bgcolor=CARD_BG, bordercolor=BORDER, font_size=12),
    xaxis_gridcolor=BORDER,
    xaxis_zeroline=False,
    xaxis_showgrid=True,
    yaxis_gridcolor=BORDER,
    yaxis_zeroline=False,
    yaxis_showgrid=True,
    legend_bgcolor="rgba(0,0,0,0)",
    legend_bordercolor=BORDER,
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {{
    font-family: 'Inter', system-ui, sans-serif;
}}

/* Hide Streamlit chrome */
#MainMenu, footer, header {{ visibility: hidden; }}
.block-container {{ padding-top: 1.5rem; padding-bottom: 1rem; }}

/* Sidebar */
[data-testid="stSidebar"] {{
    background: #0d1117;
    border-right: 1px solid {BORDER};
}}
[data-testid="stSidebar"] .stRadio label {{
    font-size: 14px;
    font-weight: 500;
    padding: 6px 0;
}}

/* Metric cards */
.kpi-grid {{
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 16px;
    margin-bottom: 24px;
}}
.kpi-card {{
    background: {CARD_BG};
    border: 1px solid {BORDER};
    border-radius: 12px;
    padding: 20px 24px;
    position: relative;
    overflow: hidden;
}}
.kpi-card::before {{
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: var(--accent, {BTC_CLR});
    border-radius: 12px 12px 0 0;
}}
.kpi-label {{
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 1.2px;
    text-transform: uppercase;
    color: {TEXT_MUT};
    margin-bottom: 8px;
}}
.kpi-value {{
    font-size: 28px;
    font-weight: 700;
    color: #f0f6fc;
    line-height: 1.1;
}}
.kpi-sub {{
    font-size: 12px;
    color: {TEXT_MUT};
    margin-top: 6px;
}}

/* Alert banners */
.alert-dip {{
    background: rgba(255,75,75,0.12);
    border: 1px solid {DIP_CLR};
    border-left: 4px solid {DIP_CLR};
    border-radius: 8px;
    padding: 16px 20px;
    margin: 16px 0;
    display: flex;
    align-items: center;
    gap: 12px;
}}
.alert-safe {{
    background: rgba(0,200,150,0.10);
    border: 1px solid {SAFE_CLR};
    border-left: 4px solid {SAFE_CLR};
    border-radius: 8px;
    padding: 16px 20px;
    margin: 16px 0;
    display: flex;
    align-items: center;
    gap: 12px;
}}
.alert-title {{
    font-size: 15px;
    font-weight: 700;
    margin-bottom: 2px;
}}
.alert-body {{
    font-size: 13px;
    color: {TEXT_MUT};
}}

/* Live badge */
.live-badge {{
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(255,75,75,0.15);
    border: 1px solid {DIP_CLR};
    color: {DIP_CLR};
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 1.5px;
    padding: 4px 10px;
    border-radius: 20px;
    vertical-align: middle;
    margin-left: 10px;
}}
.live-dot {{
    width: 7px; height: 7px;
    background: {DIP_CLR};
    border-radius: 50%;
    animation: blink 1.4s infinite;
}}
@keyframes blink {{
    0%, 100% {{ opacity: 1; }}
    50% {{ opacity: 0.2; }}
}}

/* Section divider */
.section-title {{
    font-size: 15px;
    font-weight: 600;
    color: #c9d1d9;
    margin: 28px 0 14px 0;
    padding-bottom: 10px;
    border-bottom: 1px solid {BORDER};
}}

/* Stat chips */
.stat-row {{
    display: flex;
    gap: 12px;
    margin-bottom: 20px;
    flex-wrap: wrap;
}}
.stat-chip {{
    background: {CARD_BG};
    border: 1px solid {BORDER};
    border-radius: 8px;
    padding: 10px 16px;
    font-size: 13px;
}}
.stat-chip b {{ color: #f0f6fc; }}

/* Page title */
.page-title {{
    font-size: 24px;
    font-weight: 700;
    color: #f0f6fc;
    margin-bottom: 4px;
}}
.page-subtitle {{
    font-size: 13px;
    color: {TEXT_MUT};
    margin-bottom: 24px;
}}

/* Data table */
.stDataFrame {{ border-radius: 8px; overflow: hidden; }}
</style>
""", unsafe_allow_html=True)


# ── Model loading ──────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    lgb_model  = lgb.Booster(model_file=str(MODELS_DIR / "lgb_model.txt"))
    lstm_model = tf.keras.models.load_model(str(MODELS_DIR / "lstm_model.keras"))
    meta_clf   = joblib.load(MODELS_DIR / "meta_clf.pkl")
    scaler     = joblib.load(MODELS_DIR / "scaler.pkl")
    threshold  = joblib.load(MODELS_DIR / "threshold.pkl")
    metrics    = joblib.load(MODELS_DIR / "metrics.pkl")
    return lgb_model, lstm_model, meta_clf, scaler, threshold, metrics


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_recent_data():
    from src.data_pipeline import fetch_ohlcv_fast, load_daily_data
    btc = fetch_ohlcv_fast("BTC-USD", days=60)
    eth = fetch_ohlcv_fast("ETH-USD", days=60)
    btc_daily, eth_daily = load_daily_data()
    return btc, eth, btc_daily, eth_daily


def meta_features(lgb_p, lstm_p):
    return np.column_stack([
        lgb_p, lstm_p, lgb_p * lstm_p,
        np.maximum(lgb_p, lstm_p), np.abs(lgb_p - lstm_p),
    ])


def run_prediction(btc, eth, btc_daily, eth_daily, lgb_model, lstm_model, meta_clf, scaler):
    df = btc.copy()
    df = add_btc_features(df)
    df = add_eth_features(df, eth)
    df = add_daily_features(df, btc_daily, eth_daily)
    df = df.replace([float("inf"), float("-inf")], float("nan"))
    df = df.dropna(subset=FEATURE_COLS)
    if len(df) < 24:
        return None, None, None

    X    = df[FEATURE_COLS].values
    X_sc = scaler.transform(X)
    lgb_prob  = lgb_model.predict(X_sc)
    seq       = X_sc[-24:].reshape(1, 24, X_sc.shape[1])
    lstm_prob = float(lstm_model.predict(seq, verbose=0).flatten()[0])
    X_meta    = meta_features(np.array([lgb_prob[-1]]), np.array([lstm_prob]))
    prob      = meta_clf.predict_proba(X_meta)[0, 1]
    return prob, lgb_prob, df


def kpi(label, value, sub="", accent=BTC_CLR):
    return f"""
    <div class="kpi-card" style="--accent:{accent}">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        {"<div class='kpi-sub'>" + sub + "</div>" if sub else ""}
    </div>"""


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""
    <div style="display:flex;align-items:center;gap:12px;margin-bottom:4px">
        <span style="font-size:28px">📉</span>
        <div>
            <div style="font-size:17px;font-weight:700;color:#f0f6fc">BTC Dip Predictor</div>
            <div style="font-size:11px;color:{TEXT_MUT}">Ensemble ML · Early Warning</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown(f"<hr style='border-color:{BORDER};margin:16px 0'>", unsafe_allow_html=True)

    page = st.radio(
        "Navigation",
        ["Live Prediction", "Prediction History", "Model Performance", "Feature Importance"],
        label_visibility="collapsed",
    )

    st.markdown(f"<hr style='border-color:{BORDER};margin:16px 0'>", unsafe_allow_html=True)
    st.markdown(f"""
    <div style="font-size:12px;color:{TEXT_MUT};line-height:1.8">
        <b style="color:#c9d1d9">Model Stack</b><br>
        LightGBM + Bi-LSTM<br>
        Logistic Regression (meta)<br><br>
        <b style="color:#c9d1d9">Data</b><br>
        Hourly: Binance (2 yr)<br>
        Daily context: 2017–now<br>
        BTC-USD · ETH-USD<br><br>
        <b style="color:#c9d1d9">University of San Carlos</b><br>
        BS Computer Science<br>
        Barria · Pilapil · 2025
    </div>
    """, unsafe_allow_html=True)

# ── Guard ──────────────────────────────────────────────────────────────────────
if not all((MODELS_DIR / f).exists() for f in ["lgb_model.txt", "lstm_model.keras", "meta_clf.pkl"]):
    st.error("Models not found. Run `python train.py` first.")
    st.stop()

lgb_model, lstm_model, meta_clf, scaler, threshold, metrics = load_models()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — Live Prediction
# ══════════════════════════════════════════════════════════════════════════════
if page == "Live Prediction":
    st.markdown(f"""
    <div class="page-title">
        Live Bitcoin Dip Prediction
        <span class="live-badge"><span class="live-dot"></span>LIVE</span>
    </div>
    <div class="page-subtitle">
        Detects whether BTC-USD will drop &ge;2% within the next 6 hours using an
        ensemble of LightGBM + Bi-LSTM + Logistic Regression.
    </div>
    """, unsafe_allow_html=True)

    with st.spinner("Fetching market data..."):
        btc, eth, btc_daily, eth_daily = fetch_recent_data()
        prob, lgb_prob, feat_df = run_prediction(
            btc, eth, btc_daily, eth_daily, lgb_model, lstm_model, meta_clf, scaler
        )

    if prob is None:
        st.warning("Not enough data to generate a prediction.")
        st.stop()

    is_dip       = prob >= threshold
    close        = float(btc["close"].iloc[-1])
    prev_close   = float(btc["close"].iloc[-2])
    change_1h    = (close - prev_close) / prev_close
    close_24h    = float(btc["close"].iloc[-25]) if len(btc) > 25 else prev_close
    change_24h   = (close - close_24h) / close_24h
    eth_close    = float(eth["close"].iloc[-1])
    updated_at   = btc.index[-1].strftime("%Y-%m-%d %H:%M UTC")

    # ── KPI cards ──
    signal_color = DIP_CLR if is_dip else SAFE_CLR
    prob_color   = DIP_CLR if prob > 0.5 else (WARN_CLR if prob > 0.25 else SAFE_CLR)
    chg1h_color  = DIP_CLR if change_1h < 0 else SAFE_CLR
    chg24h_color = DIP_CLR if change_24h < 0 else SAFE_CLR

    st.markdown(f"""
    <div class="kpi-grid">
        {kpi("BTC Price", f"${close:,.2f}", f"1h: {change_1h:+.2%}", accent=BTC_CLR)}
        {kpi("24h Change", f"{change_24h:+.2%}", f"ETH: ${eth_close:,.2f}", accent=chg24h_color)}
        {kpi("Dip Probability", f"{prob:.1%}", f"Threshold: {threshold:.0%}", accent=prob_color)}
        {kpi("Signal", "⚠ DIP ALERT" if is_dip else "✓ NO DIP", updated_at, accent=signal_color)}
    </div>
    """, unsafe_allow_html=True)

    # ── Alert banner ──
    if is_dip:
        st.markdown(f"""
        <div class="alert-dip">
            <span style="font-size:28px">⚠️</span>
            <div>
                <div class="alert-title" style="color:{DIP_CLR}">DIP ALERT — High Probability Event</div>
                <div class="alert-body">
                    Model confidence: <b style="color:{DIP_CLR}">{prob:.1%}</b> &nbsp;|&nbsp;
                    A drop of &ge;2% is forecast within the next <b>6 hours</b>.
                    Consider reviewing your position or setting stop-loss orders.
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="alert-safe">
            <span style="font-size:28px">✅</span>
            <div>
                <div class="alert-title" style="color:{SAFE_CLR}">No Dip Expected</div>
                <div class="alert-body">
                    Probability <b style="color:{SAFE_CLR}">{prob:.1%}</b> is below the
                    <b>{threshold:.0%}</b> alert threshold.
                    Market conditions appear stable for the next 6 hours.
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Gauge + BTC/ETH charts side by side ──
    col_g, col_p = st.columns([1, 2])

    with col_g:
        st.markdown('<div class="section-title">Dip Probability Gauge</div>', unsafe_allow_html=True)
        needle_color = DIP_CLR if is_dip else SAFE_CLR
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=round(prob * 100, 1),
            number={"suffix": "%", "font": {"size": 36, "color": needle_color}},
            delta={"reference": threshold * 100, "valueformat": ".1f",
                   "increasing": {"color": DIP_CLR}, "decreasing": {"color": SAFE_CLR}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": BORDER,
                         "tickfont": {"color": TEXT_MUT}},
                "bar": {"color": needle_color, "thickness": 0.25},
                "bgcolor": CARD_BG,
                "borderwidth": 0,
                "steps": [
                    {"range": [0,  threshold * 100], "color": "rgba(0,200,150,0.15)"},
                    {"range": [threshold * 100, 100], "color": "rgba(255,75,75,0.15)"},
                ],
                "threshold": {
                    "line": {"color": WARN_CLR, "width": 3},
                    "thickness": 0.85,
                    "value": threshold * 100,
                },
            },
        ))
        fig_gauge.update_layout(
            height=280,
            paper_bgcolor=BG,
            font=dict(color="#c9d1d9"),
            margin=dict(t=20, b=10, l=20, r=20),
        )
        st.plotly_chart(fig_gauge, use_container_width=True)
        st.markdown(
            f"<div style='text-align:center;font-size:12px;color:{TEXT_MUT}'>"
            f"Alert threshold: <b style='color:{WARN_CLR}'>{threshold:.0%}</b></div>",
            unsafe_allow_html=True,
        )

    with col_p:
        st.markdown('<div class="section-title">BTC Price + Volume — Last 7 Days</div>', unsafe_allow_html=True)
        recent = btc.tail(7 * 24).copy()
        recent_close = recent["close"].astype(float)
        recent_vol   = recent["volume"].astype(float)

        fig_price = make_subplots(
            rows=2, cols=1, shared_xaxes=True,
            row_heights=[0.75, 0.25], vertical_spacing=0.03,
        )
        fig_price.add_trace(go.Scatter(
            x=recent.index, y=recent_close,
            name="BTC Close", mode="lines",
            line=dict(color=BTC_CLR, width=2),
            fill="tozeroy",
            fillcolor="rgba(247,147,26,0.08)",
        ), row=1, col=1)
        fig_price.add_trace(go.Bar(
            x=recent.index, y=recent_vol,
            name="Volume", marker_color="rgba(247,147,26,0.4)",
            showlegend=False,
        ), row=2, col=1)

        fig_price.update_layout(
            **CHART, height=280,
            legend=dict(orientation="h", y=1.05, x=0),
        )
        fig_price.update_yaxes(title_text="USD", row=1, col=1, gridcolor=BORDER)
        fig_price.update_yaxes(title_text="Vol",  row=2, col=1, gridcolor=BORDER)
        fig_price.update_xaxes(gridcolor=BORDER, row=2, col=1)
        st.plotly_chart(fig_price, use_container_width=True)

    # ── BTC vs ETH returns ──
    st.markdown('<div class="section-title">BTC vs ETH — Normalized 7-Day Returns</div>', unsafe_allow_html=True)
    r7 = 7 * 24
    btc7 = btc["close"].iloc[-r7:].astype(float)
    eth7 = eth["close"].iloc[-r7:].astype(float)
    btc7_norm = (btc7 / btc7.iloc[0] - 1) * 100
    eth7_norm = (eth7 / eth7.iloc[0] - 1) * 100

    fig_comp = go.Figure()
    fig_comp.add_trace(go.Scatter(x=btc7.index, y=btc7_norm, name="BTC",
                                   line=dict(color=BTC_CLR, width=2)))
    fig_comp.add_trace(go.Scatter(x=eth7.index, y=eth7_norm, name="ETH",
                                   line=dict(color=ETH_CLR, width=2)))
    fig_comp.add_hline(y=0, line_dash="dot", line_color=BORDER)
    fig_comp.update_layout(**CHART, height=220,
                            yaxis_title="Return (%)",
                            legend=dict(orientation="h", y=1.1))
    st.plotly_chart(fig_comp, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — Prediction History
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Prediction History":
    st.markdown('<div class="page-title">Prediction History</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Out-of-sample test set results — chronological hold-out, not seen during training.</div>', unsafe_allow_html=True)

    hist_path = MODELS_DIR / "pred_history.csv"
    if not hist_path.exists():
        st.warning("pred_history.csv not found. Run train.py first.")
        st.stop()

    hist = pd.read_csv(hist_path, parse_dates=["timestamp"])

    total       = len(hist)
    n_dip_pred  = int(hist["prediction"].sum())
    n_dip_actual = int(hist["actual"].sum())
    tp = int(((hist["prediction"] == 1) & (hist["actual"] == 1)).sum())
    recall_disp = tp / n_dip_actual if n_dip_actual else 0

    st.markdown(f"""
    <div class="stat-row">
        <div class="stat-chip">Total predictions: <b>{total:,}</b></div>
        <div class="stat-chip">Actual dips: <b>{n_dip_actual}</b> ({n_dip_actual/total:.1%})</div>
        <div class="stat-chip">Predicted dips: <b>{n_dip_pred}</b></div>
        <div class="stat-chip">True positives: <b>{tp}</b></div>
        <div class="stat-chip">Recall: <b>{recall_disp:.1%}</b></div>
    </div>
    """, unsafe_allow_html=True)

    # ── Main chart: price + probability + dip markers ──
    st.markdown('<div class="section-title">Dip Probability vs BTC Price</div>', unsafe_allow_html=True)

    actual_dips  = hist[hist["actual"] == 1]
    true_pos     = hist[(hist["actual"] == 1) & (hist["prediction"] == 1)]
    false_neg    = hist[(hist["actual"] == 1) & (hist["prediction"] == 0)]

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(go.Scatter(
        x=hist["timestamp"], y=hist["btc_close"],
        name="BTC Close", mode="lines",
        line=dict(color=BTC_CLR, width=1.5),
        fill="tozeroy", fillcolor="rgba(247,147,26,0.05)",
    ), secondary_y=False)

    fig.add_trace(go.Scatter(
        x=hist["timestamp"], y=hist["probability"],
        name="Dip Probability", mode="lines",
        line=dict(color=DIP_CLR, width=1.2, dash="dot"),
        opacity=0.85,
    ), secondary_y=True)

    fig.add_trace(go.Scatter(
        x=true_pos["timestamp"], y=true_pos["btc_close"],
        mode="markers", name="Caught Dip",
        marker=dict(color=SAFE_CLR, size=8, symbol="triangle-down",
                    line=dict(color="white", width=1)),
    ), secondary_y=False)

    fig.add_trace(go.Scatter(
        x=false_neg["timestamp"], y=false_neg["btc_close"],
        mode="markers", name="Missed Dip",
        marker=dict(color=DIP_CLR, size=8, symbol="x",
                    line=dict(color="white", width=1)),
    ), secondary_y=False)

    fig.add_hline(y=threshold, line_dash="dash", line_color=WARN_CLR,
                  annotation_text=f"Threshold {threshold:.0%}",
                  annotation_font_color=WARN_CLR,
                  secondary_y=True)

    fig.update_layout(
        **CHART, height=420,
        legend=dict(orientation="h", y=1.08, x=0),
    )
    fig.update_yaxes(title_text="BTC Price (USD)", secondary_y=False, gridcolor=BORDER)
    fig.update_yaxes(title_text="Dip Probability", secondary_y=True,
                     range=[0, 1], gridcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, use_container_width=True)

    # ── Table ──
    st.markdown('<div class="section-title">Prediction Log</div>', unsafe_allow_html=True)

    display = hist.copy().sort_values("timestamp", ascending=False).reset_index(drop=True)
    display["btc_close"]   = display["btc_close"].map("${:,.2f}".format)
    display["probability"] = display["probability"].map("{:.1%}".format)
    display["prediction"]  = display["prediction"].map({1: "⚠ DIP", 0: "NO DIP"})
    display["actual"]      = display["actual"].map({1: "DIP", 0: "NO DIP"})
    display.columns        = ["Timestamp", "BTC Close", "Probability", "Prediction", "Actual"]

    st.dataframe(
        display,
        height=380,
        use_container_width=True,
        hide_index=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — Model Performance
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Model Performance":
    st.markdown('<div class="page-title">Model Performance</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Evaluated on a chronological hold-out test set not seen during training.</div>', unsafe_allow_html=True)

    r = metrics["recall"]
    p = metrics["precision"]
    f = metrics["f2"]
    a = metrics["auprc"]

    def met_color(val, target):
        return SAFE_CLR if val >= target else (WARN_CLR if val >= target * 0.8 else DIP_CLR)

    st.markdown(f"""
    <div class="kpi-grid">
        {kpi("Recall", f"{r:.1%}", "Target ≥ 80%", accent=met_color(r, 0.80))}
        {kpi("Precision", f"{p:.1%}", "Target ≥ 40%", accent=met_color(p, 0.40))}
        {kpi("F₂ Score", f"{f:.3f}", "Target ≥ 0.65", accent=met_color(f, 0.65))}
        {kpi("AUPRC", f"{a:.3f}", "Target ≥ 0.50", accent=met_color(a, 0.50))}
    </div>
    """, unsafe_allow_html=True)

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown('<div class="section-title">Confusion Matrix</div>', unsafe_allow_html=True)
        cm = np.array(metrics["confusion_matrix"])
        tn, fp, fn, tp_ = cm[0,0], cm[0,1], cm[1,0], cm[1,1]
        z    = [[tn, fp], [fn, tp_]]
        text = [[f"TN\n{tn:,}", f"FP\n{fp:,}"], [f"FN\n{fn:,}", f"TP\n{tp_:,}"]]

        fig_cm = go.Figure(go.Heatmap(
            z=z,
            text=text,
            texttemplate="%{text}",
            textfont=dict(size=16, color="white"),
            colorscale=[[0, "#161b22"], [0.5, "#1f4788"], [1.0, BTC_CLR]],
            showscale=False,
            xgap=4, ygap=4,
        ))
        fig_cm.update_layout(**CHART, height=300)
        fig_cm.update_xaxes(tickvals=[0,1], ticktext=["Predicted: No Dip","Predicted: Dip"],
                            gridcolor="rgba(0,0,0,0)", showgrid=False)
        fig_cm.update_yaxes(tickvals=[0,1], ticktext=["Actual: No Dip","Actual: Dip"],
                            gridcolor="rgba(0,0,0,0)", showgrid=False, autorange="reversed")
        st.plotly_chart(fig_cm, use_container_width=True)

    with col_b:
        st.markdown('<div class="section-title">Target vs Achieved</div>', unsafe_allow_html=True)
        metrics_df = pd.DataFrame({
            "Metric":   ["Recall", "Precision", "F₂ Score", "AUPRC"],
            "Target":   [0.80, 0.40, 0.65, 0.50],
            "Achieved": [round(r,3), round(p,3), round(f,3), round(a,3)],
        })
        fig_bar = go.Figure()
        fig_bar.add_bar(
            x=metrics_df["Metric"], y=metrics_df["Target"],
            name="Target", marker_color="rgba(255,255,255,0.12)",
            marker_line_color=BORDER, marker_line_width=1,
        )
        fig_bar.add_bar(
            x=metrics_df["Metric"], y=metrics_df["Achieved"],
            name="Achieved",
            marker_color=[met_color(v, t) for v, t in
                          zip(metrics_df["Achieved"], metrics_df["Target"])],
            opacity=0.9,
        )
        fig_bar.update_layout(
            **CHART, height=300,
            barmode="group",
            yaxis=dict(range=[0, 1], title="Score", gridcolor=BORDER),
            legend=dict(orientation="h", y=1.1),
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    # ── Metric explanations ──
    st.markdown('<div class="section-title">Metric Definitions</div>', unsafe_allow_html=True)
    cols = st.columns(4)
    defs = [
        ("Recall", "% of real dips the model caught.\nPriority metric — missing a dip is costly.", DIP_CLR),
        ("Precision", "% of predicted dips that were real.\nControls the false alarm rate.", WARN_CLR),
        ("F₂ Score", "Harmonic mean weighting recall 2× over precision.\nPrimary evaluation metric.", BTC_CLR),
        ("AUPRC", "Area under precision-recall curve.\nRobust measure under class imbalance.", ETH_CLR),
    ]
    for col, (name, desc, color) in zip(cols, defs):
        col.markdown(f"""
        <div class="kpi-card" style="--accent:{color}">
            <div class="kpi-label">{name}</div>
            <div style="font-size:12px;color:{TEXT_MUT};line-height:1.6;white-space:pre-line">{desc}</div>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — Feature Importance
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Feature Importance":
    st.markdown('<div class="page-title">Feature Importance</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">LightGBM gain-based importance — how much each feature reduces prediction error.</div>', unsafe_allow_html=True)

    fi_path = MODELS_DIR / "feature_importance.csv"
    if not fi_path.exists():
        st.warning("feature_importance.csv not found. Run train.py first.")
        st.stop()

    fi = pd.read_csv(fi_path)
    fi["category"] = fi["feature"].apply(
        lambda x: "Daily Context" if x.startswith("d_") else
                  ("ETH Signal" if "eth" in x else
                   ("Volatility" if "volatil" in x or "atr" in x or "bb" in x else
                    ("Momentum" if "return" in x or "momentum" in x or "rsi" in x or "macd" in x else
                     ("Volume" if "volume" in x else "Other"))))
    )

    cat_colors = {
        "Daily Context": ETH_CLR,
        "ETH Signal":    "#A78BFA",
        "Volatility":    DIP_CLR,
        "Momentum":      BTC_CLR,
        "Volume":        SAFE_CLR,
        "Other":         TEXT_MUT,
    }

    col_chart, col_cats = st.columns([3, 1])

    with col_chart:
        st.markdown('<div class="section-title">Top 25 Features by Gain</div>', unsafe_allow_html=True)
        top = fi.head(25).sort_values("importance")
        colors = [cat_colors.get(c, TEXT_MUT) for c in top["category"]]

        fig_fi = go.Figure(go.Bar(
            x=top["importance"], y=top["feature"],
            orientation="h",
            marker=dict(color=colors, line=dict(color="rgba(0,0,0,0)")),
            text=top["importance"].map("{:,.0f}".format),
            textposition="outside",
            textfont=dict(size=10, color=TEXT_MUT),
        ))
        fig_fi.update_layout(**CHART, height=620, showlegend=False)
        fig_fi.update_xaxes(title_text="Gain Importance", gridcolor=BORDER)
        fig_fi.update_yaxes(gridcolor="rgba(0,0,0,0)", showgrid=False, tickfont=dict(size=11))
        st.plotly_chart(fig_fi, use_container_width=True)

    with col_cats:
        st.markdown('<div class="section-title">By Category</div>', unsafe_allow_html=True)
        cat_totals = fi.groupby("category")["importance"].sum().sort_values(ascending=False)
        total_imp  = cat_totals.sum()
        for cat, val in cat_totals.items():
            color = cat_colors.get(cat, TEXT_MUT)
            st.markdown(f"""
            <div style="margin-bottom:14px">
                <div style="display:flex;justify-content:space-between;margin-bottom:4px">
                    <span style="font-size:12px;color:{color};font-weight:600">{cat}</span>
                    <span style="font-size:12px;color:{TEXT_MUT}">{val/total_imp:.0%}</span>
                </div>
                <div style="background:{BORDER};border-radius:4px;height:6px">
                    <div style="background:{color};width:{val/total_imp*100:.0f}%;height:6px;border-radius:4px"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="section-title" style="margin-top:24px">Legend</div>
        """, unsafe_allow_html=True)
        for cat, color in cat_colors.items():
            st.markdown(f"""
            <div style="display:flex;align-items:center;gap:8px;margin-bottom:8px">
                <div style="width:10px;height:10px;background:{color};border-radius:50%;flex-shrink:0"></div>
                <span style="font-size:12px;color:{TEXT_MUT}">{cat}</span>
            </div>
            """, unsafe_allow_html=True)
