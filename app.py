import streamlit as st
import numpy as np
import pandas as pd
import json
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
from datetime import datetime

from logic.context import ContextRetriever
from logic.velocity import VelocityEngine
from logic.slider import FrictionSlider
from logic.ml_model import SAVEModel
from simulation.benchmarker import run_benchmark
from simulation.metrics import QualityMetrics

# ─────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────
st.set_page_config(page_title="SAVE Engine Simulator", layout="wide", page_icon="🛡️")
st.title("🛡️ SAVE: Session-Aware Velocity Engine")
st.caption("Real-time adaptive fraud risk scoring. Adjust any input to see scores update instantly.")

# ─────────────────────────────────────────────
# Load static data
# ─────────────────────────────────────────────
BASE = Path(__file__).parent
personas_path = BASE / "data" / "personas.json"
config_path   = BASE / "data" / "weights_config.json"

with open(personas_path) as f:
    personas = json.load(f)
with open(config_path) as f:
    weights_config = json.load(f)

persona_options = {p["label"]: p for p in personas}
cfg_thresholds  = weights_config.get("thresholds", {})
HARD_BLOCK = cfg_thresholds.get("hard_block", 0.70)
STEP_UP    = cfg_thresholds.get("step_up",    0.50)

# ─────────────────────────────────────────────
# Session-state: persist engine instances &
# transaction log across Streamlit reruns
# ─────────────────────────────────────────────
if "engines" not in st.session_state:
    st.session_state.engines = {}          # keyed by persona_id
if "tx_log" not in st.session_state:
    st.session_state.tx_log = []
if "wallets" not in st.session_state:
    st.session_state.wallets = {}          # keyed by persona_id → current balance
if "active_attack" not in st.session_state:
    st.session_state.active_attack = None  # used by Try It buttons
if "ml_model" not in st.session_state:
    with st.spinner("Loading PaySim dataset & training ML model…"):
        model = SAVEModel()
        model.train()
    st.session_state.ml_model = model

ml_model: SAVEModel = st.session_state.ml_model

# ─────────────────────────────────────────────
# Sidebar — persona & attack configuration
# ─────────────────────────────────────────────
st.sidebar.header("👤 User Context")
persona_choice   = st.sidebar.selectbox("Active Persona", list(persona_options.keys()))
selected_persona = persona_options[persona_choice]
persona_id       = selected_persona["persona_id"]

# Persist VelocityEngine per persona so token bucket is preserved
if persona_id not in st.session_state.engines:
    st.session_state.engines[persona_id] = {
        "retriever": ContextRetriever(selected_persona),
        "velocity":  VelocityEngine(
            persona_id,
            selected_persona["behavioral_traits"]["avg_daily_velocity"],
            avg_daily_spend=selected_persona["behavioral_traits"].get("avg_daily_spend", 200),
            config=weights_config,
        ),
        "slider": FrictionSlider(
            base_threshold=selected_persona["default_threshold"],
            config=weights_config,
        ),
    }

# Wallet balance — initialise once per persona with their starting balance
_starting_balance = selected_persona.get("starting_balance", 500.0)
if persona_id not in st.session_state.wallets:
    st.session_state.wallets[persona_id] = _starting_balance
current_balance = st.session_state.wallets[persona_id]

engines   = st.session_state.engines[persona_id]
retriever: ContextRetriever = engines["retriever"]
velocity:  VelocityEngine   = engines["velocity"]
slider:    FrictionSlider   = engines["slider"]

# Persona info panel
with st.sidebar.expander("📋 Persona Details", expanded=True):
    st.markdown(f"**Income:** {selected_persona.get('income_range', 'N/A')}")
    st.markdown(f"**Age Group:** {selected_persona.get('age_group', 'N/A')}")
    st.markdown(f"**Platform:** {selected_persona.get('preferred_platform', 'N/A')}")
    st.markdown(f"**Freq:** {selected_persona['behavioral_traits']['usage_frequency']}")
    st.markdown(f"**Daily Velocity:** {selected_persona['behavioral_traits']['avg_daily_velocity']} tx/day")
    st.markdown(f"**Avg Daily Spend:** ₱ {selected_persona['behavioral_traits'].get('avg_daily_spend', '?')}")

# ── ML data source badge ────────────────────────────────────────────────────
_ml: SAVEModel = st.session_state.ml_model
st.sidebar.caption(
    f"📦 **ML trained on:** {_ml.data_source}  "
    f"({_ml.training_rows:,} rows)"
)

# ── Wallet Card ─────────────────────────────
st.sidebar.markdown("---")
st.sidebar.subheader("💳 Wallet Balance")
_bal_color = "normal" if current_balance > _starting_balance * 0.25 else "inverse"
st.sidebar.metric(
    label=f"₱ Available ({persona_choice})",
    value=f"₱ {current_balance:,.2f}",
    delta=f"₱ {current_balance - _starting_balance:,.2f} from start",
)
if st.sidebar.button("🔄 Refresh Balance", key="refresh_balance", help="Reset wallet to starting balance"):
    st.session_state.wallets[persona_id] = _starting_balance
    st.rerun()

st.sidebar.markdown("---")


# ── Attack Vector Controls ──────────────────
st.sidebar.header("⚠️ Attack Simulation")
_atk = st.session_state.active_attack  # pre-set by Try It buttons
is_hijacked       = st.sidebar.toggle("🚨 Session Hijack",          value=(_atk == "hijack"))
is_cnp            = st.sidebar.toggle("💳 Card-Not-Present Fraud",  value=(_atk == "cnp"))
is_cred_stuff     = st.sidebar.toggle("🤖 Credential Stuffing",      value=(_atk == "cred_stuff"))
is_account_drain  = st.sidebar.toggle("💸 Account Drain (Rapid Tx)", value=(_atk == "account_drain"))
# Clear the pre-set once rendered so toggling off works normally
st.session_state.active_attack = None

attack_active = any([is_hijacked, is_cnp, is_cred_stuff, is_account_drain])

# ── Expert Mode: Weight Tuning ──────────────
st.sidebar.markdown("---")
with st.sidebar.expander("🔬 Expert Mode — Tune Weights", expanded=False):
    st.caption("Override default signal weights (must sum ≈ 1.0)")
    w_model    = st.slider("ML Model",      0.0, 1.0, slider.w_model,    0.05, key="w_model")
    w_drift    = st.slider("Purpose Drift", 0.0, 1.0, slider.w_drift,    0.05, key="w_drift")
    w_preamble = st.slider("Preamble",      0.0, 1.0, slider.w_preamble, 0.05, key="w_preamble")
    w_velocity = st.slider("Velocity",      0.0, 1.0, slider.w_velocity, 0.05, key="w_velocity")
    w_amount   = st.slider("Amount Risk",   0.0, 1.0, slider.w_amount,   0.05, key="w_amount")
    total_w = w_model + w_drift + w_preamble + w_velocity + w_amount
    st.caption(f"Total weight: **{total_w:.2f}** {'✅' if abs(total_w - 1.0) < 0.15 else '⚠️ not ~1.0'}")
    # Apply custom weights live
    slider.w_model    = w_model
    slider.w_drift    = w_drift
    slider.w_preamble = w_preamble
    slider.w_velocity = w_velocity
    slider.w_amount   = w_amount

# ─────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────
tab_live, tab_session, tab_benchmark, tab_explainer = st.tabs(
    ["🔴 Live Monitor", "📋 Session Log", "📊 Benchmark Analysis", "💡 How Attacks Work"]
)

# ══════════════════════════════════════════════
# TAB 1 — LIVE MONITOR
# ══════════════════════════════════════════════
with tab_live:
    st.write("### Transaction Input")
    col_in1, col_in2 = st.columns(2)
    with col_in1:
        amount = st.number_input(
            "Amount to Send (₱)",
            min_value=1,
            max_value=99999,
            value=100,
            step=50,
            help="Type any amount. The engine scores in real-time.",
        )
        if amount > current_balance:
            st.warning(f"⚠️ Amount exceeds wallet balance (₱ {current_balance:,.2f}). "
                       "Engine still scores — but this would be declined at the bank layer.")
    with col_in2:
        purpose_keys = list(selected_persona["behavioral_traits"]["purpose_weights"].keys())
        purpose = st.selectbox("Purpose", purpose_keys)

    # ── Compute signals ──────────────────────
    # Preamble: hijacked / credential stuffing sessions skip normal navigation
    preamble_events = [] if (is_hijacked or is_cred_stuff) else ["app_nav", "check_balance"]
    preamble_score  = retriever.validate_human_preamble(preamble_events)

    drift       = retriever.calculate_purpose_drift(purpose)
    amount_risk = retriever.calculate_amount_risk(amount)

    # For velocity, account-drain simulates rapid-fire → drain the bucket
    vel_amount  = amount if not is_account_drain else amount * 3
    _, vel_penalty = velocity.check_velocity(vel_amount)

    # ML model probability
    p_model = ml_model.predict_proba(drift, preamble_score, vel_penalty, amount_risk)
    # Card-not-present and credential stuffing boost the ML score directly
    if is_cnp:
        p_model = min(p_model + 0.35, 1.0)
    if is_cred_stuff:
        p_model = min(p_model + 0.25, 1.0)
    if is_hijacked:
        p_model = 0.90

    friction, contributions = slider.calculate_friction_with_breakdown(
        p_model, drift, preamble_score, vel_penalty, amount_risk
    )

    spend_info = velocity.get_spend_info()

    # ── Decision ────────────────────────────
    st.markdown("---")
    st.write("### Risk Assessment")

    gcol1, gcol2, gcol3 = st.columns([1.1, 1.2, 1.2])

    with gcol1:
        # Plotly gauge
        if friction >= HARD_BLOCK:
            gauge_color = "#e74c3c"
            decision_label = "BLOCKED"
            decision_emoji = "🔴"
        elif friction >= STEP_UP:
            gauge_color = "#f39c12"
            decision_label = "STEP-UP AUTH"
            decision_emoji = "🟡"
        else:
            gauge_color = "#2ecc71"
            decision_label = "AUTO-APPROVE"
            decision_emoji = "🟢"

        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=friction,
            number={"font": {"size": 40, "color": gauge_color}},
            gauge={
                "axis": {"range": [0, 1], "tickwidth": 1},
                "bar": {"color": gauge_color, "thickness": 0.25},
                "steps": [
                    {"range": [0.0,  STEP_UP],    "color": "#d5f5e3"},
                    {"range": [STEP_UP, HARD_BLOCK], "color": "#fdebd0"},
                    {"range": [HARD_BLOCK, 1.0],  "color": "#fadbd8"},
                ],
                "threshold": {
                    "line": {"color": "#c0392b", "width": 3},
                    "thickness": 0.8,
                    "value": HARD_BLOCK
                }
            },
            title={"text": f"Friction Score<br><b>{decision_emoji} {decision_label}</b>",
                   "font": {"size": 16}},
        ))
        fig_gauge.update_layout(height=260, margin=dict(t=20, b=0, l=20, r=20))
        st.plotly_chart(fig_gauge, use_container_width=True)

        if friction >= HARD_BLOCK:
            st.error("**ACTION: BLOCK & FACE-ID REQUIRED**")
        elif friction >= STEP_UP:
            st.warning("**ACTION: OTP / STEP-UP AUTH**")
        else:
            st.success("**ACTION: AUTO-APPROVE**")

        # Velocity spend bar
        st.caption("Daily Spend Usage")
        st.progress(spend_info["pct_used"],
                    text=f"₱{spend_info['spent_today']:,.0f} / ₱{spend_info['daily_limit']:,.0f}")
        if spend_info["is_off_hours"]:
            st.warning("⏰ Off-hours transaction detected")

    with gcol2:
        st.write("**Signal Breakdown**")
        signals = {
            "ML Model Score":   p_model,
            "Purpose Drift":    drift,
            "Preamble Anomaly": preamble_score,
            "Velocity Penalty": min((vel_penalty - 1) / 2, 1.0),
            "Amount Risk":      amount_risk,
        }
        for label, val in signals.items():
            st.progress(float(val), text=f"{label}: {val:.2f}")

    with gcol3:
        # Explainability waterfall chart
        st.write("**Signal Contribution (Waterfall)**")
        labels = list(contributions.keys())
        values = [round(v, 4) for v in contributions.values()]

        fig_wf = go.Figure(go.Bar(
            x=values,
            y=labels,
            orientation="h",
            marker_color=[
                "#e74c3c" if v > 0.1 else "#f39c12" if v > 0.05 else "#2ecc71"
                for v in values
            ],
            text=[f"{v:.3f}" for v in values],
            textposition="outside",
        ))
        fig_wf.update_layout(
            xaxis_title="Weighted Contribution to Score",
            height=260,
            margin=dict(t=10, b=10, l=10, r=40),
            xaxis=dict(range=[0, max(values) * 1.4 if values else 1]),
        )
        st.plotly_chart(fig_wf, use_container_width=True)

    # ── Plain-English Explanation ────────────
    st.markdown("---")
    st.write("**What's driving this score?**")
    reasons = []
    if is_hijacked:        reasons.append("🚨 **Session hijack** active — ML score forced high")
    if is_cnp:             reasons.append("💳 **Card-not-present** fraud flag — unknown device fingerprint")
    if is_cred_stuff:      reasons.append("🤖 **Credential stuffing** — new device with no app history")
    if is_account_drain:   reasons.append("💸 **Account drain** simulated — velocity bucket depleted")
    if drift > 0.5:        reasons.append(f"⚠️ **Unusual purpose** — `{purpose}` is uncommon for this persona")
    if preamble_score > 0.5: reasons.append("⚠️ **Flash transaction** — no natural navigation before payment")
    if vel_penalty > 1.0:  reasons.append(f"⚠️ **High velocity** — penalty ×{vel_penalty:.1f}")
    if amount_risk > 0.5:  reasons.append(f"⚠️ **Large amount** — ₱{amount:,} is {amount / spend_info['daily_limit'] * 100:.0f}% of daily limit")
    if not reasons:        reasons.append("✅ All signals look normal for this persona")
    for r in reasons:
        st.markdown(f"- {r}")

    # ── Log + Submit button ──────────────────
    st.markdown("---")
    btn_cols = st.columns([2, 1])
    with btn_cols[0]:
        submit_clicked = st.button("✅ Submit Transaction", type="primary", use_container_width=True)
    with btn_cols[1]:
        log_only = st.button("📝 Log Only", use_container_width=True)

    if submit_clicked or log_only:
        # Deduct balance only on auto-approved non-attack submissions
        if submit_clicked and decision_label == "AUTO-APPROVE" and amount <= current_balance:
            st.session_state.wallets[persona_id] -= amount
            st.success(f"✅ Transaction approved. ₱ {amount:,} deducted — new balance: "
                       f"₱ {st.session_state.wallets[persona_id]:,.2f}")
        elif submit_clicked and decision_label == "BLOCKED":
            st.error("🔴 Transaction blocked. No funds deducted.")
        elif submit_clicked and decision_label == "STEP-UP AUTH":
            st.warning("🟡 Step-up required. Transaction held — no funds deducted.")

        st.session_state.tx_log.append({
            "Time":      datetime.now().strftime("%H:%M:%S"),
            "Amount":    f"₱ {amount:,}",
            "Purpose":   purpose,
            "Friction":  friction,
            "Decision":  decision_label,
            "Attack":    ", ".join(
                [a for a, f in [("Hijack", is_hijacked), ("CNP", is_cnp),
                                 ("CredStuff", is_cred_stuff), ("Drain", is_account_drain)] if f]
            ) or "None",
        })
        if log_only:
            st.info("📝 Logged (no funds moved).")

    st.caption(
        f"Hard block @ `{HARD_BLOCK}` · Step-up @ `{STEP_UP}` · "
        f"Base threshold: `{selected_persona['default_threshold']}`"
    )

# ══════════════════════════════════════════════
# TAB 2 — SESSION LOG
# ══════════════════════════════════════════════
with tab_session:
    st.write("### Transaction Session History")
    if not st.session_state.tx_log:
        st.info("No transactions logged yet. Go to **Live Monitor**, set your inputs, and click **Submit Transaction**.")
    else:
        df_log = pd.DataFrame(st.session_state.tx_log)
        st.dataframe(df_log, use_container_width=True)

        # Friction over time mini-chart
        fig_line = px.line(
            df_log.reset_index(),
            x="index", y="Friction",
            markers=True,
            color_discrete_sequence=["#e74c3c"],
            title="Friction Score Over Session",
            labels={"index": "Transaction #"},
        )
        fig_line.add_hline(y=HARD_BLOCK, line_dash="dash", line_color="red",
                           annotation_text="Hard Block", annotation_position="top right")
        fig_line.add_hline(y=STEP_UP, line_dash="dot", line_color="orange",
                           annotation_text="Step-Up", annotation_position="top right")
        fig_line.update_layout(height=300)
        st.plotly_chart(fig_line, use_container_width=True)

        if st.button("🗑️ Clear Session Log"):
            st.session_state.tx_log = []
            st.rerun()

# ══════════════════════════════════════════════
# TAB 3 — BENCHMARK ANALYSIS
# ══════════════════════════════════════════════
with tab_benchmark:
    st.write("### Population-Scale Benchmark")
    st.caption("Runs the SAVESimulator across 1,000 normal + 100 attack transactions for the selected persona.")

    bcol1, bcol2 = st.columns(2)
    with bcol1:
        n_normal = st.number_input("Normal Transactions", 100, 5000, 1000, 100)
    with bcol2:
        n_attack = st.number_input("Attack Transactions",  10,  500,  100,  10)

    if st.button("▶ Run Benchmark"):
        with st.spinner("Running simulation…"):
            df_bench = run_benchmark(
                selected_persona,
                config=weights_config,
                n_normal=int(n_normal),
                n_attack=int(n_attack),
            )
            metrics  = QualityMetrics(df_bench)
            cm       = metrics.calculate_confusion_matrix()
            econ     = metrics.calculate_economic_impact()

        st.session_state.bench_df   = df_bench
        st.session_state.bench_cm   = cm
        st.session_state.bench_econ = econ

    if "bench_df" in st.session_state:
        df_bench = st.session_state.bench_df
        cm       = st.session_state.bench_cm
        econ     = st.session_state.bench_econ

        # ── Key Metrics ─────────────────────
        st.markdown("#### Performance Metrics")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Precision",  f"{cm['precision']:.1%}")
        m2.metric("Recall",     f"{cm['recall']:.1%}")
        m3.metric("False Positives", cm['fp_count'])
        m4.metric("True Positives",  cm['tp_count'])

        # ── Economic Impact ──────────────────
        st.markdown("#### Economic Impact (Net Error Revenue)")
        e1, e2, e3 = st.columns(3)
        e1.metric("Net Revenue Saved",  f"₱{econ['net_revenue_saved']:,.0f}")
        e2.metric("ROI Ratio",          f"{econ['roi_ratio']:.1f}x")
        e3.metric("FP/TP Ratio",        f"{econ['friction_ratio']:.2f}")

        # ── Confusion Matrix Heatmap ─────────
        st.markdown("#### Confusion Matrix")
        tp = cm['tp_count']
        fp = cm['fp_count']
        fn = int(n_attack) - tp
        tn = int(n_normal) - fp

        fig_cm = px.imshow(
            [[tn, fp], [fn, tp]],
            labels=dict(x="Predicted", y="Actual", color="Count"),
            x=["Not Blocked", "Blocked"],
            y=["Normal", "Attack"],
            color_continuous_scale="RdYlGn_r",
            text_auto=True,
        )
        fig_cm.update_layout(height=300)
        st.plotly_chart(fig_cm, use_container_width=True)

        # ── Friction Distribution ────────────
        st.markdown("#### Friction Score Distribution")
        fig_hist = px.histogram(
            df_bench, x="friction", color="is_attack",
            nbins=40, barmode="overlay", opacity=0.7,
            color_discrete_map={True: "#e74c3c", False: "#2ecc71"},
            labels={"friction": "Friction Score", "is_attack": "Is Attack"},
        )
        fig_hist.add_vline(x=HARD_BLOCK, line_dash="dash", line_color="red",
                           annotation_text="Hard Block")
        fig_hist.add_vline(x=STEP_UP, line_dash="dot", line_color="orange",
                           annotation_text="Step-Up")
        fig_hist.update_layout(height=320)
        st.plotly_chart(fig_hist, use_container_width=True)

        # ── Feature Importances ──────────────
        st.markdown("#### ML Feature Importances")
        fi = ml_model.feature_importances()
        fig_fi = px.bar(
            x=list(fi.keys()), y=list(fi.values()),
            labels={"x": "Signal", "y": "Importance"},
            color=list(fi.values()),
            color_continuous_scale="Blues",
        )
        fig_fi.update_layout(height=280, showlegend=False)
        st.plotly_chart(fig_fi, use_container_width=True)

# ══════════════════════════════════════════════
# TAB 4 — HOW ATTACKS WORK
# ══════════════════════════════════════════════
with tab_explainer:
    st.write("### 💡 How Attacks Work — and How SAVE Detects Them")
    st.caption(
        "Each card below explains an attack vector: what the attacker does, "
        "which SAVE signals spike, and why the engine flags it. "
        "Click **Try It** to jump to Live Monitor with that attack pre-enabled."
    )

    # ── Helper: render a coloured impact badge ──
    def _badge(level: str) -> str:
        if level == "HIGH":   return "🔴 **HIGH**"
        if level == "MEDIUM": return "🟡 **MEDIUM**"
        return "🟢 Low"

    # ── Attack definitions ──────────────────────
    attacks = [
        {
            "key":   "hijack",
            "emoji": "🚨",
            "name":  "Session Hijack",
            "what":  (
                "The attacker takes over an **already authenticated session** — "
                "e.g. by stealing a session token via XSS, a compromised device, "
                "or a man-in-the-browser attack. The victim's credentials are not "
                "needed because the session is already 'trusted'."
            ),
            "how": [
                "Attacker injects or replays the victim's live session token.",
                "They skip the usual app navigation (no balance check, no menu browsing).",
                "A large P2P transfer is initiated instantly — a 'flash transaction'.",
                "The app behaves exactly as if the real user sent the payment.",
            ],
            "detection": (
                "The absence of natural app navigation before the payment is the "
                "primary giveaway. SAVE's **Preamble Gate** scores this as 0.9 "
                "friction. The **ML Model** also forces P(attack) = 0.90 because "
                "the full behavioural pattern matches a hijack archetype."
            ),
            "impacts": {
                "ML Model Score":   "HIGH",
                "Purpose Drift":    "MEDIUM",
                "Preamble Anomaly": "HIGH",
                "Velocity Penalty": "Low",
                "Amount Risk":      "MEDIUM",
            },
        },
        {
            "key":   "cnp",
            "emoji": "💳",
            "name":  "Card-Not-Present (CNP) Fraud",
            "what":  (
                "The attacker uses **stolen card details** (number, CVV, expiry) "
                "without physically possessing the card. This is the most common "
                "e-commerce fraud type. The device fingerprint is unknown and the "
                "session has no prior history with the platform."
            ),
            "how": [
                "Attacker obtains card data from a phishing kit or dark-web dump.",
                "They create a new session on an unrecognised device or browser.",
                "They skip to checkout directly — no browsing, no history.",
                "A purchase or transfer is attempted, often to a mule account.",
            ],
            "detection": (
                "The new/unknown device means no human preamble — **Preamble score = 0.9**. "
                "The ML model receives a **+0.35 boost** to P(attack) to reflect "
                "the unknown device fingerprint signal. Together these almost always "
                "breach the Hard Block threshold."
            ),
            "impacts": {
                "ML Model Score":   "HIGH",
                "Purpose Drift":    "MEDIUM",
                "Preamble Anomaly": "HIGH",
                "Velocity Penalty": "Low",
                "Amount Risk":      "MEDIUM",
            },
        },
        {
            "key":   "cred_stuff",
            "emoji": "🤖",
            "name":  "Credential Stuffing",
            "what":  (
                "Bots use **leaked username/password pairs** from other data breaches "
                "and try them in bulk against the app. Successful logins are handed off "
                "for fraud. The volume is high but each individual attempt looks like "
                "a fresh, legitimate login from a new device."
            ),
            "how": [
                "Bot cycles through thousands of credential pairs automatically.",
                "On a successful login, the session is immediately used for a transaction.",
                "No prior app behaviour — the bot doesn't browse, it just pays.",
                "The transaction may be modest to avoid obvious amount-risk flags.",
            ],
            "detection": (
                "No natural session preamble is the strongest signal — **Preamble = 0.9**. "
                "The ML model gets a **+0.25 boost** for the credential stuffing pattern "
                "(new device, no history, immediate transact). "
                "Amount risk may be low, but preamble + ML together are enough to trigger Step-Up."
            ),
            "impacts": {
                "ML Model Score":   "HIGH",
                "Purpose Drift":    "Low",
                "Preamble Anomaly": "HIGH",
                "Velocity Penalty": "Low",
                "Amount Risk":      "Low",
            },
        },
        {
            "key":   "account_drain",
            "emoji": "💸",
            "name":  "Account Drain (Rapid Transactions)",
            "what":  (
                "The attacker has access to the account (via hijack or credential stuffing) "
                "and **rapidly fires multiple large transactions** to drain the balance "
                "before the victim notices. The goal is speed — move funds before any "
                "manual review or system block can occur."
            ),
            "how": [
                "Attacker queues several large P2P transfers in quick succession.",
                "Each transaction depletes the velocity token bucket further.",
                "Amounts are typically large — often a significant % of daily limit.",
                "Transactions happen at unusual hours to delay victim detection.",
            ],
            "detection": (
                "The **Velocity Engine's token bucket** drains rapidly: once tokens hit 0, "
                "the penalty multiplier doubles (×2). Cumulative daily spend exceeding "
                "the persona's limit multiplies the penalty further (up to ×3). "
                "Large per-transaction **Amount Risk** scores combine with velocity "
                "to push friction well above the Hard Block threshold."
            ),
            "impacts": {
                "ML Model Score":   "MEDIUM",
                "Purpose Drift":    "MEDIUM",
                "Preamble Anomaly": "Low",
                "Velocity Penalty": "HIGH",
                "Amount Risk":      "HIGH",
            },
        },
    ]

    # ── Render each attack card ─────────────────
    for atk in attacks:
        with st.expander(f"{atk['emoji']} {atk['name']}", expanded=False):
            st.markdown(f"**What is it?**\n\n{atk['what']}")
            st.markdown("---")

            col_steps, col_table = st.columns([1.2, 1])

            with col_steps:
                st.markdown("**🔍 How the attacker behaves:**")
                for i, step in enumerate(atk["how"], 1):
                    st.markdown(f"{i}. {step}")
                st.markdown("")
                st.markdown("**🛡️ How SAVE detects it:**")
                st.info(atk["detection"])

            with col_table:
                st.markdown("**📊 Signal Impact at a Glance:**")
                rows = []
                for signal, level in atk["impacts"].items():
                    rows.append(f"| {signal} | {_badge(level)} |")
                table_md = (
                    "| Signal | Impact |\n"
                    "|---|---|\n" +
                    "\n".join(rows)
                )
                st.markdown(table_md)

            st.markdown("")
            if st.button(
                f"▶ Try It — Enable {atk['name']}",
                key=f"try_{atk['key']}",
                help="Switches to Live Monitor with this attack pre-enabled",
            ):
                st.session_state.active_attack = atk["key"]
                st.rerun()