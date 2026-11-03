"""
ReturnIQ — Streamlit Application
Minimal, dark-themed retail return risk intelligence dashboard.
Features dynamic model selection across benchmarked ML models (Predict & Benchmark tabs).
"""

import os, sys, json, logging
import pandas as pd
import numpy as np
import streamlit as st

sys.path.insert(0, os.path.dirname(__file__))
from src.inference import load_artifacts, get_available_models, get_model_threshold, predict

# ── paths ─────────────────────────────────────────────────────────────────────
FIGURES   = "reports/figures"
CSV_PATH  = "reports/model_comparison.csv"
JSON_PATH = "reports/final_test_metrics.json"

logging.basicConfig(level=logging.WARNING)

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ReturnIQ",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* reset & base */
html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, sans-serif;
    background-color: #0B0B0D;
    color: #E4E4E7;
}
#MainMenu, footer, header { visibility: hidden; }
.block-container {
    padding: 1.8rem 2.5rem 3rem;
    max-width: 1080px;
}

/* ── header ── */
.riq-header {
    display: flex; align-items: baseline; gap: 0;
    font-size: 1.35rem; font-weight: 700; letter-spacing: -0.4px;
    color: #F4F4F5; margin-bottom: 1px;
}
.riq-header .iq { color: #10B981; }
.riq-sub {
    font-size: 0.72rem; color: #52525B;
    letter-spacing: 0.5px; text-transform: uppercase;
    margin-bottom: 0;
}
.status-dot {
    display: inline-block; width: 6px; height: 6px;
    border-radius: 50%; margin-right: 5px; margin-bottom: 1px;
    vertical-align: middle;
}
.status-ready { background: #10B981; }
.status-error { background: #EF4444; }
.status-text  { font-size: 0.72rem; color: #52525B; }

/* ── section label ── */
.sec-label {
    font-size: 0.63rem; font-weight: 600; text-transform: uppercase;
    letter-spacing: 1px; color: #52525B;
    border-bottom: 1px solid #1C1C20; padding-bottom: 6px;
    margin-bottom: 0.85rem; margin-top: 0.25rem;
}

/* ── stat card ── */
.kpi {
    background: #141417; border: 1px solid #29292F;
    border-radius: 8px; padding: 1rem 1.15rem;
}
.kpi-label { font-size: 0.65rem; font-weight: 600; text-transform: uppercase;
             letter-spacing: 0.8px; color: #71717A; margin-bottom: 5px; }
.kpi-value { font-size: 1.55rem; font-weight: 700; color: #F4F4F5;
             line-height: 1; letter-spacing: -0.5px; }
.kpi-sub   { font-size: 0.68rem; color: #52525B; margin-top: 4px; }

/* ── form card ── */
.form-card {
    background: #141417; border: 1px solid #29292F;
    border-radius: 10px; padding: 1.25rem 1.4rem;
}
.form-title { font-size: 0.9rem; font-weight: 600; color: #F4F4F5;
              margin-bottom: 2px; }
.form-sub   { font-size: 0.73rem; color: #71717A; margin-bottom: 0.8rem; }

/* ── model selector banner ── */
.selector-box {
    background: #18181B; border: 1px solid #27272A;
    border-radius: 8px; padding: 0.85rem 1rem;
    margin-bottom: 1rem;
}
.model-caption {
    font-size: 0.72rem; color: #71717A; margin-top: 4px;
}
.model-caption span { color: #10B981; font-weight: 600; }

/* ── result panel ── */
.result-empty {
    background: #141417; border: 1px solid #29292F; border-radius: 10px;
    padding: 3.5rem 1.5rem; text-align: center;
    color: #3F3F46; font-size: 0.82rem;
}
.result-panel {
    background: #141417; border: 1px solid #29292F; border-radius: 10px;
    padding: 1.5rem;
}
.prob-big {
    font-size: 3.2rem; font-weight: 800; line-height: 1;
    letter-spacing: -1px;
}
.prob-label { font-size: 0.65rem; font-weight: 600; text-transform: uppercase;
              letter-spacing: 0.8px; color: #71717A; margin-bottom: 6px; }

.badge {
    display: inline-block; padding: 4px 12px; border-radius: 4px;
    font-size: 0.75rem; font-weight: 600; letter-spacing: 0.4px;
    text-transform: uppercase; margin-top: 10px;
}
.badge-high { background: #3B0F0F; color: #F87171; border: 1px solid #7F1D1D; }
.badge-low  { background: #052E16; color: #34D399; border: 1px solid #065F46; }

/* progress bar track */
.bar-track {
    background: #1C1C20; border-radius: 3px; height: 6px;
    margin: 16px 0 10px; overflow: hidden;
}
.bar-fill { height: 100%; border-radius: 3px; transition: width 0.4s ease; }

/* threshold info row */
.thresh-row {
    display: flex; flex-direction: column; gap: 4px; margin-top: 1.1rem;
    font-size: 0.76rem; color: #71717A;
}
.thresh-row span { color: #A1A1AA; font-weight: 500; }
.tag-badge {
    display: inline-block; padding: 2px 6px; border-radius: 3px;
    font-size: 0.65rem; font-weight: 600; text-transform: uppercase;
    background: #27272A; color: #D4D4D8; margin-left: 6px;
}

/* action suggestion */
.action-box {
    margin-top: 1.2rem; padding: 0.85rem 1rem;
    background: #18181B; border: 1px solid #27272A;
    border-radius: 6px; font-size: 0.8rem; color: #D4D4D8;
    line-height: 1.45;
}
.action-title { font-weight: 600; color: #F4F4F5; margin-bottom: 2px; }

/* ── tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: transparent;
    border-bottom: 1px solid #29292F;
    gap: 0;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: #71717A !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    padding: 0.6rem 1.4rem !important;
    border-radius: 0 !important;
    border: none !important;
}
.stTabs [aria-selected="true"] {
    color: #F4F4F5 !important;
    border-bottom: 2px solid #10B981 !important;
}

/* ── inputs ── */
.stNumberInput input, .stSelectbox > div > div {
    background: #1C1C20 !important;
    border: 1px solid #29292F !important;
    border-radius: 7px !important;
    color: #E4E4E7 !important;
    font-size: 0.84rem !important;
}
label { font-size: 0.78rem !important; color: #A1A1AA !important;
        font-weight: 500 !important; }

/* ── primary button ── */
.stButton > button {
    background: #10B981 !important; color: #052E16 !important;
    border: none !important; border-radius: 7px !important;
    font-weight: 700 !important; font-size: 0.83rem !important;
    padding: 0.55rem 1.4rem !important; width: 100% !important;
    letter-spacing: 0.2px !important;
}
.stButton > button:hover {
    background: #059669 !important; color: #fff !important;
}

/* ── dataframe ── */
.stDataFrame { font-size: 0.8rem; }
.stDataFrame th { background: #1C1C20 !important; color: #71717A !important;
                  font-size: 0.68rem !important; text-transform: uppercase;
                  letter-spacing: 0.5px; }
.stDataFrame td { background: #141417 !important; color: #E4E4E7 !important; }

/* ── images ── */
.stImage img { border-radius: 8px; border: 1px solid #29292F; }

/* ── divider ── */
hr { border: none; border-top: 1px solid #1C1C20; margin: 1.2rem 0; }
</style>
""", unsafe_allow_html=True)

# ── cached loaders ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_bundle():
    return load_artifacts()

@st.cache_data(show_spinner=False)
def get_comparison():
    if os.path.exists(CSV_PATH):
        return pd.read_csv(CSV_PATH)
    return None

@st.cache_data(show_spinner=False)
def get_test_metrics():
    return json.load(open(JSON_PATH)) if os.path.exists(JSON_PATH) else None

bundle = get_bundle()
model_ready = bundle is not None
available_models = get_available_models(bundle) if model_ready else []
comparison_df = get_comparison()
test_metrics_json = get_test_metrics()

# Helper dict for validation PR-AUCs
val_metrics_map = {}
if comparison_df is not None:
    for _, row in comparison_df.iterrows():
        val_metrics_map[row["model_name"]] = {
            "pr_auc": float(row["pr_auc"]),
            "roc_auc": float(row["roc_auc"]),
            "f1": float(row["f1"])
        }

# ── helpers ───────────────────────────────────────────────────────────────────
def kpi(label, value, sub=""):
    st.markdown(
        f'<div class="kpi"><div class="kpi-label">{label}</div>'
        f'<div class="kpi-value">{value}</div>'
        f'<div class="kpi-sub">{sub}</div></div>',
        unsafe_allow_html=True,
    )

def sec(label):
    st.markdown(f'<div class="sec-label">{label}</div>', unsafe_allow_html=True)

def fig(path, caption=""):
    if os.path.exists(path):
        st.image(path, caption=caption, use_column_width=True)
    else:
        st.markdown(
            f'<div style="background:#141417;border:1px solid #29292F;border-radius:8px;'
            f'padding:2rem;text-align:center;color:#3F3F46;font-size:0.78rem;">'
            f'Run <code>python main.py</code> to generate this figure.</div>',
            unsafe_allow_html=True,
        )

# ── header ────────────────────────────────────────────────────────────────────
h_left, h_right = st.columns([7, 3])
with h_left:
    st.markdown(
        '<div class="riq-header">Return<span class="iq">IQ</span></div>'
        '<div class="riq-sub">Retail Return Risk Intelligence</div>',
        unsafe_allow_html=True,
    )
with h_right:
    st.markdown("<br>", unsafe_allow_html=True)
    dot_cls = "status-ready" if model_ready else "status-error"
    active_count = len(available_models)
    msg = f"{active_count} Models Ready" if model_ready else "Model not found — run <code>python main.py</code>"
    st.markdown(
        f'<div class="status-text" style="text-align:right">'
        f'<span class="status-dot {dot_cls}"></span>{msg}</div>',
        unsafe_allow_html=True,
    )

st.markdown("<hr>", unsafe_allow_html=True)

# ── navigation (Predict & Benchmark) ──────────────────────────────────────────
tab_pred, tab_bench = st.tabs(["Predict", "Benchmark"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — PREDICT
# ══════════════════════════════════════════════════════════════════════════════
with tab_pred:
    if not model_ready or not available_models:
        st.error(
            "Model artifacts not found. Run `python main.py` to train and serialize the benchmarked models.",
            icon="🚫",
        )
    else:
        # ── Model Selector Bar ────────────────────────────────────────────────
        st.markdown(
            '<div class="form-card" style="margin-bottom:1rem;padding:1.1rem 1.4rem;">'
            '<div class="form-title">Active Inference Model</div>'
            '<div class="form-sub">Select which benchmarked algorithm to use for return risk evaluation.</div>'
            '</div>',
            unsafe_allow_html=True,
        )

        sel_c1, sel_c2 = st.columns([6, 4])
        with sel_c1:
            default_index = 0
            selected_model = st.selectbox(
                "Model Architecture",
                options=available_models,
                index=default_index,
                help="Choose from the algorithms benchmarked on the chronological dataset."
            )
            # Display model metadata caption
            m_stats = val_metrics_map.get(selected_model, {})
            pr_val_text = f"Val PR-AUC: <span>{m_stats.get('pr_auc', 0.0):.4f}</span>" if "pr_auc" in m_stats else ""
            roc_val_text = f"Val ROC-AUC: <span>{m_stats.get('roc_auc', 0.0):.4f}</span>" if "roc_auc" in m_stats else ""
            st.markdown(
                f'<div class="model-caption">Architecture: <b>{selected_model}</b> &nbsp;·&nbsp; {pr_val_text} &nbsp;·&nbsp; {roc_val_text}</div>',
                unsafe_allow_html=True,
            )

        with sel_c2:
            model_opt_t = get_model_threshold(bundle, selected_model, use_optimal=True)
            threshold_mode = st.radio(
                "Decision Threshold",
                options=[f"Optimized T* ({model_opt_t:.2f})", "Default (0.50)"],
                index=0,
                horizontal=True,
                help="Optimized T* minimizes net cost under imbalanced 1.6% return rate. Default 0.50 is the standard classification boundary."
            )
            use_opt_threshold = "Optimized" in threshold_mode
            active_threshold_val = model_opt_t if use_opt_threshold else 0.50

        st.markdown("<br>", unsafe_allow_html=True)
        left, right = st.columns([5, 4], gap="large")

        # ── left: form ────────────────────────────────────────────────────────
        with left:
            st.markdown(
                '<div class="form-card">'
                '<div class="form-title">New Prediction</div>'
                '<div class="form-sub">Enter transaction details to estimate return risk.</div>'
                '</div>',
                unsafe_allow_html=True,
            )
            st.markdown("<br>", unsafe_allow_html=True)

            with st.form("predict_form"):
                sec("Transaction")
                c1, c2 = st.columns(2)
                qty   = c1.number_input("Quantity", min_value=1.0, max_value=500.0, value=6.0, step=1.0)
                price = c2.number_input("Unit Price (£)", min_value=0.01, max_value=999.0, value=3.75, step=0.25)
                total_spend = qty * price

                c3, c4 = st.columns(2)
                cat   = c3.selectbox("Product Category", ["85","22","84","47","21","20","23","79","OTHER"])
                is_uk = c4.selectbox("Country", ["United Kingdom", "International"]) == "United Kingdom"

                c5, c6, c7 = st.columns(3)
                hour   = c5.number_input("Hour (0–23)", 0, 23, 14, step=1)
                dow    = c6.number_input("Day of Week (0=Mon)", 0, 6, 2, step=1)
                month  = c7.number_input("Month (1–12)", 1, 12, 10, step=1)

                st.markdown("<br>", unsafe_allow_html=True)
                sec("Customer History  ·  prior to this order")
                d1, d2 = st.columns(2)
                prior_orders  = d1.number_input("Prior Orders", 0.0, 500.0, 4.0, step=1.0)
                prior_items   = d2.number_input("Prior Items", 0.0, 5000.0, 28.0, step=1.0)
                d3, d4 = st.columns(2)
                prior_spend   = d3.number_input("Prior Spend (£)", 0.0, 50000.0, 120.0, step=10.0)
                prior_returns = d4.number_input("Prior Returns", 0.0, max(prior_items, 1.0), 2.0, step=1.0)
                prior_rr = prior_returns / prior_items if prior_items > 0 else 0.0
                days_since = 14.0 if prior_orders > 0 else -1.0

                st.markdown("<br>", unsafe_allow_html=True)
                sec("Product History  ·  prior to this transaction")
                e1, e2 = st.columns(2)
                prod_purch   = e1.number_input("Historical Purchases", 0.0, 100000.0, 250.0, step=10.0)
                prod_returns = e2.number_input("Historical Returns", 0.0, max(prod_purch, 1.0), 15.0, step=1.0)
                prod_rr = prod_returns / prod_purch if prod_purch > 0 else 0.0

                st.markdown("<br>", unsafe_allow_html=True)
                submitted = st.form_submit_button(f"Predict with {selected_model}")

        # ── right: result ─────────────────────────────────────────────────────
        with right:
            if not submitted and "last_pred" not in st.session_state:
                st.markdown(
                    '<div class="result-empty">'
                    'Enter transaction details<br>and click Predict Return Risk.'
                    '</div>',
                    unsafe_allow_html=True,
                )
            else:
                if submitted:
                    payload = {
                        "Quantity": qty, "UnitPrice": price, "TotalSpend": total_spend,
                        "Hour": hour, "DayOfWeek": dow, "Month": month,
                        "IsWeekend": int(dow >= 5), "IsUK": int(is_uk),
                        "ProductCategory": cat,
                        "cust_prior_orders": prior_orders,
                        "cust_prior_items": prior_items,
                        "cust_prior_spend": prior_spend,
                        "cust_prior_returns": prior_returns,
                        "cust_prior_return_rate": prior_rr,
                        "cust_days_since_last_order": days_since,
                        "prod_prior_purchases": prod_purch,
                        "prod_prior_returns": prod_returns,
                        "prod_prior_return_rate": prod_rr,
                    }
                    try:
                        prob, label, active_t, res_name = predict(
                            bundle,
                            payload,
                            model_name=selected_model,
                            threshold=active_threshold_val
                        )
                        st.session_state["last_pred"] = {
                            "prob": prob,
                            "label": label,
                            "threshold": active_t,
                            "is_optimal": use_opt_threshold,
                            "model_name": res_name,
                            "payload": payload,
                        }
                    except Exception as exc:
                        logging.exception("Inference error")
                        st.error(f"Inference failed with {selected_model}: {exc}")
                        st.stop()

                res = st.session_state.get("last_pred", {})
                if res:
                    prob  = res["prob"]
                    label = res["label"]
                    thr   = res["threshold"]
                    m_used = res["model_name"]
                    is_opt = res.get("is_optimal", True)
                    pct   = prob * 100

                    high = label == "High Risk"
                    prob_color = "#F87171" if high else "#34D399"
                    badge_cls  = "badge-high" if high else "badge-low"
                    bar_color  = "#F87171" if high else "#10B981"

                    thresh_type_badge = "Optimized T*" if is_opt else "Default 0.50"
                    action_msg = (
                        "Flag transaction for proactive confirmation or fit/sizing guidance."
                        if high else
                        "Standard checkout. No risk intervention recommended."
                    )

                    st.markdown(
                        f'<div class="result-panel">'
                        f'<div class="prob-label">Estimated Return Probability</div>'
                        f'<div class="prob-big" style="color:{prob_color}">{pct:.1f}%</div>'
                        f'<span class="badge {badge_cls}">{label}</span>'
                        f'<div class="bar-track"><div class="bar-fill" '
                        f'style="width:{min(pct,100):.1f}%;background:{bar_color}"></div></div>'
                        f'<div class="thresh-row">'
                        f'<div>Inference Model: <span>{m_used}</span></div>'
                        f'<div>Applied Threshold: <span>{thr:.2f}</span><span class="tag-badge">{thresh_type_badge}</span></div>'
                        f'</div>'
                        f'<div class="action-box">'
                        f'<div class="action-title">Recommendation</div>'
                        f'{action_msg}'
                        f'</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — BENCHMARK
# ══════════════════════════════════════════════════════════════════════════════
with tab_bench:
    test_m = get_test_metrics()
    comp   = get_comparison()

    # Top KPI row highlighting primary benchmark achievements
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        val = f"{comp['pr_auc'].max():.4f}" if comp is not None else "—"
        kpi("Best Val PR-AUC", val, "validation set")
    with k2:
        val = f"{comp['roc_auc'].max():.4f}" if comp is not None else "—"
        kpi("Best Val ROC-AUC", val, "validation set")
    with k3:
        b_name = bundle.get("model_name", "HistGradientBoosting") if model_ready else "—"
        kpi("Top Ranked Model", b_name.replace("Classifier",""), "highest validation PR-AUC")
    with k4:
        val = f"{test_m['metrics_at_optimal_threshold']['roc_auc']:.4f}" if test_m else "—"
        kpi("Test ROC-AUC", val, "held-out Nov–Dec 2011")

    st.markdown("<br>", unsafe_allow_html=True)
    sec("Validation Performance (Sep–Nov 2011)")

    if comp is not None:
        # Model Comparison Multiselect Filter
        all_comp_models = comp["model_name"].tolist()
        
        filter_c1, filter_c2 = st.columns([7, 3])
        with filter_c1:
            selected_comp_models = st.multiselect(
                "Filter Models to Compare",
                options=all_comp_models,
                default=all_comp_models,
                help="Select which benchmarked models to display in the comparison table."
            )
        with filter_c2:
            st.markdown("<br>", unsafe_allow_html=True)
            active_m_label = st.session_state.get("last_pred", {}).get("model_name", available_models[0] if available_models else "—")
            st.caption(f"Active Predictor: **{active_m_label}**")

        filtered_comp = comp[comp["model_name"].isin(selected_comp_models)]
        show = filtered_comp[["model_name","roc_auc","pr_auc","precision","recall","f1","train_time_sec"]].copy()
        show.columns = ["Model","ROC-AUC","PR-AUC","Precision","Recall","F1","Train (s)"]
        for c in ["ROC-AUC","PR-AUC","Precision","Recall","F1"]:
            show[c] = show[c].map("{:.4f}".format)
        show["Train (s)"] = show["Train (s)"].map("{:.2f}".format)
        st.dataframe(show, hide_index=True, use_container_width=True)
    else:
        st.warning("No benchmark data. Run `python main.py`.")

    st.markdown("<br>", unsafe_allow_html=True)
    b1, b2 = st.columns(2)
    with b1:
        fig(f"{FIGURES}/roc_curves.png", "ROC Curves · Validation Set")
    with b2:
        fig(f"{FIGURES}/pr_curves.png", "Precision-Recall Curves · Validation Set")

    st.markdown("<br>", unsafe_allow_html=True)
    sec("Held-Out Test Set Performance (Nov–Dec 2011)")

    if test_m and "by_model" in test_m:
        by_model_data = test_m["by_model"]
        test_rows = []
        for m_name in selected_comp_models if comp is not None else by_model_data.keys():
            if m_name in by_model_data:
                entry = by_model_data[m_name]
                opt_metrics = entry["metrics_at_optimal_threshold"]
                test_rows.append({
                    "Model": m_name,
                    "Optimal T*": f"{entry['optimal_threshold']:.2f}",
                    "Test ROC-AUC": f"{opt_metrics['roc_auc']:.4f}",
                    "Test PR-AUC": f"{opt_metrics['pr_auc']:.4f}",
                    "Precision @ T*": f"{opt_metrics['precision']:.2%}",
                    "Recall @ T*": f"{opt_metrics['recall']:.2%}",
                    "F1 @ T*": f"{opt_metrics['f1']:.4f}"
                })
        
        if test_rows:
            test_df_display = pd.DataFrame(test_rows)
            st.dataframe(test_df_display, hide_index=True, use_container_width=True)

        st.markdown("<br>", unsafe_allow_html=True)
        _, cm_col, _ = st.columns([2, 4, 2])
        with cm_col:
            fig(f"{FIGURES}/confusion_matrix.png", f"Confusion Matrix · Selected Best Model on Held-Out Test Set")
    elif test_m:
        m  = test_m["metrics_at_optimal_threshold"]
        t1, t2, t3, t4 = st.columns(4)
        with t1: kpi("Test ROC-AUC", f"{m['roc_auc']:.4f}", "held-out period")
        with t2: kpi("Test PR-AUC",  f"{m['pr_auc']:.4f}", "9× above base prevalence")
        with t3: kpi("Precision @ T*", f"{m['precision']:.2%}", f"T* = {test_m['optimal_threshold']:.2f}")
        with t4: kpi("Recall @ T*",    f"{m['recall']:.2%}", "")

        st.markdown("<br>", unsafe_allow_html=True)
        _, cm_col, _ = st.columns([2, 4, 2])
        with cm_col:
            fig(f"{FIGURES}/confusion_matrix.png", "Confusion Matrix · Held-Out Test Set")
    else:
        st.warning("No test metrics. Run `python main.py`.")
