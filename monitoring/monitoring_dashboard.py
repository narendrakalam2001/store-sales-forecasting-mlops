# ============================================================
# MONITORING DASHBOARD — Store Sales Forecasting System
# ============================================================

import streamlit as st
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import os

st.set_page_config(page_title="Store Sales Forecast Dashboard", layout="wide")
st.title("🛒 Store Sales Forecasting — Monitoring Dashboard")

API_URL      = os.getenv("FORECAST_API_URL", "http://localhost:8000") + "/forecast"
PSI_MODERATE = 0.10
PSI_HIGH     = 0.20

try:
    _SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    BASE_DIR    = os.path.dirname(_SCRIPT_DIR)
except Exception:
    BASE_DIR = os.getcwd()

MODEL_DIR     = os.path.join(BASE_DIR, "forecast_models")
MONITOR_PATH  = os.path.join(MODEL_DIR, "monitor_scores.csv")
DRIFT_PATH    = os.path.join(MODEL_DIR, "feature_drift_report.csv")
RESULTS_PATH  = os.path.join(MODEL_DIR, "model_experiment_results.csv")
CARD_PATH     = os.path.join(MODEL_DIR, "model_card_ensemble_v1.json")
LOG_PATH      = os.path.join(BASE_DIR,  "logs", "prediction_logs.csv")

FAMILIES = [
    "GROCERY I", "BEVERAGES", "PRODUCE", "CLEANING", "DAIRY",
    "MEATS", "BREAD/BAKERY", "PERSONAL CARE", "FROZEN FOODS",
    "DELI", "EGGS", "HOME CARE", "PREPARED FOODS", "SEAFOOD",
    "BEAUTY", "AUTOMOTIVE", "BABY CARE", "BOOKS", "CELEBRATION",
    "HARDWARE", "HOME AND KITCHEN I", "HOME AND KITCHEN II",
    "HOME APPLIANCES", "LADIESWEAR", "LAWN AND GARDEN", "LINGERIE",
    "LIQUOR,WINE,BEER", "MAGAZINES", "PET SUPPLIES",
    "PLAYERS AND ELECTRONICS", "POLO SHIRTS",
    "SCHOOL AND OFFICE SUPPLIES", "GROCERY II",
]

# ── SIDEBAR ───────────────────────────────────────────────────
st.sidebar.header("🔮 Forecast Store Sales")
store_nbr     = st.sidebar.selectbox("Store Number", list(range(1, 55)))
family        = st.sidebar.selectbox("Product Family", FAMILIES)
forecast_date = st.sidebar.date_input("Forecast Date", value=pd.Timestamp("2017-08-16"))
onpromotion   = st.sidebar.number_input("Items on Promotion", min_value=0, value=0)
oil_price     = st.sidebar.number_input("Oil Price (WTI)", min_value=0.0, value=47.5, step=0.5)
is_holiday    = st.sidebar.selectbox("National Holiday?", [0, 1],
                                      format_func=lambda x: "Yes" if x else "No")

if st.sidebar.button("Get Forecast"):
    payload = {
        "store_nbr": store_nbr, "family": family,
        "date": str(forecast_date), "onpromotion": onpromotion,
        "dcoilwtico": oil_price, "is_national_holiday": is_holiday,
    }
    with st.sidebar:
        with st.spinner("Calling API..."):
            try:
                resp = requests.post(API_URL, json=payload, timeout=90)
                if resp.status_code == 200:
                    r    = resp.json()
                    pred = r.get("predicted_sales", 0)
                    band = r.get("confidence_band", "MEDIUM")
                    rule = r.get("rule_triggered")
                    cmap = {"HIGH": "green", "MEDIUM": "orange", "LOW": "red"}
                    st.success("Forecast received!")
                    st.metric("Predicted Sales (units)", f"{pred:,.2f}")
                    st.markdown(
                        f"<h4 style='color:{cmap.get(band,'gray')}'>Confidence: {band}</h4>",
                        unsafe_allow_html=True
                    )
                    if rule:
                        st.warning(f"Rule triggered: {rule}")
                    st.json(r)
                else:
                    st.error(f"API error: HTTP {resp.status_code}")
            except requests.exceptions.Timeout:
                st.warning("Request timed out — Render warming up, try again.")
            except Exception as e:
                st.error(f"Connection error: {e}")

# ── SECTION 1: ALERTS ─────────────────────────────────────────
st.markdown("---")
st.subheader("🚨 Real-Time Monitoring Alerts")
alerts_found = False

if os.path.exists(DRIFT_PATH):
    df_drift = pd.read_csv(DRIFT_PATH)
    if "drift_score" in df_drift.columns and len(df_drift):
        max_psi  = df_drift["drift_score"].max()
        top_feat = df_drift.iloc[0]["feature"] if "feature" in df_drift.columns else "unknown"
        if max_psi >= PSI_HIGH:
            st.error(f"🔴 CRITICAL DRIFT: PSI={max_psi:.4f} on '{top_feat}'. Retrain recommended.")
            alerts_found = True
        elif max_psi >= PSI_MODERATE:
            st.warning(f"🟡 MODERATE DRIFT: PSI={max_psi:.4f} on '{top_feat}'. Monitor closely.")
            alerts_found = True

if os.path.exists(MONITOR_PATH):
    df_mon = pd.read_csv(MONITOR_PATH)
    if "pct_error" in df_mon.columns:
        high_err_pct = (df_mon["pct_error"] > 50).mean() * 100
        if high_err_pct > 20:
            st.error(f"🔴 HIGH ERROR RATE: {high_err_pct:.1f}% rows with >50% error.")
            alerts_found = True

if not alerts_found:
    st.success("✅ All systems normal — no alerts triggered")

# ── SECTION 2: MODEL KPIs ─────────────────────────────────────
st.markdown("---")
st.subheader("📊 Model Performance KPIs")

if os.path.exists(CARD_PATH):
    with open(CARD_PATH) as f:
        card = json.load(f)
    best  = card.get("best_model", {})
    ens_w = card.get("ensemble_weights", {})

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("RMSLE",       f"{best.get('rmsle', 0):.5f}")
    c2.metric("R²",          f"{best.get('r2',    0):.4f}")
    c3.metric("MAE",         f"{best.get('mae',   0):.2f}")
    c4.metric("MAPE",        f"{best.get('mape',  0):.2f}%")
    c5.metric("LGBM Weight", f"{ens_w.get('lgbm', 1.0):.0%}")

    if ens_w:
        models  = [k for k in ["lgbm", "prophet", "lstm"] if k in ens_w]
        weights = [ens_w[k] for k in models]
        fig, ax = plt.subplots(figsize=(6, 2.5))
        colors  = ["#3498db", "#e74c3c", "#2ecc71"][:len(models)]
        ax.barh(models, weights, color=colors, alpha=0.85)
        ax.set_title("Ensemble Weights")
        ax.set_xlim(0, 1)
        for i, v in enumerate(weights):
            ax.text(v + 0.01, i, f"{v:.2f}", va="center")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
else:
    st.info("Model card not found — run train_model.py first.")

# ── SECTION 3: MODEL COMPARISON + CHAMPION vs CHALLENGER ──────
st.markdown("---")
st.subheader("🏆 Model Comparison")

if os.path.exists(RESULTS_PATH):
    df_res = pd.read_csv(RESULTS_PATH)
    cols   = [c for c in ["model", "rmsle", "rmse", "mae", "mape", "wape", "r2"]
              if c in df_res.columns]
    df_show = df_res[cols].sort_values("rmsle") if "rmsle" in df_res.columns else df_res
    st.dataframe(df_show, use_container_width=True)

    # ── Champion vs Challenger 3-Gate Visual ──────────────────
    st.markdown("#### ⚔️ Champion vs Challenger — Promotion Gates")

    if "rmsle" in df_res.columns and len(df_res) >= 2:
        df_sorted  = df_res.sort_values("rmsle").reset_index(drop=True)
        champion   = df_sorted.iloc[0]
        challenger = df_sorted.iloc[1]

        champ_rmsle  = float(champion.get("rmsle", 0))
        chal_rmsle   = float(challenger.get("rmsle", champ_rmsle))
        chal_r2      = float(challenger.get("r2",   0))
        champ_wape   = float(champion.get("wape",   100))
        chal_wape    = float(challenger.get("wape",  100))

        rmsle_improvement = champ_rmsle - chal_rmsle

        gate1_pass = rmsle_improvement >= 0.005
        gate2_pass = chal_r2 >= 0.80
        gate3_pass = chal_wape <= champ_wape
        promoted   = gate1_pass and gate2_pass and gate3_pass

        # ── Champion / Challenger header cards ────────────────
        col_champ, col_chal = st.columns(2)
        col_champ.markdown(
            f"""<div style='background:#0d1b4b;padding:14px;border-radius:8px;
            border-left:4px solid #3498db'>
            <b style='color:#7eb3f7;font-size:15px'>👑 Champion</b><br>
            <span style='color:#ffffff;font-size:20px;font-weight:700'>{champion['model']}</span><br>
            <span style='color:#c0d8ff;font-size:13px'>
            RMSLE: <b>{champ_rmsle:.5f}</b> &nbsp;|&nbsp; WAPE: <b>{champ_wape:.2f}%</b>
            </span></div>""",
            unsafe_allow_html=True
        )
        col_chal.markdown(
            f"""<div style='background:#2a0a0a;padding:14px;border-radius:8px;
            border-left:4px solid #e74c3c'>
            <b style='color:#f8a0a0;font-size:15px'>🥊 Challenger</b><br>
            <span style='color:#ffffff;font-size:20px;font-weight:700'>{challenger['model']}</span><br>
            <span style='color:#ffd0d0;font-size:13px'>
            RMSLE: <b>{chal_rmsle:.5f}</b> &nbsp;|&nbsp;
            R²: <b>{chal_r2:.4f}</b> &nbsp;|&nbsp;
            WAPE: <b>{chal_wape:.2f}%</b>
            </span></div>""",
            unsafe_allow_html=True
        )

        st.markdown("<br>", unsafe_allow_html=True)

        # ── 3 Gate cards ──────────────────────────────────────
        def _gate_card(col, title, passed, metric_label, metric_val, threshold_label):
            border = "#2ecc71" if passed else "#e74c3c"
            icon   = "✅" if passed else "❌"
            status = "PASS"  if passed else "FAIL"
            status_color = "#2ecc71" if passed else "#e74c3c"
            col.markdown(
                f"""<div style='background:#0a0a1a;padding:18px;border-radius:10px;
                border:2px solid {border};text-align:center'>
                <div style='font-size:32px'>{icon}</div>
                <div style='color:{border};font-weight:700;font-size:15px;margin-top:4px'>{title}</div>
                <div style='color:{status_color};font-size:26px;font-weight:800;margin:6px 0'>{status}</div>
                <hr style='border-color:{border};margin:10px 0;opacity:0.4'>
                <div style='color:#dddddd;font-size:13px'>{metric_label}</div>
                <div style='color:#ffffff;font-size:16px;font-weight:700;margin-top:4px'>{metric_val}</div>
                <div style='color:#aaaaaa;font-size:11px;margin-top:6px'>{threshold_label}</div>
                </div>""",
                unsafe_allow_html=True
            )

        g1, g2, g3 = st.columns(3)
        _gate_card(
            g1, "Gate 1 — RMSLE", gate1_pass,
            "Improvement",
            f"{rmsle_improvement:+.5f}",
            "Threshold: ≥ 0.005"
        )
        _gate_card(
            g2, "Gate 2 — R²", gate2_pass,
            "Challenger R²",
            f"{chal_r2:.4f}",
            "Threshold: ≥ 0.80"
        )
        _gate_card(
            g3, "Gate 3 — WAPE", gate3_pass,
            f"Challenger {chal_wape:.2f}% vs Champion {champ_wape:.2f}%",
            "Lower is better",
            "Challenger WAPE ≤ Champion WAPE"
        )

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Final verdict ──────────────────────────────────────
        gates_passed = sum([gate1_pass, gate2_pass, gate3_pass])
        if promoted:
            st.success(
                f"🚀 **PROMOTED** — Challenger '{challenger['model']}' passes all 3 gates "
                f"({gates_passed}/3). Ready to replace champion."
            )
        else:
            failed = []
            if not gate1_pass: failed.append("RMSLE improvement insufficient")
            if not gate2_pass: failed.append(f"R² {chal_r2:.4f} below 0.80")
            if not gate3_pass: failed.append("WAPE worse than champion")
            st.error(
                f"🔒 **REJECTED** — Challenger '{challenger['model']}' fails "
                f"{3 - gates_passed}/3 gate(s): {' | '.join(failed)}"
            )
    else:
        st.info("Need at least 2 models in results to run Champion vs Challenger comparison.")
else:
    st.info("No experiment results found.")

# ── SECTION 4: FORECAST ACCURACY CHARTS ──────────────────────
st.markdown("---")
st.subheader("📈 Forecast vs Actual")

if os.path.exists(MONITOR_PATH):
    df_mon = pd.read_csv(MONITOR_PATH, parse_dates=["date"])
    ca, cb = st.columns(2)

    with ca:
        daily = df_mon.groupby("date").agg(
            actual    =("actual_sales",    "sum"),
            predicted =("predicted_sales", "sum"),
        ).reset_index()
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(daily["date"], daily["actual"],    label="Actual",
                linewidth=1.2, color="steelblue")
        ax.plot(daily["date"], daily["predicted"], label="Predicted",
                linewidth=1.2, color="coral", linestyle="--")
        ax.set_title("Total Daily Sales — Actual vs Predicted")
        ax.legend()
        plt.tight_layout()
        st.pyplot(fig); plt.close(fig)

    with cb:
        if "pct_error" in df_mon.columns:
            err = df_mon["pct_error"].clip(0, 200)
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.hist(err, bins=60, color="mediumpurple", alpha=0.8)
            ax.axvline(err.mean(), color="red", linestyle="--",
                       label=f"Mean={err.mean():.1f}%")
            ax.set_title("Prediction % Error Distribution")
            ax.set_xlabel("% Error (capped 200%)")
            ax.legend()
            plt.tight_layout()
            st.pyplot(fig); plt.close(fig)

    cc, cd = st.columns(2)
    fam_err = df_mon.groupby("family")["abs_error"].mean().sort_values()

    with cc:
        fig, ax = plt.subplots(figsize=(8, 5))
        fam_err.head(10).plot(kind="barh", ax=ax, color="seagreen", alpha=0.85)
        ax.set_title("Best 10 Families (lowest MAE)")
        plt.tight_layout()
        st.pyplot(fig); plt.close(fig)

    with cd:
        fig, ax = plt.subplots(figsize=(8, 5))
        fam_err.tail(10).plot(kind="barh", ax=ax, color="tomato", alpha=0.85)
        ax.set_title("Worst 10 Families (highest MAE)")
        plt.tight_layout()
        st.pyplot(fig); plt.close(fig)
else:
    st.info("Monitor scores not found — run training pipeline first.")

# ── SECTION 5: CONFIDENCE BANDS ───────────────────────────────
st.markdown("---")
st.subheader("🎯 Forecast Confidence Distribution")

if os.path.exists(MONITOR_PATH):
    df_mon2 = pd.read_csv(MONITOR_PATH)
    if "confidence_band" in df_mon2.columns:
        ce, cf = st.columns(2)
        with ce:
            bc   = df_mon2["confidence_band"].value_counts()
            cmap = {"HIGH": "#2ecc71", "MEDIUM": "#f39c12", "LOW": "#e74c3c"}
            fig, ax = plt.subplots(figsize=(5, 3.5))
            ax.bar(bc.index, bc.values,
                   color=[cmap.get(b, "gray") for b in bc.index], alpha=0.85)
            ax.set_title("Confidence Band Distribution")
            plt.tight_layout()
            st.pyplot(fig); plt.close(fig)
        with cf:
            if "rule_triggered" in df_mon2.columns:
                rc = df_mon2["rule_triggered"].dropna().value_counts()
                if len(rc):
                    st.write("**Business Rules Triggered:**")
                    st.bar_chart(rc)
                else:
                    st.success("No business rules triggered in validation set.")

# ── SECTION 6: PSI DRIFT ──────────────────────────────────────
st.markdown("---")
st.subheader("📉 Feature Drift Report (PSI)")

if os.path.exists(DRIFT_PATH):
    df_psi = pd.read_csv(DRIFT_PATH)
    if "drift_score" in df_psi.columns:
        def _flag(v):
            if v >= PSI_HIGH:     return "🔴 CRITICAL"
            if v >= PSI_MODERATE: return "🟡 MODERATE"
            return "🟢 OK"
        df_psi["status"] = df_psi["drift_score"].apply(_flag)
        cg, ch = st.columns(2)
        with cg:
            st.dataframe(df_psi.head(15), use_container_width=True)
        with ch:
            fig, ax = plt.subplots(figsize=(7, 5))
            clrs = ["#e74c3c" if v >= PSI_HIGH else "#f39c12" if v >= PSI_MODERATE
                    else "#2ecc71" for v in df_psi["drift_score"].head(15)]
            ax.barh(df_psi["feature"].head(15), df_psi["drift_score"].head(15),
                    color=clrs, alpha=0.85)
            ax.axvline(PSI_MODERATE, color="orange", linestyle="--",
                       linewidth=1, label="Moderate (0.10)")
            ax.axvline(PSI_HIGH,     color="red",    linestyle="--",
                       linewidth=1, label="Critical (0.20)")
            ax.set_title("Feature PSI Scores")
            ax.legend(fontsize=8)
            plt.tight_layout()
            st.pyplot(fig); plt.close(fig)
else:
    st.warning("Feature drift report not found.")

# ── SECTION 7: FEATURE IMPORTANCE + SHAP ─────────────────────
st.markdown("---")
st.subheader("🔍 Feature Importance (LightGBM + SHAP)")

if os.path.exists(CARD_PATH):
    with open(CARD_PATH) as f:
        card = json.load(f)
    ci, cj = st.columns(2)
    with ci:
        fi = card.get("feature_importances", {})
        if fi:
            fi_df = (pd.DataFrame(fi.items(), columns=["feature", "importance"])
                     .sort_values("importance").tail(15))
            fig, ax = plt.subplots(figsize=(7, 5))
            ax.barh(fi_df["feature"], fi_df["importance"], color="steelblue", alpha=0.85)
            ax.set_title("Feature Importance (Gain) — Top 15")
            plt.tight_layout()
            st.pyplot(fig); plt.close(fig)
    with cj:
        shap = card.get("shap_top_features", {})
        if shap:
            shap_df = (pd.DataFrame(shap.items(), columns=["feature", "mean_shap"])
                       .sort_values("mean_shap").tail(15))
            fig, ax = plt.subplots(figsize=(7, 5))
            ax.barh(shap_df["feature"], shap_df["mean_shap"], color="coral", alpha=0.85)
            ax.set_title("SHAP Values — Top 15 Features")
            plt.tight_layout()
            st.pyplot(fig); plt.close(fig)

# ── SECTION 8: RECENT PREDICTIONS ────────────────────────────
st.markdown("---")
st.subheader("📋 Recent API Predictions")
if os.path.exists(LOG_PATH):
    st.dataframe(pd.read_csv(LOG_PATH).tail(20), use_container_width=True)
else:
    st.info("Prediction logs written by Render API — run locally to see logs.")