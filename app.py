import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from model_utils import load_and_process_data, train_models, NUM_COLS_TO_SCALE

FILE_PATH = "data.csv"

st.set_page_config(
    page_title="Job Risk AI",
    layout="wide",
    page_icon="🤖",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
#MainMenu, footer, header {visibility: hidden;}

/* Metric cards */
.stat-card {
    background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
    border: 1px solid #334155;
    border-radius: 14px;
    padding: 22px 16px;
    text-align: center;
}
.stat-value {
    font-size: 2.2rem;
    font-weight: 800;
    color: #818cf8;
    line-height: 1;
}
.stat-label {
    font-size: 0.8rem;
    color: #94a3b8;
    margin-top: 6px;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}

/* Model cards */
.model-card {
    border-radius: 12px;
    padding: 18px;
    border: 1px solid #334155;
    background: #1e293b;
    margin-bottom: 8px;
    text-align: center;
}
.model-name   { font-size: 0.95rem; font-weight: 700; color: #e2e8f0; }
.model-acc    { font-size: 1.8rem;  font-weight: 800; margin-top: 4px; }
.model-badge  {
    display: inline-block;
    font-size: 0.7rem;
    padding: 2px 8px;
    border-radius: 20px;
    margin-top: 6px;
    font-weight: 600;
    letter-spacing: 0.05em;
}

/* Risk result card */
.risk-card {
    border-radius: 16px;
    padding: 28px 24px;
    text-align: center;
    margin-top: 12px;
}
.risk-card-title { font-size: 1.7rem; font-weight: 800; margin-bottom: 10px; }
.risk-card-desc  { font-size: 0.97rem; line-height: 1.7; color: #e2e8f0; }
.risk-card-tip   {
    margin-top: 14px;
    font-size: 0.85rem;
    color: #94a3b8;
    border-top: 1px solid rgba(255,255,255,0.1);
    padding-top: 12px;
}

/* Section titles */
.section-title {
    font-size: 1.15rem;
    font-weight: 700;
    color: #818cf8;
    border-left: 4px solid #6366f1;
    padding-left: 10px;
    margin: 20px 0 12px 0;
}

/* Divider */
.fancy-divider {
    border: none;
    height: 1px;
    background: linear-gradient(90deg, transparent, #6366f1, transparent);
    margin: 28px 0;
}
</style>
""", unsafe_allow_html=True)


# ── Helpers ────────────────────────────────────────────────────────────────────
RISK_CONFIG = {
    0: {
        "label": "🟢  VERY SAFE",
        "range": "0 – 20 %",
        "desc":  "Automation is unlikely. This role relies on creativity, empathy, or complex judgment that AI cannot replicate.",
        "tip":   "💡 Keep building domain expertise — your role is well-protected.",
        "bg":    "linear-gradient(135deg,#052e16,#065f46)",
        "border":"#10b981",
        "acc":   "#10b981",
        "gauge_mid": 10,
    },
    1: {
        "label": "🟡  SAFE",
        "range": "20 – 40 %",
        "desc":  "AI will likely act as a productivity tool here, not a replacement. Human oversight and decision-making remain central.",
        "tip":   "💡 Learn to work alongside AI tools — that skill amplifies your value.",
        "bg":    "linear-gradient(135deg,#1a2e05,#3f6212)",
        "border":"#84cc16",
        "acc":   "#a3e635",
        "gauge_mid": 30,
    },
    2: {
        "label": "🟠  MODERATE",
        "range": "40 – 60 %",
        "desc":  "A hybrid role. Routine, repetitive sub-tasks are automatable, but complex decisions and client interactions still need humans.",
        "tip":   "💡 Identify which parts of your job are routine — upskill into the non-routine ones.",
        "bg":    "linear-gradient(135deg,#431407,#7c2d12)",
        "border":"#f97316",
        "acc":   "#fb923c",
        "gauge_mid": 50,
    },
    3: {
        "label": "🔴  HIGH RISK",
        "range": "60 – 80 %",
        "desc":  "Significant automation is expected in the near future. Roles in this band are being actively disrupted.",
        "tip":   "⚠️ Act now — explore adjacent roles or specialise in a niche AI cannot easily replicate.",
        "bg":    "linear-gradient(135deg,#450a0a,#7f1d1d)",
        "border":"#ef4444",
        "acc":   "#f87171",
        "gauge_mid": 70,
    },
    4: {
        "label": "⛔  CRITICAL RISK",
        "range": "80 – 100 %",
        "desc":  "Full automation is highly likely. This role consists largely of rule-based or repetitive tasks that AI already handles well.",
        "tip":   "🚨 Pivoting to adjacent roles that require human judgment is strongly recommended.",
        "bg":    "linear-gradient(135deg,#3b0764,#4a044e)",
        "border":"#a855f7",
        "acc":   "#c084fc",
        "gauge_mid": 90,
    },
}

def grade_label(risk):
    if risk <= 20:   return "Very Safe"
    elif risk <= 40: return "Safe"
    elif risk <= 60: return "Moderate"
    elif risk <= 80: return "High Risk"
    else:            return "Critical"

def stat_card(value, label):
    return f"""
    <div class="stat-card">
        <div class="stat-value">{value}</div>
        <div class="stat-label">{label}</div>
    </div>"""

def model_card(name, acc, best=False):
    color   = "#f59e0b" if best else "#818cf8"
    badge   = "🏆 BEST" if best else "MODEL"
    bg_clr  = "#1c1917" if best else "#1e293b"
    border  = "#f59e0b" if best else "#334155"
    return f"""
    <div class="model-card" style="background:{bg_clr};border-color:{border};">
        <div class="model-name">{name}</div>
        <div class="model-acc" style="color:{color};">{acc:.1f}%</div>
        <span class="model-badge" style="background:{color}22;color:{color};">{badge}</span>
    </div>"""

def risk_result_card(cfg):
    return f"""
    <div class="risk-card" style="background:{cfg['bg']};border:2px solid {cfg['border']};">
        <div class="risk-card-title" style="color:{cfg['acc']};">{cfg['label']}</div>
        <div style="color:{cfg['border']};font-size:0.85rem;font-weight:600;margin-bottom:8px;">
            Risk Range: {cfg['range']}
        </div>
        <div class="risk-card-desc">{cfg['desc']}</div>
        <div class="risk-card-tip">{cfg['tip']}</div>
    </div>"""


# ── App ────────────────────────────────────────────────────────────────────────
def main():
    # ── Hero ──────────────────────────────────────────────────────────────────
    st.markdown("""
    <h1 style="text-align:center;font-size:2.8rem;font-weight:900;
               background:linear-gradient(90deg,#818cf8,#c084fc,#f472b6);
               -webkit-background-clip:text;-webkit-text-fill-color:transparent;
               margin-bottom:4px;">
        🤖 AI Job Market Risk Analyzer
    </h1>
    <p style="text-align:center;color:#94a3b8;font-size:1rem;margin-bottom:28px;">
        CAI3101 · AAST · End-to-End Machine Learning Project
    </p>
    """, unsafe_allow_html=True)

    # Load data
    df_original, df_processed, feature_names, encoders, scaler = load_and_process_data(FILE_PATH)

    if df_original is None:
        st.error(f"Cannot load `{FILE_PATH}`. Make sure the file is in the same folder.")
        st.stop()

    # Prepare X/y and train models once (cached)
    X = df_processed[feature_names]
    y = df_processed["Risk_Grade"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    trained_models = train_models(X_train, y_train)

    results = [
        {"Model": name, "Accuracy": accuracy_score(y_test, m.predict(X_test)) * 100}
        for name, m in trained_models.items()
    ]
    results_df = pd.DataFrame(results).sort_values("Accuracy", ascending=False).reset_index(drop=True)
    best_name  = results_df.iloc[0]["Model"]

    # Hero stats
    c1, c2, c3, c4 = st.columns(4)
    for col, val, lbl in zip(
        [c1, c2, c3, c4],
        ["30 K", "8", "4", "5"],
        ["Jobs in Dataset", "Features Used", "ML Models", "Risk Levels"],
    ):
        col.markdown(stat_card(val, lbl), unsafe_allow_html=True)

    st.markdown('<hr class="fancy-divider">', unsafe_allow_html=True)

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab1, tab2, tab3 = st.tabs(["📊  Data Explorer", "⚙️  Model Lab", "🎯  Risk Analyzer"])

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 1 — Data Explorer
    # ══════════════════════════════════════════════════════════════════════════
    with tab1:
        df_viz = df_original.copy()
        df_viz["Risk Grade"] = df_original["Automation Risk (%)"].apply(grade_label)

        # Dataset overview
        col_l, col_r = st.columns([1, 1])
        with col_l:
            st.markdown('<div class="section-title">Dataset Overview</div>', unsafe_allow_html=True)
            mc1, mc2, mc3 = st.columns(3)
            mc1.metric("Rows",    f"{df_original.shape[0]:,}")
            mc2.metric("Columns", df_original.shape[1])
            mc3.metric("Missing", df_original.isnull().sum().sum())
            with st.expander("📄 View raw data (first 10 rows)"):
                st.dataframe(df_original.head(10), use_container_width=True)

        with col_r:
            st.markdown('<div class="section-title">Summary Statistics</div>', unsafe_allow_html=True)
            st.dataframe(
                df_original.describe().style.format("{:.1f}"),
                use_container_width=True, height=280,
            )

        st.markdown('<hr class="fancy-divider">', unsafe_allow_html=True)

        # ── Charts row 1 ──────────────────────────────────────────────────────
        st.markdown('<div class="section-title">Data Visualizations</div>', unsafe_allow_html=True)
        r1c1, r1c2 = st.columns(2)

        with r1c1:
            fig = px.histogram(
                df_original, x="Automation Risk (%)", nbins=25,
                title="Risk Score Distribution",
                color_discrete_sequence=["#818cf8"],
                template="plotly_dark",
            )
            fig.update_layout(bargap=0.05, title_font_size=14)
            st.plotly_chart(fig, use_container_width=True)

        with r1c2:
            grade_counts = df_viz["Risk Grade"].value_counts().reset_index()
            grade_counts.columns = ["Risk Grade", "Count"]
            order = ["Very Safe", "Safe", "Moderate", "High Risk", "Critical"]
            grade_counts["Risk Grade"] = pd.Categorical(grade_counts["Risk Grade"], categories=order, ordered=True)
            grade_counts = grade_counts.sort_values("Risk Grade")
            fig = px.pie(
                grade_counts, names="Risk Grade", values="Count",
                title="Class Balance (Risk Grades)",
                color_discrete_sequence=["#10b981","#84cc16","#f59e0b","#ef4444","#a855f7"],
                template="plotly_dark",
                hole=0.4,
            )
            fig.update_traces(textposition="inside", textinfo="percent+label")
            fig.update_layout(showlegend=False, title_font_size=14)
            st.plotly_chart(fig, use_container_width=True)

        # ── Charts row 2 ──────────────────────────────────────────────────────
        r2c1, r2c2 = st.columns(2)

        with r2c1:
            fig = px.box(
                df_viz, x="Risk Grade", y="Median Salary (USD)",
                category_orders={"Risk Grade": order},
                title="Salary Distribution by Risk Grade",
                color="Risk Grade",
                color_discrete_sequence=["#10b981","#84cc16","#f59e0b","#ef4444","#a855f7"],
                template="plotly_dark",
            )
            fig.update_layout(showlegend=False, title_font_size=14)
            st.plotly_chart(fig, use_container_width=True)

        with r2c2:
            top_ind = df_original["Industry"].value_counts().head(10).reset_index()
            top_ind.columns = ["Industry", "Count"]
            fig = px.bar(
                top_ind.sort_values("Count"), x="Count", y="Industry",
                orientation="h",
                title="Top 10 Industries by Job Count",
                color="Count",
                color_continuous_scale="Bluyl",
                template="plotly_dark",
            )
            fig.update_layout(coloraxis_showscale=False, title_font_size=14)
            st.plotly_chart(fig, use_container_width=True)

        # ── Charts row 3 ──────────────────────────────────────────────────────
        r3c1, r3c2 = st.columns(2)

        with r3c1:
            ai_risk = df_viz.groupby(["AI Impact Level","Risk Grade"]).size().reset_index(name="Count")
            fig = px.bar(
                ai_risk, x="AI Impact Level", y="Count", color="Risk Grade",
                barmode="stack",
                title="AI Impact Level vs Risk Grade",
                category_orders={"Risk Grade": order},
                color_discrete_sequence=["#10b981","#84cc16","#f59e0b","#ef4444","#a855f7"],
                template="plotly_dark",
            )
            fig.update_layout(title_font_size=14)
            st.plotly_chart(fig, use_container_width=True)

        with r3c2:
            num_cols = df_original.select_dtypes(include=[np.number]).columns.tolist()
            corr = df_original[num_cols].corr()
            fig = px.imshow(
                corr, text_auto=".2f", aspect="auto",
                title="Correlation Heatmap",
                color_continuous_scale="RdBu_r",
                template="plotly_dark",
            )
            fig.update_layout(title_font_size=14)
            st.plotly_chart(fig, use_container_width=True)

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 2 — Model Lab
    # ══════════════════════════════════════════════════════════════════════════
    with tab2:
        st.markdown('<div class="section-title">Model Performance</div>', unsafe_allow_html=True)

        # Model cards
        card_cols = st.columns(len(results_df))
        for col, (_, row) in zip(card_cols, results_df.iterrows()):
            col.markdown(
                model_card(row["Model"], row["Accuracy"], row["Model"] == best_name),
                unsafe_allow_html=True,
            )

        st.markdown('<hr class="fancy-divider">', unsafe_allow_html=True)

        # Bar chart
        col_chart, col_info = st.columns([2, 1])

        with col_chart:
            fig = px.bar(
                results_df, x="Model", y="Accuracy",
                title="Accuracy Comparison on 20% Test Set",
                color="Accuracy",
                color_continuous_scale="Aggrnyl",
                text=results_df["Accuracy"].apply(lambda x: f"{x:.1f}%"),
                template="plotly_dark",
            )
            fig.update_traces(textposition="outside")
            fig.update_layout(
                yaxis_range=[0, 105],
                coloraxis_showscale=False,
                title_font_size=14,
            )
            st.plotly_chart(fig, use_container_width=True)

        with col_info:
            st.markdown('<div class="section-title">Algorithm Guide</div>', unsafe_allow_html=True)
            info = {
                "Random Forest":        ("🌲", "Ensemble of 100 trees. Reduces variance by voting. Best general-purpose model."),
                "Neural Network (MLP)": ("🧠", "Two hidden layers. Learns non-linear patterns. Needs feature scaling."),
                "Decision Tree":        ("🌿", "If-then rules. Easy to interpret but prone to overfitting."),
                "Naive Bayes":          ("📐", "Probabilistic baseline. Fast. Assumes feature independence."),
            }
            for name, (icon, desc) in info.items():
                with st.expander(f"{icon}  {name}"):
                    st.caption(desc)

        # Train/test split info
        st.markdown('<div class="section-title">Data Split</div>', unsafe_allow_html=True)
        sp1, sp2, sp3 = st.columns(3)
        sp1.metric("Training Samples", f"{X_train.shape[0]:,}",  "80%")
        sp2.metric("Testing Samples",  f"{X_test.shape[0]:,}",   "20%")
        sp3.metric("Features Used",    X_train.shape[1])

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 3 — Risk Analyzer
    # ══════════════════════════════════════════════════════════════════════════
    with tab3:

        def get_options(col_name):
            return sorted(encoders[col_name].classes_) if col_name in encoders else ["Unknown"]

        col_form, col_result = st.columns([1, 1], gap="large")

        with col_form:
            st.markdown('<div class="section-title">Job Parameters</div>', unsafe_allow_html=True)

            with st.container():
                selected_job      = st.selectbox("Job Title",          get_options("Job Title"))
                selected_industry = st.selectbox("Industry",           get_options("Industry"))
                selected_status   = st.selectbox("Job Status",         get_options("Job Status"))
                selected_edu      = st.selectbox("Required Education", get_options("Required Education"))
                selected_loc      = st.selectbox("Location",           get_options("Location"))

            st.markdown('<div class="section-title">Numeric Details</div>', unsafe_allow_html=True)

            salary      = st.slider("Median Salary (USD)",         10_000, 200_000, 70_000, 1_000,
                                    format="$%d")
            experience  = st.slider("Experience Required (Years)", 0, 25, 5)
            remote      = st.slider("Remote Work Ratio (%)",       0, 100, 30)

            st.markdown('<div class="section-title">Model</div>', unsafe_allow_html=True)
            model_choice  = st.selectbox("Choose Model", list(trained_models.keys()))
            active_model  = trained_models[model_choice]

            analyze_btn = st.button("🎯  Analyze Risk", type="primary", use_container_width=True)

        with col_result:
            st.markdown('<div class="section-title">Result</div>', unsafe_allow_html=True)

            if analyze_btn:
                try:
                    input_dict = {
                        "Job Title":                  encoders["Job Title"].transform([selected_job])[0],
                        "Industry":                   encoders["Industry"].transform([selected_industry])[0],
                        "Job Status":                 encoders["Job Status"].transform([selected_status])[0],
                        "Required Education":         encoders["Required Education"].transform([selected_edu])[0],
                        "Location":                   encoders["Location"].transform([selected_loc])[0],
                        "Median Salary (USD)":        salary,
                        "Experience Required (Years)": experience,
                        "Remote Work Ratio (%)":      remote,
                    }

                    input_df   = pd.DataFrame([input_dict])
                    final_df   = input_df[feature_names].copy()
                    scale_cols = [c for c in NUM_COLS_TO_SCALE if c in final_df.columns]
                    if scale_cols:
                        final_df[scale_cols] = scaler.transform(final_df[scale_cols])

                    grade = active_model.predict(final_df)[0]
                    cfg   = RISK_CONFIG[grade]

                    # Gauge chart
                    fig_gauge = go.Figure(go.Indicator(
                        mode  = "gauge+number+delta",
                        value = cfg["gauge_mid"],
                        number= {"suffix": "%", "font": {"size": 36, "color": cfg["acc"]}},
                        title = {"text": "Estimated Automation Risk",
                                 "font": {"size": 14, "color": "#94a3b8"}},
                        gauge = {
                            "axis": {"range": [0, 100], "tickcolor": "#475569",
                                     "tickwidth": 1, "tickfont": {"color": "#94a3b8"}},
                            "bar":  {"color": cfg["acc"], "thickness": 0.25},
                            "bgcolor": "#1e293b",
                            "borderwidth": 0,
                            "steps": [
                                {"range": [0,  20], "color": "#064e3b"},
                                {"range": [20, 40], "color": "#1a4731"},
                                {"range": [40, 60], "color": "#431407"},
                                {"range": [60, 80], "color": "#450a0a"},
                                {"range": [80,100], "color": "#3b0764"},
                            ],
                            "threshold": {
                                "line": {"color": "white", "width": 3},
                                "thickness": 0.8,
                                "value": cfg["gauge_mid"],
                            },
                        },
                    ))
                    fig_gauge.update_layout(
                        height=260,
                        margin=dict(t=50, b=0, l=20, r=20),
                        paper_bgcolor="#0f172a",
                        font_color="#e2e8f0",
                    )
                    st.plotly_chart(fig_gauge, use_container_width=True)

                    # Risk card
                    st.markdown(risk_result_card(cfg), unsafe_allow_html=True)

                    # Confidence scores
                    if hasattr(active_model, "predict_proba"):
                        probs = active_model.predict_proba(final_df)[0]
                        class_names = ["Very Safe", "Safe", "Moderate", "High Risk", "Critical"]
                        prob_df = pd.DataFrame({
                            "Class": class_names,
                            "Probability": (probs * 100).round(1),
                        })
                        bar_colors = ["#10b981","#84cc16","#f59e0b","#ef4444","#a855f7"]
                        fig_prob = px.bar(
                            prob_df, x="Probability", y="Class",
                            orientation="h",
                            title="Model Confidence per Class (%)",
                            color="Class",
                            color_discrete_sequence=bar_colors,
                            text="Probability",
                            template="plotly_dark",
                        )
                        fig_prob.update_traces(texttemplate="%{text}%", textposition="outside")
                        fig_prob.update_layout(
                            showlegend=False,
                            xaxis_range=[0, 110],
                            title_font_size=13,
                            height=260,
                            margin=dict(t=40, b=10),
                        )
                        st.plotly_chart(fig_prob, use_container_width=True)

                except Exception as e:
                    st.error(f"Prediction error: {e}")
            else:
                st.markdown("""
                <div style="text-align:center;padding:60px 20px;color:#475569;">
                    <div style="font-size:3rem;">🎯</div>
                    <div style="font-size:1rem;margin-top:12px;">
                        Fill in the job parameters and click<br>
                        <strong style="color:#818cf8;">Analyze Risk</strong> to see the prediction.
                    </div>
                </div>
                """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
