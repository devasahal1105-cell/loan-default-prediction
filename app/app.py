import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(page_title="Loan Default Predictor", page_icon="💳", layout="wide")

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "..", "models")

@st.cache_resource
def load_artifacts():
    model         = joblib.load(os.path.join(MODELS_DIR, "best_model.pkl"))
    scaler        = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
    le_dict       = joblib.load(os.path.join(MODELS_DIR, "label_encoders.pkl"))
    feature_names = joblib.load(os.path.join(MODELS_DIR, "feature_names.pkl"))
    summary       = joblib.load(os.path.join(MODELS_DIR, "training_summary.pkl"))
    return model, scaler, le_dict, feature_names, summary

model, scaler, le_dict, feature_names, summary = load_artifacts()

st.markdown("""
<style>
.main-header{background:linear-gradient(135deg,#1a1a2e 0%,#16213e 50%,#0f3460 100%);padding:2.5rem;border-radius:14px;margin-bottom:1.5rem;text-align:center;}
.main-header h1{color:#e94560;font-size:2.6rem;margin:0;}
.main-header p{color:#a8b2d8;font-size:1.05rem;margin:0.4rem 0 0;}
.section-header{font-size:1.1rem;font-weight:700;color:#0f3460;border-left:4px solid #e94560;padding-left:10px;margin:1.2rem 0 0.8rem;}
.result-safe{background:linear-gradient(135deg,#0d4f35,#155d27);border:2px solid #28a745;border-radius:14px;padding:1.5rem;text-align:center;}
.result-risk{background:linear-gradient(135deg,#5c1010,#7b1515);border:2px solid #e94560;border-radius:14px;padding:1.5rem;text-align:center;}
.result-safe h2,.result-risk h2{font-size:1.8rem;margin:0;}
.result-safe p,.result-risk p{color:#ddd;margin:0.4rem 0 0;font-size:0.95rem;}
.tip-box{background:#fffbe6;border-left:4px solid #f0c040;border-radius:6px;padding:0.6rem 1rem;font-size:0.85rem;color:#555;}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="main-header">
    <h1>💳 Loan Default Predictor</h1>
    <p>AI-powered credit risk assessment · XGBoost · 94.83% AUC</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("## 📊 Model Performance")
    for name, metrics in summary["results"].items():
        icon = "🏆" if name == summary["best_model_name"] else "📈"
        st.metric(label=f"{icon} {name}", value=f"{metrics['auc']:.2%} AUC", delta=f"Acc: {metrics['accuracy']:.2%}")
    st.divider()
    st.markdown("## 💡 Try an Example")
    st.markdown("Click a profile to auto-fill the form:")

    examples = {
        "✅ Safe Applicant": {"person_age":35,"person_income":80000,"person_home_ownership":"MORTGAGE","person_emp_length":10,"loan_intent":"HOMEIMPROVEMENT","loan_grade":"A","loan_amnt":8000,"loan_int_rate":7.5,"loan_percent_income":0.10,"cb_person_default_on_file":"N","cb_person_cred_hist_length":12},
        "⚠️ Risky Applicant": {"person_age":22,"person_income":18000,"person_home_ownership":"RENT","person_emp_length":1,"loan_intent":"PERSONAL","loan_grade":"F","loan_amnt":14000,"loan_int_rate":22.5,"loan_percent_income":0.78,"cb_person_default_on_file":"Y","cb_person_cred_hist_length":2},
        "🔶 Borderline Case": {"person_age":28,"person_income":42000,"person_home_ownership":"RENT","person_emp_length":4,"loan_intent":"EDUCATION","loan_grade":"C","loan_amnt":10000,"loan_int_rate":13.5,"loan_percent_income":0.24,"cb_person_default_on_file":"N","cb_person_cred_hist_length":5},
    }
    for label, vals in examples.items():
        if st.button(label, use_container_width=True):
            for k, v in vals.items():
                st.session_state[k] = v

    st.divider()
    st.markdown("<div class='tip-box'><b>Tip:</b> Loan grade A is best, G is worst. Higher interest rates and loan % of income increase default risk.</div>", unsafe_allow_html=True)

def sv(key, default):
    return st.session_state.get(key, default)

st.markdown("<div class='section-header'>📝 Applicant Information</div>", unsafe_allow_html=True)
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**👤 Personal Details**")
    person_age              = st.slider("Age", 18, 80, sv("person_age", 30), help="Applicant's current age")
    person_income           = st.number_input("Annual Income ($)", 5000, 500000, sv("person_income", 50000), step=1000, help="Total yearly income before tax")
    person_home_ownership   = st.selectbox("Home Ownership", ["RENT","OWN","MORTGAGE","OTHER"], index=["RENT","OWN","MORTGAGE","OTHER"].index(sv("person_home_ownership","RENT")), help="Current housing situation")
    person_emp_length       = st.slider("Employment Length (years)", 0, 40, sv("person_emp_length", 5), help="Years at current job")

with col2:
    st.markdown("**💰 Loan Details**")
    loan_intent   = st.selectbox("Loan Purpose", ["PERSONAL","EDUCATION","MEDICAL","VENTURE","HOMEIMPROVEMENT","DEBTCONSOLIDATION"], index=["PERSONAL","EDUCATION","MEDICAL","VENTURE","HOMEIMPROVEMENT","DEBTCONSOLIDATION"].index(sv("loan_intent","PERSONAL")), help="What the loan will be used for")
    loan_grade    = st.selectbox("Loan Grade", ["A","B","C","D","E","F","G"], index=["A","B","C","D","E","F","G"].index(sv("loan_grade","B")), help="A = best creditworthiness, G = worst")
    loan_amnt     = st.number_input("Loan Amount ($)", 500, 50000, sv("loan_amnt", 10000), step=500, help="Total loan amount requested")
    loan_int_rate = st.slider("Interest Rate (%)", 5.0, 25.0, float(sv("loan_int_rate", 12.0)), step=0.1, help="Annual interest rate")

with col3:
    st.markdown("**📋 Credit History**")
    loan_percent_income          = st.slider("Loan as % of Income", 0.0, 1.0, float(sv("loan_percent_income", 0.20)), step=0.01, format="%.2f", help="Loan amount / annual income")
    cb_person_default_on_file    = st.selectbox("Prior Default on File?", ["N","Y"], index=["N","Y"].index(sv("cb_person_default_on_file","N")), help="Has the applicant defaulted before?")
    cb_person_cred_hist_length   = st.slider("Credit History Length (years)", 0, 30, sv("cb_person_cred_hist_length", 5), help="How long they have had credit")

st.divider()

if st.button("🔍 Predict Loan Default Risk", use_container_width=True, type="primary"):
    raw = {
        "person_age": person_age, "person_income": person_income,
        "person_home_ownership": person_home_ownership, "person_emp_length": float(person_emp_length),
        "loan_intent": loan_intent, "loan_grade": loan_grade,
        "loan_amnt": loan_amnt, "loan_int_rate": loan_int_rate,
        "loan_percent_income": loan_percent_income,
        "cb_person_default_on_file": cb_person_default_on_file,
        "cb_person_cred_hist_length": cb_person_cred_hist_length,
    }
    df_input = pd.DataFrame([raw])
    for col in ['person_home_ownership','loan_intent','loan_grade','cb_person_default_on_file']:
        le  = le_dict[col]
        val = df_input[col].iloc[0]
        df_input[col] = le.transform([val])[0] if val in le.classes_ else 0
    df_input  = df_input[feature_names]
    df_scaled = scaler.transform(df_input)
    pred      = model.predict(df_scaled)[0]
    prob      = model.predict_proba(df_scaled)[0]
    risk_pct  = prob[1] * 100

    st.markdown("<div class='section-header'>🎯 Prediction Result</div>", unsafe_allow_html=True)
    r1, r2, r3 = st.columns([2, 1.2, 1.8])

    with r1:
        if pred == 0:
            st.markdown('<div class="result-safe"><h2>✅ Low Default Risk</h2><p>This applicant is likely to repay the loan successfully.</p></div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="result-risk"><h2>⚠️ High Default Risk</h2><p>This applicant has a high probability of defaulting on the loan.</p></div>', unsafe_allow_html=True)

    with r2:
        st.metric("Default Probability", f"{risk_pct:.1f}%")
        st.metric("Safe Probability",    f"{prob[0]*100:.1f}%")
        if risk_pct < 30:   label, color = "LOW RISK",    "green"
        elif risk_pct < 60: label, color = "MEDIUM RISK", "orange"
        else:               label, color = "HIGH RISK",   "red"
        st.markdown(f"**Risk Level:** :{color}[{label}]")

    with r3:
        fig = go.Figure(go.Indicator(
            mode="gauge+number", value=risk_pct,
            number={"suffix": "%", "font": {"size": 28}},
            title={"text": "Default Risk", "font": {"size": 14}},
            gauge={
                "axis": {"range": [0, 100]},
                "bar":  {"color": "#e94560" if risk_pct > 50 else "#28a745"},
                "steps": [
                    {"range": [0,  30], "color": "#d4edda"},
                    {"range": [30, 60], "color": "#fff3cd"},
                    {"range": [60,100], "color": "#f8d7da"},
                ],
            }
        ))
        fig.update_layout(height=220, margin=dict(t=40, b=10, l=20, r=20))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("<div class='section-header'>🔎 Key Risk Factors (Feature Importance)</div>", unsafe_allow_html=True)
    if hasattr(model, "feature_importances_"):
        fi_df = pd.DataFrame({"Feature": feature_names, "Importance": model.feature_importances_}).sort_values("Importance", ascending=True).tail(8)
        fig2  = px.bar(fi_df, x="Importance", y="Feature", orientation="h",
                       color="Importance", color_continuous_scale=["#d4edda","#fff3cd","#f8d7da","#e94560"],
                       labels={"Importance": "Importance Score", "Feature": ""})
        fig2.update_layout(height=300, margin=dict(t=10,b=10,l=10,r=10), coloraxis_showscale=False,
                           plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        fig2.update_xaxes(showgrid=True, gridcolor="#eeeeee")
        fig2.update_yaxes(showgrid=False)
        st.plotly_chart(fig2, use_container_width=True)

    with st.expander("📋 View Full Applicant Summary"):
        st.dataframe(pd.DataFrame([raw]).T.rename(columns={0:"Value"}), use_container_width=True)

st.divider()
st.caption("Built with XGBoost · scikit-learn · Streamlit · Plotly | Loan Default Prediction Project")
