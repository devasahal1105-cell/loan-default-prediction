import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os

st.set_page_config(page_title="Data Insights", page_icon="📊", layout="wide")

# ── Load data ──────────────────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "..", "data", "credit_risk_dataset.csv")

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df = df.dropna()
    df = df[df['person_age'] < 100]
    df = df[df['person_emp_length'] < 60]
    df['loan_status_label'] = df['loan_status'].map({0: "No Default", 1: "Default"})
    return df

df = load_data()

# ── Styling ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.main-header{background:linear-gradient(135deg,#1a1a2e 0%,#16213e 50%,#0f3460 100%);padding:2rem;border-radius:14px;margin-bottom:1.5rem;text-align:center;}
.main-header h1{color:#e94560;font-size:2.2rem;margin:0;}
.main-header p{color:#a8b2d8;font-size:1rem;margin:0.4rem 0 0;}
.section-header{font-size:1.1rem;font-weight:700;color:#0f3460;border-left:4px solid #e94560;padding-left:10px;margin:1.5rem 0 0.8rem;}
.stat-value{font-size:2rem;font-weight:700;color:#0f3460;margin:0;}
.stat-label{font-size:0.85rem;color:#888;margin:0;}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="main-header">
    <h1>📊 Data Insights</h1>
    <p>Explore the credit risk dataset · 32,000+ applicants · Interactive charts</p>
</div>
""", unsafe_allow_html=True)

# ── Sidebar filters ────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔧 Filters")
    age_range = st.slider("Age Range", int(df.person_age.min()), int(df.person_age.max()), (20, 60))
    home_filter = st.multiselect("Home Ownership", df.person_home_ownership.unique().tolist(),
                                  default=df.person_home_ownership.unique().tolist())
    intent_filter = st.multiselect("Loan Purpose", df.loan_intent.unique().tolist(),
                                    default=df.loan_intent.unique().tolist())
    st.divider()
    st.markdown("← Go back to **Loan Default Predictor** using the sidebar navigation above")

# Apply filters
filtered = df[
    (df.person_age >= age_range[0]) &
    (df.person_age <= age_range[1]) &
    (df.person_home_ownership.isin(home_filter)) &
    (df.loan_intent.isin(intent_filter))
]

# ── KPI cards ──────────────────────────────────────────────────────────────────
st.markdown("<div class='section-header'>📋 Dataset Overview</div>", unsafe_allow_html=True)

k1, k2, k3, k4, k5 = st.columns(5)
total       = len(filtered)
defaulted   = filtered['loan_status'].sum()
default_rate= defaulted / total * 100
avg_income  = filtered['person_income'].mean()
avg_loan    = filtered['loan_amnt'].mean()

k1.metric("Total Applicants",  f"{total:,}")
k2.metric("Defaulted",         f"{defaulted:,}")
k3.metric("Default Rate",      f"{default_rate:.1f}%")
k4.metric("Avg Annual Income", f"${avg_income:,.0f}")
k5.metric("Avg Loan Amount",   f"${avg_loan:,.0f}")

st.divider()

# ── Row 1: Default by loan grade + Default by loan intent ──────────────────────
st.markdown("<div class='section-header'>📉 Default Rate by Category</div>", unsafe_allow_html=True)
c1, c2 = st.columns(2)

with c1:
    grade_df = filtered.groupby('loan_grade')['loan_status'].mean().reset_index()
    grade_df['default_rate'] = grade_df['loan_status'] * 100
    grade_df = grade_df.sort_values('loan_grade')
    fig = px.bar(grade_df, x='loan_grade', y='default_rate',
                 color='default_rate',
                 color_continuous_scale=['#28a745','#ffc107','#e94560'],
                 labels={'loan_grade':'Loan Grade','default_rate':'Default Rate (%)'},
                 title='Default Rate by Loan Grade')
    fig.update_layout(coloraxis_showscale=False, plot_bgcolor='rgba(0,0,0,0)',
                      paper_bgcolor='rgba(0,0,0,0)', height=320,
                      margin=dict(t=40,b=20,l=10,r=10))
    fig.update_yaxes(ticksuffix='%', gridcolor='#eeeeee')
    fig.update_xaxes(showgrid=False)
    st.plotly_chart(fig, use_container_width=True)

with c2:
    intent_df = filtered.groupby('loan_intent')['loan_status'].mean().reset_index()
    intent_df['default_rate'] = intent_df['loan_status'] * 100
    intent_df = intent_df.sort_values('default_rate', ascending=True)
    fig2 = px.bar(intent_df, x='default_rate', y='loan_intent', orientation='h',
                  color='default_rate',
                  color_continuous_scale=['#28a745','#ffc107','#e94560'],
                  labels={'loan_intent':'Loan Purpose','default_rate':'Default Rate (%)'},
                  title='Default Rate by Loan Purpose')
    fig2.update_layout(coloraxis_showscale=False, plot_bgcolor='rgba(0,0,0,0)',
                       paper_bgcolor='rgba(0,0,0,0)', height=320,
                       margin=dict(t=40,b=20,l=10,r=10))
    fig2.update_xaxes(ticksuffix='%', gridcolor='#eeeeee')
    fig2.update_yaxes(showgrid=False)
    st.plotly_chart(fig2, use_container_width=True)

# ── Row 2: Income distribution + Age distribution ──────────────────────────────
st.markdown("<div class='section-header'>📊 Income & Age Distribution</div>", unsafe_allow_html=True)
c3, c4 = st.columns(2)

with c3:
    income_data = filtered[filtered.person_income < 200000].copy()
    fig3 = px.histogram(income_data,
                         x='person_income',
                         nbins=50, opacity=0.75,
                         color_discrete_sequence=['#4e89e8'],
                         labels={'person_income':'Annual Income ($)'},
                         title='Income Distribution')
    fig3.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                       height=320, margin=dict(t=40,b=20,l=10,r=10))
    fig3.update_yaxes(gridcolor='#eeeeee')
    fig3.update_xaxes(showgrid=False)
    st.plotly_chart(fig3, use_container_width=True)

with c4:
    fig4 = px.histogram(filtered, x='person_age',
                         nbins=40, opacity=0.75,
                         color_discrete_sequence=['#e94560'],
                         labels={'person_age':'Age'},
                         title='Age Distribution')
    fig4.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                       height=320, margin=dict(t=40,b=20,l=10,r=10))
    fig4.update_yaxes(gridcolor='#eeeeee')
    fig4.update_xaxes(showgrid=False)
    st.plotly_chart(fig4, use_container_width=True)
# ── Row 3: Loan amount vs income scatter + Interest rate by grade ──────────────
st.markdown("<div class='section-header'>💡 Loan Amount & Interest Rate Analysis</div>", unsafe_allow_html=True)
c5, c6 = st.columns(2)

with c5:
    income_filtered = filtered[filtered.person_income < 200000]
    sample = income_filtered.sample(min(1500, len(income_filtered)), random_state=42)
    fig5 = px.scatter(sample, x='person_income', y='loan_amnt',
                           color='loan_status_label', opacity=0.6,
                           color_discrete_map={'No Default':'#28a745','Default':'#e94560'},
                           labels={'person_income':'Annual Income ($)','loan_amnt':'Loan Amount ($)','loan_status_label':'Status'},
                           title='Loan Amount vs Income')
    fig5.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                           height=320, margin=dict(t=40,b=20,l=10,r=10),
                       legend=dict(orientation='h', yanchor='bottom', y=1.02))
    fig5.update_xaxes(showgrid=False)
    fig5.update_yaxes(gridcolor='#eeeeee')
    st.plotly_chart(fig5, use_container_width=True)

with c6:
    rate_df = filtered.groupby('loan_grade')['loan_int_rate'].mean().reset_index()
    rate_df = rate_df.sort_values('loan_grade')
    fig6 = px.line(rate_df, x='loan_grade', y='loan_int_rate', markers=True,
                    labels={'loan_grade':'Loan Grade','loan_int_rate':'Avg Interest Rate (%)'},
                    title='Average Interest Rate by Loan Grade')
    fig6.update_traces(line_color='#e94560', marker=dict(size=8, color='#0f3460'))
    fig6.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                       height=320, margin=dict(t=40,b=20,l=10,r=10))
    fig6.update_yaxes(ticksuffix='%', gridcolor='#eeeeee')
    fig6.update_xaxes(showgrid=False)
    st.plotly_chart(fig6, use_container_width=True)

# ── Row 4: Home ownership pie + Raw data ──────────────────────────────────────
st.markdown("<div class='section-header'>🏠 Home Ownership & Raw Data</div>", unsafe_allow_html=True)
c7, c8 = st.columns([1, 2])

with c7:
    own_df = filtered.groupby('person_home_ownership')['loan_status'].mean().reset_index()
    own_df['default_rate'] = own_df['loan_status'] * 100
    fig7 = px.pie(own_df, names='person_home_ownership', values='default_rate',
                   color_discrete_sequence=['#0f3460','#16213e','#e94560','#28a745'],
                   title='Default Rate by Home Ownership')
    fig7.update_layout(height=300, margin=dict(t=40,b=10,l=10,r=10),
                       paper_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig7, use_container_width=True)

with c8:
    st.markdown("**Sample of Filtered Data**")
    st.dataframe(
        filtered[['person_age','person_income','loan_amnt','loan_grade',
                   'loan_int_rate','loan_status_label']].head(10).reset_index(drop=True),
        use_container_width=True, height=280
    )

st.divider()
st.caption(f"Showing {total:,} applicants after filters · Loan Default Prediction Project")
