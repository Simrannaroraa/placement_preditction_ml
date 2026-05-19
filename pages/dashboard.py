import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Theme CSS ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background: #0d0f14; color: #e8e6e1; }
#MainMenu, footer { visibility: hidden; }
header { background: transparent !important; }
.block-container { padding-top: 1.5rem; padding-bottom: 3rem; max-width: 1400px; }

.dash-header {
    padding: 1.8rem 0 1.2rem;
    border-bottom: 1px solid #252a35;
    margin-bottom: 1.8rem;
}
.dash-header h1 {
    font-family: 'Syne', sans-serif;
    font-size: 2rem;
    font-weight: 800;
    letter-spacing: -0.02em;
    background: linear-gradient(135deg, #f5c842 0%, #ff6b35 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0 0 0.3rem;
}
.dash-header p { color: #666; font-size: 0.9rem; margin: 0; }

/* KPI cards */
.kpi-row { display: flex; gap: 1rem; margin-bottom: 1.8rem; flex-wrap: wrap; }
.kpi-card {
    flex: 1; min-width: 160px;
    background: #161a22;
    border: 1px solid #252a35;
    border-radius: 14px;
    padding: 1.2rem 1.4rem;
}
.kpi-label { font-size: 0.7rem; letter-spacing: 0.15em; text-transform: uppercase; color: #555; font-weight: 600; }
.kpi-value { font-family: 'Syne', sans-serif; font-size: 2rem; font-weight: 800; color: #e8e6e1; line-height: 1.1; margin: 0.3rem 0 0; }
.kpi-sub { font-size: 0.78rem; color: #f5c842; margin-top: 0.25rem; }

/* Section headers */
.section-head {
    font-family: 'Syne', sans-serif;
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: #f5c842;
    margin: 2rem 0 0.8rem;
}

div[data-testid="stPlotlyChart"] { border-radius: 14px; overflow: hidden; }

/* Sidebar nav styling */
[data-testid="stSidebarNav"] a {
    font-family: 'Syne', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    color: #b0b8cc !important;
}
[data-testid="stSidebarNav"] a:hover,
[data-testid="stSidebarNav"] a[aria-current="page"] {
    color: #f5c842 !important;
}
section[data-testid="stSidebar"] {
    background: #0d0f14 !important;
    border-right: 1px solid #1e2330 !important;
}
</style>
""", unsafe_allow_html=True)

# ── Load Data ──────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    import os
    paths = ["Placement_Prediction_data.csv", "Salary_prediction_data.csv"]
    upload_paths = [
        "/mnt/user-data/uploads/Placement_Prediction_data.csv",
        "/mnt/user-data/uploads/Salary_prediction_data.csv"
    ]
    for p1, p2 in [(paths[0], paths[1]), (upload_paths[0], upload_paths[1])]:
        if os.path.exists(p1) and os.path.exists(p2):
            df1 = pd.read_csv(p1)
            df2 = pd.read_csv(p2)
            break
    else:
        return None, None

    for df in [df1, df2]:
        drop = [c for c in df.columns if 'Unnamed' in c]
        df.drop(columns=drop, inplace=True)
        df.fillna(0, inplace=True)

    return df1, df2

df, df_sal = load_data()

# ── Plotly theme ───────────────────────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    paper_bgcolor="#161a22",
    plot_bgcolor="#161a22",
    font=dict(family="DM Sans, sans-serif", color="#b0b8cc", size=12),
    margin=dict(l=20, r=20, t=40, b=20),
    colorway=["#f5c842", "#ff6b35", "#4cc9f0", "#7b61ff", "#06d6a0", "#ef476f"],
    xaxis=dict(gridcolor="#252a35", linecolor="#252a35", tickcolor="#252a35"),
    yaxis=dict(gridcolor="#252a35", linecolor="#252a35", tickcolor="#252a35"),
)

ACCENT    = "#f5c842"
ACCENT2   = "#ff6b35"
PLACED    = "#06d6a0"
NOTPLACED = "#ef476f"
BLUE      = "#4cc9f0"

if df is None:
    st.error("⚠️ CSVs not found. Place `Placement_Prediction_data.csv` and `Salary_prediction_data.csv` in the same folder.")
    st.stop()

# ── Derived columns ────────────────────────────────────────────────────────────
df["Placed"] = (df["PlacementStatus"] == "Placed").astype(int)
df["cgpa_bucket"] = pd.cut(
    df["CGPA"],
    bins=[6, 7, 7.5, 8, 8.5, 9.2],
    labels=["6–7", "7–7.5", "7.5–8", "8–8.5", "8.5+"]
)

placed_df = df_sal[df_sal["salary"] > 0].copy()
placed_df["cgpa_bucket"] = pd.cut(
    placed_df["CGPA"],
    bins=[6, 7, 7.5, 8, 8.5, 9.2],
    labels=["6–7", "7–7.5", "7.5–8", "8–8.5", "8.5+"]
)

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="dash-header">
  <h1>📊 Placement Analytics Dashboard</h1>
  <p>Insights from 10,000 student records · Placement Prediction Dataset</p>
</div>
""", unsafe_allow_html=True)

# ── KPI row ────────────────────────────────────────────────────────────────────
total      = len(df)
placed_n   = df["Placed"].sum()
not_placed = total - placed_n
avg_salary = placed_df["salary"].mean()
intern_boost = (
    df[df["Internship"]=="Yes"]["Placed"].mean() -
    df[df["Internship"]=="No"]["Placed"].mean()
) * 100

st.markdown(f"""
<div class="kpi-row">
  <div class="kpi-card">
    <div class="kpi-label">Total Students</div>
    <div class="kpi-value">{total:,}</div>
    <div class="kpi-sub">in dataset</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-label">Placed</div>
    <div class="kpi-value">{placed_n:,}</div>
    <div class="kpi-sub">{placed_n/total*100:.1f}% placement rate</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-label">Not Placed</div>
    <div class="kpi-value">{not_placed:,}</div>
    <div class="kpi-sub">{not_placed/total*100:.1f}% of students</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-label">Avg Salary (Placed)</div>
    <div class="kpi-value">₹{avg_salary/100000:.1f}L</div>
    <div class="kpi-sub">per annum</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-label">Internship Boost</div>
    <div class="kpi-value">+{intern_boost:.0f}%</div>
    <div class="kpi-sub">placement rate lift</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Filters ────────────────────────────────────────────────────────────────────
with st.expander("🔧 Filters", expanded=False):
    fc1, fc2, fc3 = st.columns(3)
    with fc1:
        internship_filter = st.multiselect("Internship", ["Yes", "No"], default=["Yes", "No"])
    with fc2:
        hackathon_filter  = st.multiselect("Hackathon",  ["Yes", "No"], default=["Yes", "No"])
    with fc3:
        cgpa_range = st.slider("CGPA Range", 6.0, 9.1, (6.0, 9.1), 0.1)

mask = (
    df["Internship"].isin(internship_filter) &
    df["Hackathon"].isin(hackathon_filter) &
    df["CGPA"].between(cgpa_range[0], cgpa_range[1])
)
fdf = df[mask]

# ── ROW 1: Placement overview ──────────────────────────────────────────────────
st.markdown('<div class="section-head">Placement Overview</div>', unsafe_allow_html=True)
r1c1, r1c2, r1c3 = st.columns([1, 2, 2])

with r1c1:
    placed_count    = fdf["Placed"].sum()
    notplaced_count = len(fdf) - placed_count
    fig = go.Figure(go.Pie(
        labels=["Placed", "Not Placed"],
        values=[placed_count, notplaced_count],
        hole=0.62,
        marker_colors=[PLACED, NOTPLACED],
        textinfo="percent",
        textfont_size=13,
    ))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        height=280,
        showlegend=True,
        legend=dict(orientation="h", y=-0.15, font_size=11),
        title=dict(text="Placement split", font=dict(size=13, color="#888")),
        annotations=[dict(
            text=f"<b>{placed_count/len(fdf)*100:.0f}%</b>",
            x=0.5, y=0.5, font_size=22, font_color=PLACED,
            showarrow=False
        )]
    )
    st.plotly_chart(fig, use_container_width=True)

with r1c2:
    cgpa_p = fdf.groupby("cgpa_bucket", observed=True)["Placed"].mean().mul(100).round(1).reset_index()
    cgpa_p.columns = ["CGPA Range", "Placement Rate (%)"]
    fig = px.bar(
        cgpa_p, x="CGPA Range", y="Placement Rate (%)",
        color="Placement Rate (%)",
        color_continuous_scale=[[0, "#252a35"], [0.5, ACCENT2], [1, ACCENT]],
        text="Placement Rate (%)",
        title="Placement rate by CGPA"
    )
    fig.update_traces(texttemplate="%{text:.0f}%", textposition="outside")
    fig.update_layout(**PLOTLY_LAYOUT, height=280, coloraxis_showscale=False,
                      title_font_size=13, title_font_color="#888")
    st.plotly_chart(fig, use_container_width=True)

with r1c3:
    sk_p = fdf.groupby("Skills")["Placed"].mean().mul(100).round(1).reset_index()
    sk_p.columns = ["No. of Skills", "Placement Rate (%)"]
    fig = px.bar(
        sk_p, x="No. of Skills", y="Placement Rate (%)",
        color="Placement Rate (%)",
        color_continuous_scale=[[0, "#252a35"], [0.5, BLUE], [1, ACCENT]],
        text="Placement Rate (%)",
        title="Placement rate by skills count"
    )
    fig.update_traces(texttemplate="%{text:.0f}%", textposition="outside")
    fig.update_layout(**PLOTLY_LAYOUT, height=280, coloraxis_showscale=False,
                      title_font_size=13, title_font_color="#888")
    st.plotly_chart(fig, use_container_width=True)

# ── ROW 2: Activity factors ────────────────────────────────────────────────────
st.markdown('<div class="section-head">Activity Factors</div>', unsafe_allow_html=True)
r2c1, r2c2, r2c3 = st.columns(3)

with r2c1:
    cats = ["Internship: No", "Internship: Yes", "Hackathon: No", "Hackathon: Yes"]
    rates = [
        fdf[fdf["Internship"]=="No"]["Placed"].mean()*100,
        fdf[fdf["Internship"]=="Yes"]["Placed"].mean()*100,
        fdf[fdf["Hackathon"]=="No"]["Placed"].mean()*100,
        fdf[fdf["Hackathon"]=="Yes"]["Placed"].mean()*100,
    ]
    colors = [NOTPLACED, PLACED, NOTPLACED, PLACED]
    fig = go.Figure(go.Bar(
        x=cats, y=[round(r, 1) for r in rates],
        marker_color=colors,
        text=[f"{r:.0f}%" for r in rates],
        textposition="outside",
    ))
    fig.update_layout(**PLOTLY_LAYOUT, height=290, showlegend=False,
                      title=dict(text="Internship & hackathon impact", font=dict(size=13, color="#888")),
                      yaxis_title="Placement rate (%)")
    st.plotly_chart(fig, use_container_width=True)

with r2c2:
    bl_p = fdf[fdf["backlogs"] <= 4].groupby("backlogs")["Placed"].mean().mul(100).round(1).reset_index()
    bl_p.columns = ["Backlogs", "Placement Rate (%)"]
    fig = px.bar(
        bl_p, x="Backlogs", y="Placement Rate (%)",
        color="Placement Rate (%)",
        color_continuous_scale=[[0, NOTPLACED], [0.5, ACCENT2], [1, PLACED]],
        text="Placement Rate (%)",
        title="Effect of backlogs on placement"
    )
    fig.update_traces(texttemplate="%{text:.0f}%", textposition="outside")
    fig.update_layout(**PLOTLY_LAYOUT, height=290, coloraxis_showscale=False,
                      title_font_size=13, title_font_color="#888",
                      yaxis_title="Placement rate (%)")
    st.plotly_chart(fig, use_container_width=True)

with r2c3:
    comm_df = fdf.copy()
    comm_df["comm_bucket"] = pd.cut(
        comm_df["Communication Skill Rating"],
        bins=[0, 3, 4, 5],
        labels=["Low (1–3)", "Medium (3–4)", "High (4–5)"]
    )
    comm_p = comm_df.groupby("comm_bucket", observed=True)["Placed"].mean().mul(100).round(1).reset_index()
    comm_p.columns = ["Communication Level", "Placement Rate (%)"]
    fig = px.bar(
        comm_p, x="Communication Level", y="Placement Rate (%)",
        color="Placement Rate (%)",
        color_continuous_scale=[[0, "#252a35"], [0.5, ACCENT2], [1, ACCENT]],
        text="Placement Rate (%)",
        title="Communication skill vs placement"
    )
    fig.update_traces(texttemplate="%{text:.0f}%", textposition="outside")
    fig.update_layout(**PLOTLY_LAYOUT, height=290, coloraxis_showscale=False,
                      title_font_size=13, title_font_color="#888",
                      yaxis_title="Placement rate (%)")
    st.plotly_chart(fig, use_container_width=True)

# ── ROW 3: CGPA distribution ───────────────────────────────────────────────────
st.markdown('<div class="section-head">CGPA & Academic Distribution</div>', unsafe_allow_html=True)
r3c1, r3c2 = st.columns([3, 2])

with r3c1:
    placed_cgpa    = fdf[fdf["Placed"]==1]["CGPA"]
    notplaced_cgpa = fdf[fdf["Placed"]==0]["CGPA"]
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=placed_cgpa, name="Placed",
        marker_color=PLACED, opacity=0.75,
        xbins=dict(start=6.4, end=9.2, size=0.2),
        histnorm="percent"
    ))
    fig.add_trace(go.Histogram(
        x=notplaced_cgpa, name="Not Placed",
        marker_color=NOTPLACED, opacity=0.75,
        xbins=dict(start=6.4, end=9.2, size=0.2),
        histnorm="percent"
    ))
    fig.update_layout(
        **PLOTLY_LAYOUT, barmode="overlay", height=300,
        title=dict(text="CGPA distribution: Placed vs Not Placed", font=dict(size=13, color="#888")),
        xaxis_title="CGPA", yaxis_title="% of group",
    )
    st.plotly_chart(fig, use_container_width=True)

with r3c2:
    proj_p = fdf.groupby("Major Projects")["Placed"].mean().mul(100).round(1).reset_index()
    proj_p.columns = ["Major Projects", "Placement Rate (%)"]
    fig = px.bar(
        proj_p, x="Major Projects", y="Placement Rate (%)",
        color="Placement Rate (%)",
        color_continuous_scale=[[0, "#252a35"], [0.5, BLUE], [1, ACCENT]],
        text="Placement Rate (%)",
        title="Major projects vs placement"
    )
    fig.update_traces(texttemplate="%{text:.0f}%", textposition="outside")
    fig.update_layout(**PLOTLY_LAYOUT, height=300, coloraxis_showscale=False,
                      title_font_size=13, title_font_color="#888",
                      yaxis_title="Placement rate (%)")
    st.plotly_chart(fig, use_container_width=True)

# ── ROW 4: Salary insights ─────────────────────────────────────────────────────
st.markdown('<div class="section-head">Salary Insights (Placed Students Only)</div>', unsafe_allow_html=True)
r4c1, r4c2 = st.columns([2, 2])

with r4c1:
    sal_dist = placed_df["salary"].value_counts().sort_index().reset_index()
    sal_dist.columns = ["Salary (₹)", "Count"]
    sal_dist["Salary (L)"] = (sal_dist["Salary (₹)"] / 100000).round(1).astype(str) + "L"
    fig = px.bar(
        sal_dist, x="Salary (₹)", y="Count",
        color="Count",
        color_continuous_scale=[[0, "#252a35"], [0.5, ACCENT2], [1, ACCENT]],
        title="Salary distribution among placed students",
        hover_data={"Salary (L)": True, "Salary (₹)": False}
    )
    fig.update_layout(**PLOTLY_LAYOUT, height=310, coloraxis_showscale=False,
                      title_font_size=13, title_font_color="#888",
                      xaxis_title="Salary (₹)", yaxis_title="Students")
    fig.update_xaxes(tickformat=",")
    st.plotly_chart(fig, use_container_width=True)

with r4c2:
    sal_cgpa = placed_df.groupby("cgpa_bucket", observed=True)["salary"].mean().div(100000).round(2).reset_index()
    sal_cgpa.columns = ["CGPA Range", "Avg Salary (₹L)"]
    fig = px.line(
        sal_cgpa, x="CGPA Range", y="Avg Salary (₹L)",
        markers=True,
        title="Average salary by CGPA range"
    )
    fig.update_traces(
        line_color=ACCENT, line_width=2.5,
        marker=dict(size=10, color=ACCENT2, line=dict(color=ACCENT, width=2))
    )
    fig.update_layout(**PLOTLY_LAYOUT, height=310,
                      title_font_size=13, title_font_color="#888",
                      yaxis_title="Avg salary (₹ Lakhs)")
    st.plotly_chart(fig, use_container_width=True)

# ── ROW 5: Heatmap ────────────────────────────────────────────────────────────
st.markdown('<div class="section-head">Skills × Backlogs Heatmap</div>', unsafe_allow_html=True)

heat = (
    fdf[fdf["backlogs"] <= 4]
    .groupby(["Skills", "backlogs"])["Placed"]
    .mean()
    .mul(100)
    .round(1)
    .reset_index()
    .pivot(index="Skills", columns="backlogs", values="Placed")
    .fillna(0)
)

fig = px.imshow(
    heat,
    color_continuous_scale=[[0, "#1a1e28"], [0.4, NOTPLACED], [0.7, ACCENT2], [1, PLACED]],
    aspect="auto",
    text_auto=True,
    title="Placement rate (%) by number of skills & backlogs",
    labels=dict(x="Backlogs", y="Number of Skills", color="Placed %")
)
fig.update_layout(**PLOTLY_LAYOUT, height=280,
                  title_font_size=13, title_font_color="#888",
                  coloraxis_colorbar=dict(tickfont_color="#888"))
st.plotly_chart(fig, use_container_width=True)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center; color:#333; font-size:0.78rem; margin-top:2.5rem; padding-top:1rem; border-top:1px solid #1e2330;">
  Dashboard · Placement Prediction ML Project · 10,000 student records
</div>
""", unsafe_allow_html=True)