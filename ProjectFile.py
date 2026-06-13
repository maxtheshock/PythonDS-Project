import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import pickle

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');
    
    .custom-title {
        font-family: 'Montserrat', sans-serif;
        color: #1D093B;
        font-size: 2.7rem;
        font-weight: 750;
        line-height: 1.15;
        margin-bottom: 1rem;
        letter-spacing: -0.02em;
        text-align: center;   
    }
    </style>
    
    <div class="custom-title">
        Complex analysis<br>of graduates of top universities
    </div>
    """, unsafe_allow_html=True)

st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;700&display=swap');

    .custom-text {{
        font-family: 'Roboto', sans-serif;
        font-size: 1.2rem;
        line-height: 1.6;
        color: #7E8A91;
        text-align: justify;
        text-indent: 3em;
        margin-bottom: 2rem;
    }}
    </style>

    <div class="custom-text">
        Higher education today is an investment in the future — it affects your career prospects 
        and your future salary. But what factors really matter when choosing a university or a field 
        of study? Is a Master's degree necessary, or is a Bachelor's enough? These are the questions 
        we want to answer in our project. Our goal is to analyze several datasets: university rankings, 
        information on average salaries across six fields in US (Business & Finance, Engineering, Healthcare, IT, 
        Natural Sciences, and Social Sciences), as well as two datasets with information about graduates 
        from top universities. Another aim of the project is to showcase our skills in Python and in key data 
        analysis libraries such as SciPy, NumPy, pandas, and many others.
    </div>
""", unsafe_allow_html=True)

with open('buffer.pkl', 'rb') as f:
    mean_salary_uni_deg, alumni_data = pickle.load(f)

st.write(alumni_data)

st.caption(
    "We created a single DataFrame `alumni_data` based on `job_placement.csv` and "
    "`global_graduate_employability_index.csv`. This object contains data only for "
    "students whose universities are included in our top university list."
)

st.markdown("---")
st.subheader("Bachelor salaries: interactive analysis")

with st.sidebar:
    st.header("Plot settings")
    window = st.slider("Moving average window (size)", min_value=1, max_value=39, value=10, step=1)
    degree = st.slider("Polynomial degree", min_value=1, max_value=5, value=3, step=1)
    show_moving_avg = st.checkbox("Show moving average", value=True)
    show_poly = st.checkbox("Show polynomial trend", value=True)
    bar_color = st.color_picker("Bar color", "#5D9B9B")
    line_color_ma = st.color_picker("MA line color", "#E21C5B")
    line_color_poly = st.color_picker("Poly line color", "#4526A0")

bachelor_vals = mean_salary_uni_deg[mean_salary_uni_deg.index.get_level_values('Degree') == 'Bachelor'].values

sns.set_theme(style="whitegrid", context="talk")
plt.rcParams['font.family'] = 'Segoe UI'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['xtick.labelsize'] = 9
sns.set_palette("pastel")

fig, ax = plt.subplots(figsize=(12, 6))

sns.barplot(x=np.arange(len(bachelor_vals)), y=bachelor_vals,
            color=bar_color, alpha=0.8, edgecolor='#2C5F5F', linewidth=0.8, ax=ax)

if show_moving_avg:
    moving_average = pd.Series(bachelor_vals).rolling(window=window, min_periods=1).mean().values
    ax.plot(moving_average, color=line_color_ma, linewidth=2.5, 
            label=f"Moving average (window={window})")

if show_poly:
    x = np.arange(len(bachelor_vals))
    coeffs = np.polyfit(x, bachelor_vals, degree)
    poly = np.poly1d(coeffs)
    x_smooth = np.linspace(x.min(), x.max(), 300)
    y_smooth = poly(x_smooth)
    ax.plot(x_smooth, y_smooth, '--', color=line_color_poly, linewidth=2.5, 
            label=f"Polynomial trend (deg={degree})")

ax.set_xticks([])
ax.set_xlabel("39 university with rank from 671'st to 1'st")
ax.set_ylabel("Mean Salary $/per year")
ax.set_title("Mean salaries of bachelors graduated from top universities")
ax.legend()
ax.grid(axis='y', linestyle='--', alpha=0.4)
ax.set_axisbelow(True)
ax.set_facecolor('#FAFAFA')
sns.despine(top=True, right=True, ax=ax)

plt.tight_layout()

st.pyplot(fig)

st.caption("Use the sidebar to adjust moving average window, polynomial degree, and colors.")