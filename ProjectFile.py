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
    mean_salary_uni_deg, alumni_data, analysis_data, result, salary_col = pickle.load(f)

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
    
plt.xticks([])
ax.set_xlabel("Universities ordered by world rank, from lower-ranked to higher-ranked")
ax.set_ylabel("Mean salary, USD per year")
ax.set_title("Mean salaries of bachelor graduates by university ranking")
ax.legend()
ax.grid(axis='y', linestyle='--', alpha=0.4)
ax.set_axisbelow(True)
ax.set_facecolor('#FAFAFA')
sns.despine(top=True, right=True, ax=ax)

plt.tight_layout()

st.pyplot(fig)

st.caption("Bar chart showing the relationship between average salary and university ranking.")

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
        We took the average salaries of graduates from 39 universities with\
             rankings from 671 to 1. For each university, the minimum number\
             of bachelor's graduates in alumni_data is 20. We see a clear relationship\
             between university ranking and average graduate salary. Despite some fluctuations\
             in this relationship, the moving average and polynomial trend demonstrate growth.\
             Importantly, after a university reaches the top 300 down to the top 10, the growth\
             is minimal, and the values fluctuate around $60,000 per year. Only for the very\
             top universities, such as Harvard and Stanford, do bachelor's salaries consistently\
             reach $70,000 or more. We can conclude that salary depends on university ranking,\
             but once a university is among the world's top 300, the values do not differ\
             significantly, except for the very top institutions.
    </div>
""", unsafe_allow_html=True)

st.markdown("---")
st.subheader("Average Salary by Degree and Field of Study")

degrees = st.multiselect("Degree level(s)", options=analysis_data['Degree'].unique(), default=analysis_data['Degree'].unique())
fields = st.multiselect("Field(s) of study", options=analysis_data['Field'].unique(), default=analysis_data['Field'].unique())

filtered = analysis_data[analysis_data['Degree'].isin(degrees) & analysis_data['Field'].isin(fields)]

salary_heatmap = filtered.pivot_table(index="Degree", 
                                      columns="Field", 
                                      values=salary_col, 
                                      aggfunc="mean")

career_stages = st.multiselect("Career stage(s)", options=['Starting', 'Mid-career'], default=['Starting', 'Mid-career'])
career_data = result.set_index('Field')[['avg_starting_salary', 'avg_mid_career_salary']]
career_data.columns = ['Starting', 'Mid-career']
career_data = career_data.loc[career_data.index.isin(fields), career_stages]  

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

sns.heatmap(salary_heatmap, 
            annot=True, 
            fmt=".0f", 
            cmap="YlGnBu", 
            linewidths=0.5, 
            ax=ax1)
ax1.set_title("Average graduate salary by degree and field of study")
ax1.set_xlabel("Field of study")
ax1.set_ylabel("Degree level")

if not career_data.empty:
    sns.heatmap(career_data.T, annot=True, fmt=".0f", cmap="YlGnBu", linewidths=0.5, ax=ax2)
else:
    ax2.text(0.5, 0.5, "No data", ha='center', va='center')
ax2.set_title("Average salary by field and career stage")
ax2.set_xlabel("Field of study")
ax2.set_ylabel("Career stage")

plt.tight_layout()
st.pyplot(fig)

st.caption("Average annual salary in USD. Darker blue indicates higher pay.")

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
        We examined average salaries by degree and field. The heatmap shows that PhDs \
            in IT earn $88,000, while PhDs in Engineering earn $76,000. However, the\
             overall average salary for engineers (across all degree levels) exceeds\
             $90,000 – indicating that in engineering, a formal degree is not the most\
             critical factor. Personal skills and work experience often matter just as\
             much, if not more. Another key observation: starting salaries (first job after\
             graduation) are very close to bachelor's graduate salaries. This means that\
             earning a Master's or PhD sharply increases your pay, effectively letting you\
             skip many years of slow experience‑based growth. We conclude that advanced degrees\
             accelerate salary progression, but in fields like engineering, individual talent\
             and experience can outweigh formal education.
    </div>
""", unsafe_allow_html=True)