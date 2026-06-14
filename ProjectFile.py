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
    alumni_data, analysis_data, salary_col, degree_distribution, field_distribution, rank_col, h2_data = pickle.load(f)

uni_rank_sorted = alumni_data[['University', 'University Rate']].drop_duplicates() \
                         .set_index('University')['University Rate'].sort_values(ascending=False)
mean_salary_uni_deg = alumni_data.groupby(['University', 'Degree'])[salary_col].mean().reset_index()
mean_salary_uni_deg['Rank'] = mean_salary_uni_deg['University'].map(uni_rank_sorted)
mean_salary_uni_deg = mean_salary_uni_deg.sort_values(by='Rank', ascending=False)
mean_salary_uni_deg = mean_salary_uni_deg.drop('Rank', axis=1).set_index(['University', 'Degree'])[salary_col]

st.write(alumni_data)

st.caption(
    "We created a single DataFrame `alumni_data` based on `job_placement.csv` and "
    "`global_graduate_employability_index.csv`. This object contains data only for "
    "students whose universities are included in our top university list."
)

st.write(field_distribution)
st.write(degree_distribution)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

sns.histplot(
    analysis_data[salary_col],
    bins=30,
    kde=True,
    ax=axes[0, 0]
)
axes[0, 0].set_title("Salary distribution")
axes[0, 0].set_xlabel("Salary, USD per year")

sns.histplot(
    analysis_data[rank_col],
    bins=30,
    kde=True,
    ax=axes[0, 1]
)
axes[0, 1].set_title("University rank distribution")
axes[0, 1].set_xlabel("University rank")

sns.histplot(
    analysis_data["Log University Rank"],
    bins=30,
    kde=True,
    ax=axes[1, 0]
)
axes[1, 0].set_title("Log university rank distribution")
axes[1, 0].set_xlabel("log10(rank)")

sns.histplot(
    analysis_data["Salary Relative to Field Median"],
    bins=30,
    kde=True,
    ax=axes[1, 1]
)
axes[1, 1].set_title("Salary relative to field median")
axes[1, 1].set_xlabel("Salary / field median salary")

plt.tight_layout()
st.pyplot(fig)

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
        The graphs provide information about the alumni_data. It can be observed that\
             salaries are not distributed as a normal distribution (a single peak),\
             but rather have two distinct peaks: $30,000 per year and the largest number\
             of graduates earn about $65,000 per year (over 450 students). Also, we mostly\
             work with top‑100 universities: about 1.5 thousand students graduated from\
             universities in the top 100. Finally, the distribution of graduate salaries\
             compared to the average indicators in their fields shows a similar graph to\
             the regular salary distribution, with two main peaks at 0.5 and 1.0 of the average salary by profession.
    </div>
""", unsafe_allow_html=True)

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

st.subheader("University rank vs graduate salary")

st.markdown("""
The scatter plot shows the relationship between university world rank (log‑scaled) and graduate salary, coloured by degree level. 
The solid black line represents the general trend across all degrees.
""")

fig, ax = plt.subplots(figsize=(12, 7))

sns.scatterplot(
    data=h2_data,
    x="Log University Rank",
    y=salary_col,
    hue="Degree",
    alpha=0.35,
    s=55,
    edgecolor="white",
    linewidth=0.3,
    ax=ax
)

sns.regplot(
    data=h2_data,
    x="Log University Rank",
    y=salary_col,
    scatter=False,
    color="black",
    line_kws={"linewidth": 3, "label": "General trend"},
    ax=ax
)

rank_ticks = np.array([1, 2, 5, 10, 20, 50, 100, 200, 500, 1000])
rank_ticks = rank_ticks[
    (rank_ticks >= h2_data[rank_col].min()) &
    (rank_ticks <= h2_data[rank_col].max())
]
ax.set_xticks(np.log10(rank_ticks))
ax.set_xticklabels(rank_ticks)

ax.set_title("University rank and graduate salary", fontsize=15, weight="bold")
ax.set_xlabel("University world rank, log‑scaled axis")
ax.set_ylabel("Salary, USD per year")
ax.grid(axis="both", linestyle="--", alpha=0.3)
ax.legend(title="Degree")

st.pyplot(fig)

st.caption("Lower rank (left side) means better university. The trend line confirms that graduates from higher‑ranked universities tend to earn higher salaries, especially at PhD level.")

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
    <p>
        The correlation coefficient between log university rank and annual graduate salary\
             is -0.517. This indicates a moderate negative relationship: the higher a university’s\
             position in the ranking (i.e., the lower its numerical rank), the higher its graduates’ salaries tend to be.
    </p>
    <p>
        The result supports the assumption that graduates from higher‑ranked universities generally\
             earn higher salaries. However, it should be emphasized that this relationship is associative\
             and does not prove causality. Other factors – such as student self‑selection, access to resources,\
             and professional networks – may also influence the observed correlation.
    </p>
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

starting_salary = analysis_data[analysis_data['Degree'] == 'Bachelor'].groupby('Field')[salary_col].mean()
midcareer_salary = analysis_data[analysis_data['Degree'].isin(['Master', 'PhD'])].groupby('Field')[salary_col].mean()
career_stage_salary = pd.DataFrame({
    'Starting': starting_salary,
    'Mid-career': midcareer_salary
}).fillna(0)

career_stages = st.multiselect("Career stage(s)", options=['Starting', 'Mid-career'], default=['Starting', 'Mid-career'])
career_filtered = career_stage_salary.loc[career_stage_salary.index.isin(fields), career_stages]

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

if not career_filtered.empty:
    sns.heatmap(career_filtered.T, annot=True, fmt=".0f", cmap="YlGnBu", linewidths=0.5, ax=ax2)
else:
    ax2.text(0.5, 0.5, "No data for selected fields", ha='center', va='center')
ax2.set_title("Average salary by field and career stage")
ax2.set_xlabel("Field of study")
ax2.set_ylabel("Career stage")

plt.tight_layout()
st.pyplot(fig)

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
            <p>
        WThe heatmap reveals a monotonic increase in average salary with higher degree attainment across all fields,\
             with doctoral graduates earning substantially more than bachelor’s holders. Among disciplines, Information\
             Technology and Healthcare & Medicine exhibit the highest terminal salaries (≈88,500 and ≈82,800, respectively),\
             while Social Sciences and Natural Sciences show the lowest (≈60,800 and ≈63,800). Engineering and Business & \
            Finance occupy intermediate positions. The gradient suggests that advanced degrees yield the greatest pecuniary\
             returns in applied technical and medical fields. The upper heatmap demonstrates a clear educational premium: across\
             all fields, average salaries rise monotonically from Bachelor to PhD. The lower heatmap shows a career-stage premium:\
             experienced professionals earn significantly more than starting graduates in every discipline.
            <p>
            <p>
        Comparing the two, the gain from obtaining a PhD (relative to a Bachelor) is often comparable to or even larger than the gain\
             from moving from a starting to an experienced role. For instance, in Information Technology, the Bachelor→PhD increment\
             is roughly +33,000, while the starting→experienced increment is about +22,500. In Healthcare & Medicine, the educational\
             increment is ≈ +27,300, versus a career increment of ≈ +22,500. Thus, advanced degrees provide a salary boost at least as \
            substantial as several years of work experience, and the two effects are additive: a PhD holder at mid‑career would earn\
             considerably more than either a Bachelor’s holder with experience or a PhD graduate just starting out. The heatmaps together\
             imply that both human capital accumulation (degrees) and on‑the‑job learning (experience) are strongly rewarded, with degrees\
             having a slightly larger marginal effect in technical fields.
            <p>
    </div>
""", unsafe_allow_html=True)

st.markdown("---")
st.subheader("Conclusion")

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
        <p>
        This project demonstrates that graduate salary is associated with multiple factors.\
             First, degree level is positively related to salary: field‑adjusted comparisons\
             show higher median relative salaries for higher degree levels. Second, field of\
             study matters, as some disciplines command systematically higher salaries; thus,\
             any salary analysis must account for specialization. Third, university ranking also\
             correlates with salary, with descriptively higher graduate salaries observed for\
             better‑ranked institutions. Overall, salary should be studied as a function of\
             interacting variables rather than any single determinant. The key conclusion is that\
             degree level, field of study, and university ranking together provide useful information\
             for understanding graduate salary patterns in this dataset.
        </p>
    </div>
""", unsafe_allow_html=True)