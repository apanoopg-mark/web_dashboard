import ploty as pt
import pandas as pd
import plotly.express as px
import streamlit as st

# Set page configuration
st.set_page_config(
    page_title="Titanic Passenger Intelligence Dashboard",
    page_icon="🚢",
    layout="wide",
)

# Custom CSS for clean dashboard styling
st.markdown(
    """
    <style>
    .main {
        background-color: #f8f9fa;
    }
    </style>
""",
    unsafe_allow_html=True,
)


# Load data with caching for fast performance
@st.cache_data
def load_data():
  return pd.read_csv("cleaned_File_3.csv")


df = load_data()

# ==========================================
# SIDEBAR CONTROLS & FILTERS
# ==========================================
st.sidebar.header("🎛️ Dashboard Controls")
st.sidebar.markdown(
    "Use the filters below to segment passenger insights dynamically."
)

# Filter by Pclass
all_classes = sorted(df["Pclass"].unique())
selected_pclass = st.sidebar.multiselect(
    "Passenger Class (Pclass):", options=all_classes, default=all_classes
)

# Filter by Sex
all_sexes = df["Sex"].unique().tolist()
selected_sex = st.sidebar.multiselect(
    "Gender:", options=all_sexes, default=all_sexes
)

# Filter by Embarked Port
all_embarked = df["Embarked"].unique().tolist()
selected_embarked = st.sidebar.multiselect(
    "Embarkation Port:", options=all_embarked, default=all_embarked
)

# Age Range Slider
min_age, max_age = int(df["Age"].min()), int(df["Age"].max())
selected_age_range = st.sidebar.slider(
    "Age Range:", min_value=min_age, max_value=max_age, value=(min_age, max_age)
)

# Apply Filters to DataFrame
filtered_df = df[
    (df["Pclass"].isin(selected_pclass))
    & (df["Sex"].isin(selected_sex))
    & (df["Embarked"].isin(selected_embarked))
    & (df["Age"] >= selected_age_range[0])
    & (df["Age"] <= selected_age_range[1])
]

# ==========================================
# MAIN DASHBOARD HEADER & KPI METRICS
# ==========================================
st.title("🚢 Titanic Passenger Intelligence Dashboard")
st.markdown(
    "An interactive decision-support tool extracting key demographic,"
    " financial, and survival insights from cleaned historical logs."
)

if filtered_df.empty:
  st.warning(
      "⚠️ No records match your current filter selections. Please expand your"
      " criteria in the sidebar."
  )
else:
  # Key Performance Indicators (KPIs)
  col1, col2, col3, col4, col5 = st.columns(5)

  total_passengers = len(filtered_df)
  survival_count = filtered_df["Survived"].sum()
  survival_rate = (
      (survival_count / total_passengers) * 100 if total_passengers > 0 else 0
  )
  avg_age = filtered_df["Age"].mean()
  avg_fare = filtered_df["Fare"].mean()
  cabin_known_pct = (
      (filtered_df["Has_Cabin"].sum() / total_passengers) * 100
      if total_passengers > 0
      else 0
  )

  col1.metric("Filtered Passengers", f"{total_passengers:,}")
  col2.metric("Estimated Survival Rate", f"{survival_rate:.1f}%")
  col3.metric("Average Age", f"{avg_age:.1f} yrs")
  col4.metric("Average Ticket Fare", f"${avg_fare:.2f}")
  col5.metric("Cabin Record Rate", f"{cabin_known_pct:.1f}%")

  st.markdown("---")

  # ==========================================
  # ROW 1: DEMOGRAPHICS & SURVIVAL INSIGHTS
  # ==========================================
  chart_col1, chart_col2 = st.columns(2)

  with chart_col1:
    st.subheader("📊 Survival Breakdown by Class & Gender")
    survival_summary = (
        filtered_df.groupby(["Pclass", "Sex", "Survived"])
        .size()
        .reset_index(name="Count")
    )
    survival_summary["Survival_Status"] = survival_summary["Survived"].map(
        {0: "Did Not Survive", 1: "Survived"}
    )

    fig_survival = px.bar(
        survival_summary,
        x="Pclass",
        y="Count",
        color="Survival_Status",
        barmode="group",
        facet_col="Sex",
        labels={
            "Pclass": "Passenger Class",
            "Count": "Number of Passengers",
            "Survival_Status": "Outcome",
        },
        template="plotly_white",
        color_discrete_map={
            "Survived": "#2ecc71",
            "Did Not Survive": "#e74c3c",
        },
    )
    st.plotly_chart(fig_survival, use_container_width=True)

  with chart_col2:
    st.subheader("💰 Ticket Fare Distribution Across Classes")
    fig_box = px.box(
        filtered_df,
        x="Pclass",
        y="Fare",
        color="Pclass",
        labels={"Pclass": "Passenger Class", "Fare": "Ticket Fare ($)"},
        template="plotly_white",
    )
    st.plotly_chart(fig_box, use_container_width=True)

  # ==========================================
  # ROW 2: CORRELATIONS & GEOGRAPHIC METRICS
  # ==========================================
  chart_col3, chart_col4 = st.columns(2)

  with chart_col3:
    st.subheader("📈 Age vs. Fare Correlation")
    fig_scatter = px.scatter(
        filtered_df,
        x="Age",
        y="Fare",
        color=filtered_df["Survived"].map({0: "Perished", 1: "Survived"}),
        symbol="Sex",
        labels={
            "color": "Outcome",
            "Age": "Passenger Age",
            "Fare": "Fare Paid ($)",
        },
        template="plotly_white",
        color_discrete_map={"Survived": "#2ecc71", "Perished": "#e74c3c"},
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

  with chart_col4:
    st.subheader("🌐 Port of Embarkation Distribution")
    embarked_df = filtered_df["Embarked"].value_counts().reset_index()
    embarked_df.columns = ["Port", "Count"]
    port_mapping = {"C": "Cherbourg", "Q": "Queenstown", "S": "Southampton"}
    embarked_df["Port_Name"] = embarked_df["Port"].map(port_mapping)

    fig_pie = px.pie(
        embarked_df,
        names="Port_Name",
        values="Count",
        hole=0.4,
        template="plotly_white",
        color_discrete_sequence=px.colors.qualitative.Pastel,
    )
    st.plotly_chart(fig_pie, use_container_width=True)

  # ==========================================
  # DATA DRILL-DOWN & EXPORT SECTION
  # ==========================================
  st.markdown("---")
  st.subheader("📋 Segmented Passenger Records Table")
  st.markdown(
      "Inspect or export the individual records corresponding to your chosen"
      " configuration."
  )

  show_columns = st.multiselect(
      "Select columns to display:",
      options=filtered_df.columns.tolist(),
      default=[
          "PassengerId",
          "Name",
          "Sex",
          "Age",
          "Pclass",
          "Fare",
          "Survived",
          "Embarked",
      ],
  )

  if show_columns:
    st.dataframe(filtered_df[show_columns], use_container_width=True, height=350)
  else:
    st.dataframe(filtered_df, use_container_width=True, height=350)

  col_dl1, col_dl2 = st.columns([1, 4])
  with col_dl1:
    csv_export = filtered_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download View Data",
        data=csv_export,
        file_name="titanic_filtered_insights.csv",
        mime="text/csv",
    )
