import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# =====================================================
# PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="EduPro | Student Intelligence",
    page_icon="🎓",
    layout="wide"
)


# =====================================================
# CUSTOM CSS
# =====================================================

st.markdown("""
<style>

.main {
    background-color: #f5f7fb;
}

.metric-card {
    background-color: white;
    padding: 20px;
    border-radius: 12px;
    box-shadow: 0px 2px 8px rgba(0,0,0,0.05);
}

h1, h2, h3 {
    color: #172554;
}

</style>
""", unsafe_allow_html=True)


# =====================================================
# LOAD DATASETS
# =====================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "dataset"


@st.cache_data
def load_data():

    users = pd.read_csv(DATA_DIR / "Users.csv")
    courses = pd.read_csv(DATA_DIR / "Courses.csv")
    transactions = pd.read_csv(DATA_DIR / "Transactions.csv")
    teachers = pd.read_csv(DATA_DIR / "Teachers.csv")

    transactions["TransactionDate"] = pd.to_datetime(
        transactions["TransactionDate"],
        dayfirst=True,
        errors="coerce"
    )

    return users, courses, transactions, teachers


# =====================================================
# FEATURE ENGINEERING AND CLUSTERING
# =====================================================

@st.cache_data
def process_data(users, courses, transactions):

    # Merge transaction and course information

    data = transactions.merge(
        courses,
        on="CourseID",
        how="left"
    )

    # Aggregate learner information

    features = data.groupby("UserID").agg(

        Total_Enrollments=("CourseID", "count"),

        Unique_Categories=("CourseCategory", "nunique"),

        Average_Spending=("Amount", "mean"),

        Total_Spending=("Amount", "sum"),

        Average_Course_Rating=("CourseRating", "mean"),

        Average_Course_Price=("CoursePrice", "mean"),

        First_Enrollment=("TransactionDate", "min"),

        Last_Enrollment=("TransactionDate", "max")

    ).reset_index()

    # Enrollment frequency

    features["Enrollment_Frequency"] = (
        features["Total_Enrollments"] /
        (
            (
                features["Last_Enrollment"] -
                features["First_Enrollment"]
            ).dt.days.fillna(0) / 30 + 1
        )
    )

    # Course level preferences

    level_counts = pd.crosstab(
        data["UserID"],
        data["CourseLevel"]
    )

    for level in ["Beginner", "Intermediate", "Advanced"]:

        if level not in level_counts.columns:
            level_counts[level] = 0

    level_counts = level_counts[
        ["Beginner", "Intermediate", "Advanced"]
    ]

    level_counts = level_counts.div(
        level_counts.sum(axis=1).replace(0, 1),
        axis=0
    )

    level_counts.columns = [
        "Beginner_Ratio",
        "Intermediate_Ratio",
        "Advanced_Ratio"
    ]

    features = features.merge(
        level_counts.reset_index(),
        on="UserID",
        how="left"
    )

    # Course category preferences

    category_counts = pd.crosstab(
        data["UserID"],
        data["CourseCategory"]
    )

    category_counts = category_counts.div(
        category_counts.sum(axis=1).replace(0, 1),
        axis=0
    )

    category_counts.columns = [
        "Category_" + str(col)
        for col in category_counts.columns
    ]

    features = features.merge(
        category_counts.reset_index(),
        on="UserID",
        how="left"
    )

    # Combine with user demographics

    features = users[
        ["UserID", "UserName", "Age", "Gender"]
    ].merge(
        features,
        on="UserID",
        how="left"
    )

    # Handle students without enrollments

    numeric_columns = features.select_dtypes(
        include=np.number
    ).columns

    features[numeric_columns] = features[
        numeric_columns
    ].fillna(0)

    # Handle missing preference ratios

    ratio_columns = [
        "Beginner_Ratio",
        "Intermediate_Ratio",
        "Advanced_Ratio"
    ]

    features[ratio_columns] = features[
        ratio_columns
    ].fillna(0)

    # Clustering features

    excluded_columns = [
        "UserID",
        "UserName",
        "Gender",
        "First_Enrollment",
        "Last_Enrollment"
    ]

    model_features = features.drop(
        columns=excluded_columns,
        errors="ignore"
    )

    model_features = model_features.replace(
        [np.inf, -np.inf],
        0
    ).fillna(0)

    # Scale features

    scaler = StandardScaler()

    scaled_features = scaler.fit_transform(
        model_features
    )

    # K-Means clustering

    kmeans = KMeans(
        n_clusters=4,
        random_state=42,
        n_init=10
    )

    features["Cluster"] = kmeans.fit_predict(
        scaled_features
    )

    # Cluster evaluation

    score = silhouette_score(
        scaled_features,
        features["Cluster"],
        sample_size=min(2000, len(features)),
        random_state=42
    ) if features["Cluster"].nunique() > 1 else 0

    return features, data, score


# =====================================================
# RECOMMENDATION SYSTEM
# =====================================================

def recommend_courses(user_id, features, data, courses):

    student = features[
        features["UserID"] == user_id
    ]

    if student.empty:
        return pd.DataFrame()

    cluster = student.iloc[0]["Cluster"]

    # Courses already enrolled by student

    enrolled = data[
        data["UserID"] == user_id
    ]["CourseID"].unique()

    # Find students belonging to same cluster

    cluster_users = features[
        features["Cluster"] == cluster
    ]["UserID"]

    cluster_data = data[
        data["UserID"].isin(cluster_users)
    ]

    # Popular courses in the cluster

    popularity = cluster_data.groupby(
        "CourseID"
    ).size().reset_index(name="Cluster_Enrollments")

    recommendations = courses.merge(
        popularity,
        on="CourseID",
        how="left"
    )

    recommendations["Cluster_Enrollments"] = (
        recommendations["Cluster_Enrollments"].fillna(0)
    )

    # Remove already enrolled courses

    recommendations = recommendations[
        ~recommendations["CourseID"].isin(enrolled)
    ].copy()

    if recommendations.empty:
        return recommendations

    # Find student's preferred category and level

    student_history = data[
        data["UserID"] == user_id
    ]

    preferred_category = None
    preferred_level = None

    if not student_history.empty:

        preferred_category = (
            student_history["CourseCategory"].mode().iloc[0]
        )

        preferred_level = (
            student_history["CourseLevel"].mode().iloc[0]
        )

    # Recommendation scoring

    max_popularity = max(
        recommendations["Cluster_Enrollments"].max(),
        1
    )

    recommendations["Popularity_Score"] = (
        recommendations["Cluster_Enrollments"] /
        max_popularity
    )

    recommendations["Rating_Score"] = (
        recommendations["CourseRating"] / 5
    )

    recommendations["Category_Match"] = (
        recommendations["CourseCategory"] ==
        preferred_category
    ).astype(int)

    recommendations["Level_Match"] = (
        recommendations["CourseLevel"] ==
        preferred_level
    ).astype(int)

    recommendations["Recommendation_Score"] = (

        recommendations["Popularity_Score"] * 0.30 +

        recommendations["Rating_Score"] * 0.30 +

        recommendations["Category_Match"] * 0.25 +

        recommendations["Level_Match"] * 0.15

    )

    recommendations = recommendations.sort_values(
        "Recommendation_Score",
        ascending=False
    )

    return recommendations.head(10)


# =====================================================
# LOAD AND PROCESS EVERYTHING
# =====================================================

try:

    users, courses, transactions, teachers = load_data()

    features, transaction_data, silhouette = process_data(
        users,
        courses,
        transactions
    )

except Exception as e:

    st.error(f"Error loading or processing datasets: {e}")

    st.info(
        "Check your dataset folder and CSV filenames."
    )

    st.stop()


# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.title("🎓 EduPro")

st.sidebar.caption(
    "Student Intelligence & Personalization"
)

page = st.sidebar.radio(

    "Navigation",

    [
        "Dashboard",
        "Student Segmentation",
        "Student Explorer",
        "Course Recommendations",
        "Segment Comparison"
    ]

)


# =====================================================
# DASHBOARD
# =====================================================

if page == "Dashboard":

    st.title("🎓 EduPro Analytics Dashboard")

    st.write(
        "Understand student behaviour and learning patterns."
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Students",
        f"{len(users):,}"
    )

    col2.metric(
        "Total Courses",
        f"{len(courses):,}"
    )

    col3.metric(
        "Total Enrollments",
        f"{len(transactions):,}"
    )

    col4.metric(
        "Student Segments",
        features["Cluster"].nunique()
    )

    st.divider()

    c1, c2 = st.columns(2)

    with c1:

        st.subheader("Student Age Distribution")

        fig = px.histogram(
            users,
            x="Age",
            color="Gender",
            title="Age Distribution by Gender"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with c2:

        st.subheader("Course Category Distribution")

        fig = px.pie(
            courses,
            names="CourseCategory",
            title="Available Courses by Category"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    st.subheader("Monthly Enrollment Trends")

    monthly = transaction_data.dropna(
        subset=["TransactionDate"]
    ).copy()

    monthly["Month"] = (
        monthly["TransactionDate"].dt.to_period("M").astype(str)
    )

    monthly_count = monthly.groupby(
        "Month"
    ).size().reset_index(name="Enrollments")

    fig = px.line(
        monthly_count,
        x="Month",
        y="Enrollments",
        markers=True,
        title="Monthly Student Enrollments"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.subheader("Model Performance")

    st.metric(
        "Silhouette Score",
        round(silhouette, 3)
    )


# =====================================================
# STUDENT SEGMENTATION
# =====================================================

elif page == "Student Segmentation":

    st.title("👥 Student Segmentation")

    st.write(
        "Students are grouped according to their learning behaviour."
    )

    cluster_counts = features["Cluster"].value_counts(
    ).sort_index().reset_index()

    cluster_counts.columns = [
        "Cluster",
        "Students"
    ]

    fig = px.bar(
        cluster_counts,
        x="Cluster",
        y="Students",
        color="Cluster",
        title="Student Distribution Across Clusters"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.subheader("Cluster Characteristics")

    cluster_profile = features.groupby(
        "Cluster"
    ).agg(

        Students=("UserID", "count"),

        Average_Age=("Age", "mean"),

        Average_Enrollments=("Total_Enrollments", "mean"),

        Average_Spending=("Average_Spending", "mean"),

        Category_Diversity=("Unique_Categories", "mean"),

        Average_Rating=("Average_Course_Rating", "mean")

    ).round(2)

    st.dataframe(
        cluster_profile,
        use_container_width=True
    )

    st.subheader("Cluster Behaviour")

    fig = px.scatter(
        features,
        x="Total_Enrollments",
        y="Unique_Categories",
        color="Cluster",
        hover_data=["UserName"],
        title="Student Enrollment vs Category Diversity"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# =====================================================
# STUDENT EXPLORER
# =====================================================

elif page == "Student Explorer":

    st.title("🔍 Student Profile Explorer")

    selected_user = st.selectbox(
        "Select Student",
        features["UserID"].tolist(),
        format_func=lambda x: (
            f"{x} - " +
            str(features.loc[
                features["UserID"] == x,
                "UserName"
            ].iloc[0])
        )
    )

    student = features[
        features["UserID"] == selected_user
    ].iloc[0]

    st.divider()

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Age", int(student["Age"]))

    c2.metric(
        "Total Enrollments",
        int(student["Total_Enrollments"])
    )

    c3.metric(
        "Categories Explored",
        int(student["Unique_Categories"])
    )

    c4.metric(
        "Assigned Cluster",
        f"Cluster {int(student['Cluster'])}"
    )

    st.subheader("Student Information")

    st.write("**Student Name:**", student["UserName"])

    st.write("**Gender:**", student["Gender"])

    st.write(
        "**Average Spending:**",
        round(student["Average_Spending"], 2)
    )

    st.write(
        "**Average Course Rating:**",
        round(student["Average_Course_Rating"], 2)
    )

    st.subheader("Learning Behaviour")

    level_data = pd.DataFrame({

        "Level": [
            "Beginner",
            "Intermediate",
            "Advanced"
        ],

        "Percentage": [

            student["Beginner_Ratio"] * 100,

            student["Intermediate_Ratio"] * 100,

            student["Advanced_Ratio"] * 100

        ]

    })

    fig = px.pie(
        level_data,
        names="Level",
        values="Percentage",
        title="Student Course Level Preferences"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    history = transaction_data[
        transaction_data["UserID"] == selected_user
    ]

    st.subheader("Enrollment History")

    st.dataframe(
        history[
            [
                "CourseName",
                "CourseCategory",
                "CourseLevel",
                "Amount",
                "TransactionDate"
            ]
        ],
        use_container_width=True
    )


# =====================================================
# COURSE RECOMMENDATIONS
# =====================================================

elif page == "Course Recommendations":

    st.title("🎯 Personalized Course Recommendations")

    st.write(
        "Discover courses based on student preferences "
        "and similar learner behaviour."
    )

    selected_user = st.selectbox(
        "Choose Student",
        features["UserID"].tolist(),
        key="recommend_user"
    )

    student = features[
        features["UserID"] == selected_user
    ].iloc[0]

    st.info(
        f"Assigned Segment: Cluster {int(student['Cluster'])}"
    )

    recommendations = recommend_courses(
        selected_user,
        features,
        transaction_data,
        courses
    )

    if not recommendations.empty:

        category_filter = st.selectbox(

            "Filter by Category",

            ["All"] +
            sorted(
                recommendations["CourseCategory"].dropna().unique()
            )

        )

        level_filter = st.selectbox(

            "Filter by Level",

            ["All"] +
            sorted(
                recommendations["CourseLevel"].dropna().unique()
            )

        )

        filtered = recommendations.copy()

        if category_filter != "All":

            filtered = filtered[
                filtered["CourseCategory"] == category_filter
            ]

        if level_filter != "All":

            filtered = filtered[
                filtered["CourseLevel"] == level_filter
            ]

        st.subheader("Recommended Courses")

        for _, course in filtered.iterrows():

            with st.container(border=True):

                c1, c2 = st.columns([3, 1])

                with c1:

                    st.subheader(
                        course["CourseName"]
                    )

                    st.write(
                        f"Category: {course['CourseCategory']}"
                    )

                    st.write(
                        f"Level: {course['CourseLevel']}"
                    )

                    st.write(
                        f"Course Type: {course['CourseType']}"
                    )

                    st.write(
                        f"Duration: {course['CourseDuration']} hours"
                    )

                with c2:

                    st.metric(
                        "Course Rating",
                        f"{course['CourseRating']}/5"
                    )

                    st.metric(
                        "Price",
                        f"₹{course['CoursePrice']:.2f}"
                    )

                    st.metric(
                        "Recommendation Score",
                        f"{course['Recommendation_Score'] * 100:.1f}%"
                    )

        csv = filtered.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(

            "Download Recommendations",

            data=csv,

            file_name="EduPro_Recommendations.csv",

            mime="text/csv"

        )

    else:

        st.warning(
            "No new course recommendations available."
        )


# =====================================================
# SEGMENT COMPARISON
# =====================================================

elif page == "Segment Comparison":

    st.title("📊 Student Segment Comparison")

    cluster_list = sorted(
        features["Cluster"].unique()
    )

    col1, col2 = st.columns(2)

    with col1:

        cluster1 = st.selectbox(
            "Select First Segment",
            cluster_list,
            key="cluster1"
        )

    with col2:

        cluster2 = st.selectbox(
            "Select Second Segment",
            cluster_list,
            index=min(1, len(cluster_list)-1),
            key="cluster2"
        )

    comparison = features[
        features["Cluster"].isin(
            [cluster1, cluster2]
        )
    ]

    comparison_data = comparison.groupby(
        "Cluster"
    ).agg(

        Students=("UserID", "count"),

        Average_Age=("Age", "mean"),

        Total_Enrollments=("Total_Enrollments", "mean"),

        Average_Spending=("Average_Spending", "mean"),

        Category_Diversity=("Unique_Categories", "mean"),

        Course_Rating=("Average_Course_Rating", "mean")

    ).round(2)

    st.subheader("Comparison Results")

    st.dataframe(
        comparison_data,
        use_container_width=True
    )

    chart_data = comparison.melt(

        id_vars=["Cluster"],

        value_vars=[
            "Total_Enrollments",
            "Unique_Categories",
            "Average_Spending"
        ],

        var_name="Metric",

        value_name="Value"

    )

    fig = px.box(

        chart_data,

        x="Metric",

        y="Value",

        color="Cluster",

        title="Behaviour Comparison Between Segments"

    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# =====================================================
# FOOTER
# =====================================================

st.divider()

st.markdown(
    "<center>EduPro | Student Segmentation & "
    "Personalized Course Recommendation System</center>",
    unsafe_allow_html=True
)