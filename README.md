# 🎓 EduPro – Student Segmentation and Personalized Course Recommendation System

## 📌 Project Overview

EduPro is a Machine Learning and Data Analytics project designed to understand student learning behaviour and provide personalized course recommendations.

The system uses student demographics, course information and enrollment history to identify different learner segments using K-Means Clustering.

Based on student preferences and similar learner behaviour, the system recommends relevant courses to improve learning experiences and engagement.

## 🎯 Project Objectives

- Understand student learning behaviour.
- Identify different student segments using Machine Learning.
- Discover student course preferences.
- Develop a personalized course recommendation system.
- Provide interactive analytics through a Streamlit dashboard.
- Support data-driven learning decisions.

## 📂 Project Structure

```text
EduPro_Project/
│
├── dataset/
│   ├── Users.csv
│   ├── Courses.csv
│   ├── Transactions.csv
│   └── Teachers.csv
│
├── app.py
│
├── requirements.txt
│
└── README.md
```

## 📊 Datasets

The project uses four datasets:

| Dataset | Description |
|---|---|
| Users | Student information and demographics |
| Courses | Course details, categories, levels and ratings |
| Transactions | Student enrollment and transaction records |
| Teachers | Teacher information |

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Plotly
- Scikit-learn
- Streamlit
- K-Means Clustering
- StandardScaler

## ⚙️ Machine Learning Methodology

### 1. Data Preprocessing
- Load and combine datasets.
- Handle missing values.
- Prepare student-level data.

### 2. Feature Engineering

Important learner-level features:

- Total courses enrolled.
- Number of categories explored.
- Average spending.
- Total spending.
- Average course rating.
- Enrollment frequency.
- Beginner, Intermediate and Advanced course ratios.

### 3. Student Segmentation

K-Means Clustering is used to group students according to their learning behaviour.

Silhouette Score is used to evaluate cluster separation and consistency.

### 4. Course Recommendation

The recommendation system considers:

- Course popularity within the student's cluster.
- Course ratings.
- Preferred course category.
- Preferred course level.

## 🚀 Application Features

### 1. Analytics Dashboard
- Total students.
- Total courses.
- Total enrollments.
- Student age distribution.
- Course category distribution.
- Monthly enrollment trends.

### 2. Student Segmentation
- Student cluster distribution.
- Cluster characteristics.
- Learning behaviour visualization.

### 3. Student Profile Explorer
- Individual student information.
- Assigned student segment.
- Enrollment history.
- Course level preferences.

### 4. Personalized Course Recommendations
- Student-specific course suggestions.
- Course category filtering.
- Course level filtering.
- Recommendation scores.
- Downloadable recommendation reports.

### 5. Segment Comparison
- Compare student groups.
- Analyze enrollment patterns.
- Compare average spending.
- Compare course preferences.

## 💻 Installation and Setup

### Step 1: Clone the Repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

### Step 2: Open Project Folder

```bash
cd EduPro_Project
```

### Step 3: Install Required Libraries

```bash
pip install -r requirements.txt
```

### Step 4: Run Streamlit Application

```bash
streamlit run app.py
```

The application will open in your default web browser.

## 📈 Expected Outcomes

- Identification of different student learning segments.
- Better understanding of student preferences.
- Personalized course recommendations.
- Interactive student analytics.
- Data-driven insights for learning platforms.

## 🔮 Future Enhancements

- Dynamic cluster selection using Elbow and Silhouette methods.
- Advanced recommendation algorithms.
- Collaborative filtering.
- Personalized learning path generation.
- Student engagement prediction.
- Model persistence and scheduled retraining.

## 👨‍💻 Project Information

**Project Name:** Student Segmentation and Personalized Course Recommendation System

**Platform:** EduPro

**Project Domain:** Data Science, Machine Learning and Data Analytics

**Application:** Streamlit Web Application

---

⭐ If you find this project interesting, consider giving the repository a star!