import streamlit as st
import joblib
import re
import string
import nltk
import pandas as pd

from PyPDF2 import PdfReader
from sklearn.metrics.pairwise import cosine_similarity

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize


# ============================================================
# NLTK
# ============================================================

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)


# ============================================================
# INITIALIZATION
# ============================================================

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))


# ============================================================
# LOAD MODELS
# ============================================================

resume_model = joblib.load(
    "models/resume_classifier.pkl"
)

resume_vectorizer = joblib.load(
    "models/tfidf_vectorizer.pkl"
)

label_encoder = joblib.load(
    "models/label_encoder.pkl"
)

job_vectorizer = joblib.load(
    "models/job_vectorizer.pkl"
)

category_to_jobs = joblib.load(
    "models/category_to_jobs.pkl"
)


# ============================================================
# LOAD DATA
# ============================================================

jobs_df = pd.read_csv(
    "data/processed/jobs_recommendation.csv"
)

extra_jobs_df = pd.read_csv(
    "data/processed/extra_jobs_with_skills.csv"
)


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text):

    text = str(text)

    text = text.lower()

    text = re.sub(
        r'http\S+|www\S+',
        ' ',
        text
    )

    text = re.sub(
        r'\S+@\S+',
        ' ',
        text
    )

    text = re.sub(
        r'\+?\d[\d\s\-\(\)]{8,}\d',
        ' ',
        text
    )

    text = re.sub(
        r'\d+',
        ' ',
        text
    )

    text = text.translate(
        str.maketrans(
            '',
            '',
            string.punctuation
        )
    )

    text = re.sub(
        r'\s+',
        ' ',
        text
    ).strip()

    words = word_tokenize(text)

    words = [
        lemmatizer.lemmatize(word)
        for word in words
        if word not in stop_words
    ]

    return " ".join(words)


# ============================================================
# SKILL KEYWORDS
# ============================================================

skill_keywords = {

    "Python",
    "Java",
    "C++",
    "C",
    "SQL",
    "MySQL",
    "PostgreSQL",
    "MongoDB",

    "Machine Learning",
    "Deep Learning",
    "TensorFlow",
    "PyTorch",
    "Keras",

    "NLP",
    "Computer Vision",

    "Docker",
    "Kubernetes",

    "AWS",
    "Azure",
    "GCP",

    "Linux",
    "Git",
    "GitHub",

    "JavaScript",
    "React",
    "Node.js",

    "HTML",
    "CSS",

    "Power BI",
    "Tableau",

    "Pandas",
    "NumPy",
    "Scikit-learn",

    "Spark",
    "Hadoop",

    "DevOps",
    "Cybersecurity",
    "Networking",

    "Data Analysis",
    "Excel",
    "REST API"
}


# ============================================================
# SKILL EXTRACTION
# ============================================================

def extract_skills(text):

    text = str(text).lower()

    found_skills = []

    for skill in skill_keywords:

        if skill.lower() in text:

            found_skills.append(skill)

    return sorted(set(found_skills))


# ============================================================
# JOB TITLE MAPPING
# ============================================================

job_title_mapping = {

    "Software Engineer":
        "Machine Learning Software Engineer",

    "QA Engineer":
        "Machine Learning Software Engineer",

    "Technical Lead":
        "Lead Data Engineer",

    "Engineering Manager":
        "Lead Data Engineer",

    "Data Scientist":
        "Data Scientist",

    "Data Engineer":
        "Data Engineer",

    "Data Analyst":
        "Data Analyst",

    "Business Analyst":
        "Business Analyst",

    "AI Engineer":
        "Machine Learning Engineer",

    "Machine Learning Engineer":
        "Machine Learning Engineer",

    "MLOps Engineer":
        "Machine Learning Engineer",

    "Computer Vision Engineer":
        "Machine Learning Engineer",

    "NLP Engineer":
        "Machine Learning Engineer",

    "Cloud Engineer":
        "AWS Data Engineer",

    "Backend Developer":
        "Database Engineer",

    "Node.js Developer":
        "Database Engineer",

    "Frontend Developer":
        "Software Engineer",

    "React Developer":
        "Software Engineer",

    "Android Developer":
        "Software Engineer",

    "iOS Developer":
        "Software Engineer",

    "DevOps Engineer":
        "AWS Data Engineer",

    "Cybersecurity Analyst":
        "Data Loss Prevention (DLP) Engineer (Symantec)",

    "Power BI Developer":
        "Data Reporting Analyst",

    "Java Developer":
        "Software Engineer",

    "Python Developer":
        "Software Engineer",

    "Product Manager":
        "Business Analyst",

    "Blockchain Developer":
        "Software Engineer"
}


# ============================================================
# JOB RECOMMENDATION
# ============================================================

def recommend_jobs(
    resume_text,
    predicted_category
):

    if predicted_category not in category_to_jobs:

        return None

    filtered_jobs = jobs_df[
        jobs_df["Job_Title"].isin(
            category_to_jobs[predicted_category]
        )
    ].copy()

    if filtered_jobs.empty:

        return None

    filtered_vectors = job_vectorizer.transform(
        filtered_jobs["Search_Text"]
    )

    resume_vector = job_vectorizer.transform(
        [resume_text]
    )

    similarity = cosine_similarity(
        resume_vector,
        filtered_vectors
    ).flatten()

    filtered_jobs["Match %"] = (
        similarity * 100
    ).round(2)

    return filtered_jobs.sort_values(
        by="Match %",
        ascending=False
    ).head(10)


# ============================================================
# SKILL GAP ANALYSIS
# ============================================================

def skill_gap_analysis(
    resume_text,
    recommended_job
):

    candidate_skills = set(
        extract_skills(resume_text)
    )

    mapped_job = job_title_mapping.get(
        recommended_job,
        recommended_job
    )

    matching_jobs = extra_jobs_df[
        extra_jobs_df["title"].str.contains(
            mapped_job,
            case=False,
            na=False
        )
    ]

    if matching_jobs.empty:

        return None

    job_descriptions = " ".join(
        matching_jobs["description"].astype(str)
    )

    required_skills = set(
        extract_skills(job_descriptions)
    )

    matched_skills = (
        candidate_skills
        .intersection(required_skills)
    )

    missing_skills = (
        required_skills
        - candidate_skills
    )

    if len(required_skills) > 0:

        match_percentage = (
            len(matched_skills)
            /
            len(required_skills)
        ) * 100

    else:

        match_percentage = 0

    return {

        "candidate_skills":
            sorted(candidate_skills),

        "required_skills":
            sorted(required_skills),

        "matched_skills":
            sorted(matched_skills),

        "missing_skills":
            sorted(missing_skills),

        "match_percentage":
            round(match_percentage, 2)
    }


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(

    page_title="SmartHire AI",

    page_icon="💼",

    layout="wide",

    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* Main background */

    .stApp {
        background-color: #f7f9fc;
    }


    /* Main title */

    .main-title {
        font-size: 42px;
        font-weight: 800;
        margin-bottom: 0px;
    }


    .subtitle {
        font-size: 18px;
        color: #6b7280;
        margin-bottom: 30px;
    }


    /* Section headings */

    .section-title {
        font-size: 25px;
        font-weight: 700;
        margin-top: 25px;
        margin-bottom: 15px;
    }


    /* Cards */

    .info-card {

        background-color: white;

        padding: 25px;

        border-radius: 15px;

        border: 1px solid #e5e7eb;

        box-shadow: 0px 4px 12px rgba(0,0,0,0.05);

        margin-bottom: 20px;

    }


    /* Category */

    .category-card {

        background-color: white;

        padding: 25px;

        border-radius: 15px;

        border-left: 6px solid #2563eb;

        box-shadow: 0px 4px 12px rgba(0,0,0,0.05);

        font-size: 24px;

        font-weight: 700;

    }


    /* Skill badges */

    .skill {

        display: inline-block;

        padding: 7px 12px;

        margin: 4px;

        border-radius: 20px;

        background-color: #eef2ff;

        font-size: 14px;

        font-weight: 600;

    }


    /* Footer */

    .footer {

        text-align: center;

        color: #6b7280;

        padding: 30px;

        font-size: 14px;

    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("💼 SmartHire AI")

st.sidebar.markdown(
    """
    ### Your AI Career Assistant

    Upload your resume and SmartHire will:

    📄 Analyze your resume

    🎯 Predict your career domain

    💼 Recommend suitable jobs

    🧠 Identify skill gaps
    """
)

st.sidebar.markdown("---")

st.sidebar.info(
    "Upload a PDF resume to begin."
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">💼 SmartHire AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'AI-Powered Resume Screening & Career Recommendation'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# UPLOAD SECTION
# ============================================================

st.markdown(
    '<div class="section-title">📄 Upload Your Resume</div>',
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader(

    "Upload your resume in PDF format",

    type=["pdf"],

    label_visibility="collapsed"
)


# ============================================================
# PROCESS RESUME
# ============================================================

if uploaded_file is not None:

    st.success(
        "✅ Resume uploaded successfully!"
    )

    # --------------------------
    # Extract PDF text
    # --------------------------

    reader = PdfReader(
        uploaded_file
    )

    resume_text = ""

    for page in reader.pages:

        page_text = page.extract_text()

        if page_text:

            resume_text += (
                page_text + "\n"
            )


    # --------------------------
    # Resume Preview
    # --------------------------

    with st.expander(
        "📃 View Extracted Resume",
        expanded=False
    ):

        st.text_area(
            "Resume Text",
            resume_text,
            height=250
        )


    # --------------------------
    # Clean Resume
    # --------------------------

    cleaned_resume = clean_text(
        resume_text
    )


    # --------------------------
    # Classification
    # --------------------------

    resume_vector = (
        resume_vectorizer.transform(
            [cleaned_resume]
        )
    )

    prediction = (
        resume_model.predict(
            resume_vector
        )
    )

    predicted_category = (
        label_encoder.inverse_transform(
            prediction
        )[0]
    )


    # ========================================================
    # CLASSIFICATION RESULT
    # ========================================================

    st.markdown(
        '<div class="section-title">'
        '🎯 Resume Analysis'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        f"""
        <div class="category-card">

        Predicted Career Domain

        <br>

        <span style="color:#2563eb;">
        {predicted_category}
        </span>

        </div>
        """,
        unsafe_allow_html=True
    )


    # ========================================================
    # JOB RECOMMENDATIONS
    # ========================================================

    recommended_jobs = recommend_jobs(
        cleaned_resume,
        predicted_category
    )


    st.markdown(
        '<div class="section-title">'
        '💼 Recommended Jobs'
        '</div>',
        unsafe_allow_html=True
    )


    if recommended_jobs is None:

        st.warning(
            "No matching jobs are available "
            "for this category."
        )

    else:

        st.dataframe(

            recommended_jobs[
                [
                    "Job_Title",
                    "Company",
                    "City",
                    "Experience_Level",
                    "Education_Required",
                    "Match %"
                ]
            ],

            use_container_width=True,

            hide_index=True
        )


        # ====================================================
        # SKILL GAP
        # ====================================================

        best_job = (
            recommended_jobs.iloc[0]["Job_Title"]
        )

        gap_result = skill_gap_analysis(
            cleaned_resume,
            best_job
        )


        st.markdown(
            '<div class="section-title">'
            '🧠 Skill Gap Analysis'
            '</div>',
            unsafe_allow_html=True
        )


        if gap_result is None:

            st.warning(
                "Detailed job description is "
                "not available for this job."
            )

        else:

            # -------------------------------
            # Match Percentage
            # -------------------------------

            match = (
                gap_result[
                    "match_percentage"
                ]
            )

            st.metric(
                "🎯 Skill Match",
                f"{match}%"
            )

            st.progress(
                min(match / 100, 1.0)
            )


            # -------------------------------
            # Skills
            # -------------------------------

            col1, col2 = st.columns(2)


            with col1:

                st.markdown(
                    "### 🟢 Your Skills"
                )

                if gap_result[
                    "candidate_skills"
                ]:

                    for skill in gap_result[
                        "candidate_skills"
                    ]:

                        st.markdown(
                            f'<span class="skill">'
                            f'{skill}'
                            f'</span>',
                            unsafe_allow_html=True
                        )

                else:

                    st.write(
                        "No skills detected."
                    )


            with col2:

                st.markdown(
                    "### 📌 Required Skills"
                )

                if gap_result[
                    "required_skills"
                ]:

                    for skill in gap_result[
                        "required_skills"
                    ]:

                        st.markdown(
                            f'<span class="skill">'
                            f'{skill}'
                            f'</span>',
                            unsafe_allow_html=True
                        )

                else:

                    st.write(
                        "No required skills detected."
                    )


            # -------------------------------
            # Matched Skills
            # -------------------------------

            st.markdown(
                "### ✅ Matched Skills"
            )

            if gap_result[
                "matched_skills"
            ]:

                st.write(
                    ", ".join(
                        gap_result[
                            "matched_skills"
                        ]
                    )
                )

            else:

                st.write(
                    "No matching skills found."
                )


            # -------------------------------
            # Missing Skills
            # -------------------------------

            st.markdown(
                "### 🔴 Skills to Improve"
            )

            if gap_result[
                "missing_skills"
            ]:

                for skill in gap_result[
                    "missing_skills"
                ]:

                    st.write(
                        f"• {skill}"
                    )

            else:

                st.success(
                    "🎉 You have all the "
                    "identified required skills!"
                )


# ============================================================
# EMPTY STATE
# ============================================================

else:

    st.info(
        "👆 Upload a PDF resume above "
        "to start your SmartHire analysis."
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">

    SmartHire AI • Resume Intelligence & Career Recommendation

    </div>
    """,
    unsafe_allow_html=True
)