
# SmartHire AI

## AI-Powered Resume Screening, Job Recommendation and Skill Gap Analysis System

SmartHire AI is a machine learning and natural language processing based career assistance system designed to analyze resumes, predict relevant career domains, recommend suitable job opportunities, and identify potential skill gaps.

The system combines Natural Language Processing (NLP), TF-IDF vectorization, supervised machine learning, XGBoost classification, cosine similarity, and skill-based analysis into a Streamlit web application.

---

## Project Overview

Traditional job searching requires candidates to manually search through large numbers of job listings and determine whether their skills and qualifications match each position.

SmartHire AI aims to simplify this process by analyzing a candidate's resume and providing three major outputs:

1. Resume career-domain classification
2. Relevant job recommendations
3. Skill gap analysis

The overall workflow is:


Resume PDF
    |
    v
Text Extraction
    |
    v
Text Preprocessing
    |
    v
TF-IDF Representation
    |
    v
Resume Classification
    |
    v
Predicted Career Domain
    |
    v
Relevant Job Filtering
    |
    v
Cosine Similarity
    |
    v
Top Job Recommendations
    |
    v
Skill Gap Analysis
    |
    v
Matched and Missing Skills


---

## Key Features

### 1. Resume Classification

The application accepts a resume in PDF format and extracts the text using PDF parsing.

The extracted text is then processed using an NLP preprocessing pipeline consisting of:

* Lowercase conversion
* URL removal
* Email removal
* Phone number removal
* Number removal
* Punctuation removal
* Extra whitespace removal
* Tokenization
* Stopword removal
* Lemmatization

The cleaned resume is converted into numerical features using TF-IDF vectorization.

An XGBoost classifier is then used to predict the most relevant resume category.

The dataset contains 24 resume categories, including:

* ACCOUNTANT
* ADVOCATE
* AGRICULTURE
* APPAREL
* ARTS
* AUTOMOBILE
* AVIATION
* BANKING
* BPO
* BUSINESS-DEVELOPMENT
* CHEF
* CONSTRUCTION
* CONSULTANT
* DESIGNER
* DIGITAL-MEDIA
* ENGINEERING
* FINANCE
* FITNESS
* HEALTHCARE
* HR
* INFORMATION-TECHNOLOGY
* PUBLIC-RELATIONS
* SALES
* TEACHER

---

### 2. Job Recommendation

After predicting the candidate's career category, the system narrows the available job listings using the predicted category.

This prevents the system from comparing a resume against every unrelated job in the dataset.

The filtered jobs are then represented using TF-IDF and compared with the cleaned resume using cosine similarity.

The jobs are ranked according to their similarity score and the top recommendations are displayed.

The application displays:

* Job Title
* Company
* City
* Experience Level
* Education Requirement
* Match Score

The system displays the top 10 recommended jobs.

---

### 3. Skill Gap Analysis

The skill gap module analyzes the candidate's resume and compares the detected skills against skills identified from relevant job descriptions.

The system produces:

* Candidate Skills
* Required Skills
* Matched Skills
* Missing Skills
* Skill Match Percentage

This allows candidates to understand which skills are already present in their resume and which additional skills may be useful for their recommended roles.

---

## System Architecture


                         SmartHire AI
                              |
                              v
                     Upload Resume PDF
                              |
                              v
                     Extract Resume Text
                              |
                              v
                    Text Preprocessing
                              |
                              v
                       TF-IDF Vectorizer
                              |
                              v
                    XGBoost Classifier
                              |
                              v
                  Predicted Resume Category
                              |
                              v
                    Filter Relevant Jobs
                              |
                              v
                    Cosine Similarity
                              |
                              v
                     Top 10 Job Results
                              |
                              v
                       Skill Analysis
                              |
                              v
                +-------------+-------------+
                |                           |
                v                           v
          Matched Skills              Missing Skills
                |                           |
                +-------------+-------------+
                              |
                              v
                     Skill Match Percentage


---

## Machine Learning Pipeline

### Resume Classification Pipeline


Resume Text
    |
    v
Data Cleaning
    |
    v
Text Preprocessing
    |
    v
TF-IDF Vectorization
    |
    v
Train/Test Split
    |
    v
Model Training
    |
    +----------------------+
    |                      |
    v                      v
Traditional ML Models    XGBoost
    |                      |
    +----------+-----------+
               |
               v
        Model Evaluation
               |
               v
       Final Classifier


Several machine learning models were evaluated during development, including:

* Logistic Regression
* Naive Bayes
* Random Forest
* Linear SVM
* XGBoost

The XGBoost model achieved approximately 79–80% accuracy on the evaluated test set and was selected for the final application.

---

## Job Recommendation Pipeline


Candidate Resume
       |
       v
Predicted Career Category
       |
       v
Filter Relevant Job Titles
       |
       v
Create Job Text Representation
       |
       v
TF-IDF Vectorization
       |
       v
Cosine Similarity
       |
       v
Similarity Ranking
       |
       v
Top 10 Recommended Jobs


The category prediction is used as the first filtering stage. Cosine similarity is then applied only to the relevant job subset.

This two-stage approach reduces irrelevant comparisons and provides more focused recommendations.

---

## Skill Gap Analysis Pipeline


Candidate Resume
       |
       v
Candidate Skill Extraction
       |
       v
Recommended Job
       |
       v
Job Title Mapping
       |
       v
Relevant Job Descriptions
       |
       v
Required Skill Extraction
       |
       v
Skill Comparison
       |
       +----------------------+
       |                      |
       v                      v
Matched Skills          Missing Skills
       |                      |
       +----------+-----------+
                  |
                  v
          Skill Match Percentage


Because the structured jobs dataset and the additional job-description dataset contain different job-title naming conventions, a job-title mapping layer is used to connect related job roles when necessary.

---

## Datasets

The project uses three main datasets.

### 1. Resume Dataset

The resume dataset contains resume text and corresponding category labels.

Main columns include:


Resume_str
Category


This dataset is used for supervised resume classification.

The final cleaned and preprocessed resume dataset contains approximately 2,481 usable resume records across 24 categories.

---

### 2. Jobs Dataset

The structured jobs dataset contains information about available job opportunities.

Important fields include:


Job_Title
Company
City
Experience_Level
Skills_Required
Education_Required


This dataset is used for job filtering and recommendation.

The dataset contains 5,000 job records.

---

### 3. Extra Jobs Dataset

The additional job-description dataset contains job titles and descriptions.

Important fields include:


title
description


This dataset is primarily used for skill-gap analysis.

The dataset contains additional job descriptions that help identify skills associated with different roles.

---

## Data Preprocessing

The preprocessing pipeline was applied to the text-based datasets before machine learning and recommendation tasks.

The main preprocessing operations include:

1. Convert text to lowercase
2. Remove URLs
3. Remove email addresses
4. Remove phone numbers
5. Remove numeric values
6. Remove punctuation
7. Remove extra whitespace
8. Tokenize text
9. Remove stopwords
10. Lemmatize words

The resulting cleaned text is used for TF-IDF vectorization and similarity calculations.

---

## Exploratory Data Analysis

Exploratory Data Analysis was performed before model development to understand the structure and distribution of the datasets.

The analysis included:

* Resume category distribution
* Job experience-level distribution
* Education requirement distribution
* Job distribution across cities
* Most frequently required skills

The resume categories were relatively distributed across many categories, although some categories contained fewer samples than others.

The job dataset contained a mixture of experience levels, education requirements, cities, and technical skills.

---

## Model Evaluation

The resume classification task was treated as a multi-class classification problem.

The following models were evaluated:

| Model               | Approximate Accuracy |
| ------------------- | -------------------: |
| Naive Bayes         |                  55% |
| Logistic Regression |                  66% |
| Linear SVM          |                  72% |
| Random Forest       |                  75% |
| XGBoost             |               79–80% |

XGBoost provided the strongest overall performance among the evaluated models and was therefore selected for the final resume classification component.

Performance was also evaluated using:

* Precision
* Recall
* F1-score
* Classification report
* Confusion matrix

---

## Technologies Used

### Programming Language

* Python

### Machine Learning

* Scikit-learn
* XGBoost

### Natural Language Processing

* NLTK
* TF-IDF

### Data Processing

* Pandas
* NumPy

### Similarity

* Cosine Similarity

### Web Application

* Streamlit

### PDF Processing

* PyPDF2

### Model Serialization

* Joblib

### Development Tools

* Google Colab
* Visual Studio Code
* Git
* GitHub

---

## Project Structure


SmartHire/
|
|-- README.md
|-- requirements.txt
|-- .gitignore
|-- app.py
|
|-- data/
|   |-- raw/
|   |-- interim/
|   |-- processed/
|       |-- resume_clean.csv
|       |-- resume_preprocessed.csv
|       |-- jobs_clean.csv
|       |-- jobs_preprocessed.csv
|       |-- extra_jobs_clean.csv
|       |-- extra_jobs_preprocessed.csv
|       |-- extra_jobs_with_skills.csv
|       |-- category_mapping.csv
|
|-- notebooks/
|   |-- 01_eda.ipynb
|   |-- 02_resume_classifier.ipynb
|   |-- 03_recommender.ipynb
|   |-- 04_clustering_topics.ipynb
|   |-- 05_fit_predictor.ipynb
|
|-- models/
|   |-- resume_classifier.pkl
|   |-- tfidf_vectorizer.pkl
|   |-- label_encoder.pkl
|   |-- job_vectorizer.pkl
|   |-- category_to_jobs.pkl
|
|-- utils/
|
|-- reports/
|
|-- tests/
|
|-- smarthire_env/


The `smarthire_env` directory is a local Python virtual environment and should not be uploaded to GitHub.

---

## Saved Models

The application uses serialized machine learning and preprocessing components.

### Resume Classifier


models/resume_classifier.pkl


Stores the trained XGBoost resume classification model.

### TF-IDF Vectorizer


models/tfidf_vectorizer.pkl


Stores the TF-IDF vectorizer used to transform resume text into the same feature representation used during model training.

### Label Encoder

models/label_encoder.pkl


Stores the mapping between numerical class labels and resume category names.

### Job Vectorizer


models/job_vectorizer.pkl


Stores the vectorizer used for job recommendation and cosine similarity.

### Category-to-Jobs Mapping


models/category_to_jobs.pkl


Stores the mapping used to connect predicted resume categories with relevant job titles.

---

## Installation

Clone the repository:


git clone https://github.com/YOUR_USERNAME/SmartHire.git


Navigate to the project directory:


cd SmartHire


Create a virtual environment:


python -m venv smarthire_env


Activate the environment on Windows:


smarthire_env\Scripts\activate


Install the required dependencies:


pip install -r requirements.txt


---

## Running the Application

Start the Streamlit application using:


streamlit run app.py


The application will open in a web browser.

The user can then:

1. Upload a PDF resume.
2. View the extracted resume text.
3. Obtain the predicted career category.
4. View the top recommended jobs.
5. View the skill match percentage.
6. View matched skills.
7. View missing skills.

---

## Application Workflow

The complete user workflow is:


Upload Resume
      |
      v
Extract PDF Text
      |
      v
Clean and Preprocess Text
      |
      v
Predict Career Category
      |
      v
Filter Relevant Jobs
      |
      v
Calculate Cosine Similarity
      |
      v
Display Top 10 Jobs
      |
      v
Analyze Skills
      |
      v
Display Skill Gap


---

## Results

The final resume classification model achieved approximately 79–80% accuracy on the evaluated test set.

The recommendation component successfully ranks relevant job listings using cosine similarity after category-based filtering.

The skill-gap component identifies candidate skills and compares them with skills extracted from relevant job descriptions.

The Streamlit application integrates these components into a single user-facing system.

---

## Limitations

The current system has several limitations:

1. Resume classification accuracy is not perfect and may produce incorrect categories for some resumes.

2. The resume dataset and job dataset use different naming conventions for career categories and job titles.

3. Job-title mapping is therefore required for some roles during skill-gap analysis.

4. Skill extraction currently uses a predefined skill keyword list and does not represent every possible technical or domain-specific skill.

5. Job recommendations are based primarily on textual similarity and category filtering.

6. The current system does not use real-time job listings.

7. Salary, location preference, and other candidate-specific constraints are not currently used as primary ranking factors.

8. The system is an academic prototype and should not be treated as a guaranteed employment or recruitment decision system.

---

## Future Improvements

Future versions of SmartHire AI could include:

* Transformer-based resume understanding using BERT or similar models
* Sentence embeddings for improved semantic job matching
* A larger and more comprehensive skill ontology
* Improved skill extraction using NLP models
* Experience-level matching
* Education-level matching
* Salary-based filtering
* Location preference matching
* Remote/hybrid preference matching
* Real-time job data integration
* Personalized job ranking
* Resume improvement recommendations
* Resume quality scoring
* Automated resume section analysis
* Downloadable candidate analysis reports
* Online deployment

---

## Project Objective

The primary objective of SmartHire AI is to create an intelligent resume analysis and job recommendation system that can assist candidates in understanding their compatibility with available job opportunities.

The system combines classification, information retrieval, similarity analysis, and skill-gap analysis into a single workflow.

The final objective can be summarized as:


Resume
  |
  v
Career Domain
  |
  v
Relevant Jobs
  |
  v
Job Similarity
  |
  v
Skill Compatibility
  |
  v
Skill Gap
  |
  v
Career Improvement Suggestions


---

## Conclusion

SmartHire AI demonstrates how machine learning and natural language processing can be combined to build a practical career assistance application.

The project covers the complete machine learning workflow, including:

* Data preprocessing
* Exploratory data analysis
* Feature engineering
* TF-IDF vectorization
* Multi-class classification
* Model evaluation
* Job recommendation
* Cosine similarity
* Skill extraction
* Skill gap analysis
* Model serialization
* Streamlit application development

The resulting system provides an end-to-end prototype for intelligent resume analysis and job recommendation.

---

## Author

Satwik Guptha

B.Tech - Artificial Intelligence and Data Science

---

## Disclaimer

SmartHire AI is an academic and project prototype developed for demonstrating machine learning, natural language processing, recommendation systems, and web application development.

The predictions, job recommendations, and skill-gap results generated by the system are informational and should not be considered guaranteed employment recommendations or professional recruitment decisions.


