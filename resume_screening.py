# Task 3 - Resume Screening System
# Future Interns - FUTURE_ML_03

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Load data
df = pd.read_csv('Resume.csv')
print("Shape:", df.shape)
print("\nCategories:")
print(df['Category'].value_counts().head(10))

# ML Skills list
ml_skills = ['python', 'machine learning', 'deep learning', 'tensorflow',
             'pytorch', 'sklearn', 'data science', 'neural network',
             'pandas', 'numpy', 'sql', 'statistics', 'nlp']

# Score each resume
def calculate_score(resume_text):
    resume_lower = resume_text.lower()
    matched_skills = [skill for skill in ml_skills if skill in resume_lower]
    return len(matched_skills), matched_skills

df['Resume_str'] = df['Resume_str'].fillna('')
df['Score'], df['Matched_Skills'] = zip(*df['Resume_str'].apply(calculate_score))
df['Match_Percentage'] = (df['Score'] / len(ml_skills) * 100).round(2)

# Top 10 candidates
top10 = df.nlargest(10, 'Score')[['Category', 'Score', 'Match_Percentage', 'Matched_Skills']]
print("\nTop 10 Candidates:")
print(top10)

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
df['Category'].value_counts().head(10).plot(kind='bar', ax=axes[0],
                                             color='skyblue', edgecolor='black')
axes[0].set_title('Top 10 Resume Categories', fontsize=14)
axes[0].tick_params(axis='x', rotation=45)
top10['Score'].plot(kind='bar', ax=axes[1],
                    color='green', edgecolor='black')
axes[1].set_title('Top 10 Candidates Match Score', fontsize=14)
plt.tight_layout()
plt.savefig('resume_screening.png')
plt.show()

# Skill gap analysis
top_resume = df.nlargest(1, 'Score').iloc[0]
matched = top_resume['Matched_Skills']
missing = [skill for skill in ml_skills if skill not in matched]
print(f"\nTop Candidate: {top_resume['Category']}")
print(f"Matched Skills: {matched}")
print(f"Missing Skills: {missing}")
print(f"Match Score: {top_resume['Match_Percentage']}%")
print("✅ Task 3 Complete!")
