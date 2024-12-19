## Group Members:
- Jai Advitheeya Lella     50607407
- Prathyusha Reddy Allam  50613111
- Niharika Reddy Katakam    50610925
- Kundavaram Joseph Sujith Kumar  50600443

## Code Location and Analysis Breakdown
- Each member's individual codes are under the folder "codes".
- Each member's files are named as their first name
- Everyone's files have the same code for phase 1
- Each member's Phase 2 code starts after the "Phase 2" heading as a markdown cell

 
### Data Preprocessing 
- Data loading and initial exploration
- Categorical variable encoding (Geography and Gender)
- Feature scaling using StandardScaler
- Train-test split implementation

## Models used by each team member
### Jai Advitheeya Lella
- Random Forest
- Naive Bayes
- Neural Network
### Prathyusha 
- Logistic Regression
- Gradient Boosting
### Sujith
- XG Boost
- Decision Tree
### Niharika
- Linear SVM
- KNN



### Final Analysis and Comparison
- Comparative analysis of all three models
- Performance metrics visualization
- Comprehensive markdown report with findings

## Required Libraries
- sklearn
- pandas
- numpy
- matplotlib
- seaborn
- tensorflow/keras
- XGboost

## Reports
- Each persons report is under the folder reports


# Phase 3:
# Data Intensive Computing Project - Phase 3

## Team Members and Questions
- Niharika Reddy Katakam (50610925)
    - Question 1: Analysis of how churn is impacted by customer age


- Prathyusha Reddy Allam (50613222)
    - Question 2: Analysis of how churn is impacted by account balance and estimated salary

- Kundavaram Joseph Sujith Kumar (50600443)
    - Question 3: Analysis of how churn is impacted by credit score


- Jai Advitheeya Leela (50607407)
    - Question 4: Analysis of how churn is impacted by tenure and number of products
    


## Project Structure
```
main folder contains the following:
└── app/
    ├── app.py                # Streamlit application file
    └── bank_customers.db     # SQLite database file
└── exp/
    └── exp.ipynb            # Final notebook including phase 3
└── README.md                # Project documentation
└── Reports/
    ├── report_phase3.py      # Final project report
(Video file size is too large to add into GITHUB, have submitted it in the UBLearns platform)
```

## Requirements
```
streamlit
pandas
numpy
scikit-learn
joblib
matplotlib
seaborn
```

## App Installation Steps
1. put all the files in one folder
2. Run the Streamlit application:
in terminal:
 streamlit run app.py
```

# Highlights

1. Implementation of a robust SQLite database system for persistent data storage.

2. App highlights: Development of a Streamlit-based web application integrating:
   - Real-time individual churn predictions
   - Comprehensive customer segmentation analysis
   - Feature importance visualization
   - Retention strategy recommendations

3. Model Accuracy of 86% for both Neural Network and Random Forest Models

4. Detailed data analysis through visualizations in the app and notebook.

5. Handled complex and imbalanced dataset via thorough data cleaning processes.
