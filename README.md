# Machine Learning estimation of creditworthiness

This repository contains an end-to-end machine learning project to estimate a bank customer’s **creditworthiness** and support decisions on **credit card issuance**. The goal is to build a model that predicts a binary TARGET where 1 = high creditworthiness (consistent installment payments) and 0 = otherwise, using anonymized customer profile data. The project addresses practical ML challenges, such as handling of mixed data types, class-imbalance strategies and evaluation with metrics that reflect real-world trade-offs. The full project is contained in the [`creditworthiness_ml_estimation.ipynb`](https://github.com/lgucrl/machine-learning-creditworthiness-estimation/blob/main/creditworthiness_ml_estimation.ipynb) notebook.

---

## Dataset

The project is based on a dataset containing **~338,000 records**, where each row represents a customer profile, with the following **19 variables** (mixed numeric + categorical):

- `ID`: customer identification number  
- `CODE_GENDER`: customer's gender  
- `FLAG_OWN_CAR`: indicator of car ownership  
- `FLAG_OWN_REALTY`: indicator of home ownership  
- `CNT_CHILDREN`: number of children  
- `AMT_INCOME_TOTAL`: annual income  
- `NAME_INCOME_TYPE`: type of income  
- `NAME_EDUCATION_TYPE`: level of education  
- `NAME_FAMILY_STATUS`: family status  
- `NAME_HOUSING_TYPE`: type of housing  
- `DAYS_BIRTH`: number of days since birth (commonly stored as a negative value)  
- `DAYS_EMPLOYED`: number of days since the date of hiring (if positive, indicates days since becoming unemployed)  
- `FLAG_MOBIL`: presence of a cell phone number  
- `FLAG_WORK_PHONE`: presence of a work phone number  
- `FLAG_PHONE`: presence of a phone number  
- `FLAG_EMAIL`: presence of an email address  
- `OCCUPATION_TYPE`: type of employment  
- `CNT_FAM_MEMBERS`: number of family members  
- `TARGET`: binary outcome label described above  

The `TARGET` variable is strongly imbalanced (~91% class `0` vs ~9% class `1`), so the workflow includes resampling strategies and the selection of metrics suitable for imbalanced classification.

---

## Project workflow

1. **Data loading and initial validation**  
   The dataset is loaded and a basic inspection is performed, including check for data types and missing values, to determine what needs cleaning and prevent unexpected issues for the modeling process.

2. **Exploratory Data Analysis (EDA)**  
   Data distributions are explored to understand how variables behave and how they relate to `TARGET`. Numerical variables (e.g, `AMT_INCOME_TOTAL`, `DAYS_BIRTH`, `DAYS_EMPLOYED`) are examined for outliers and skew; categorical variables are checked for rare levels and dominant classes. The analysis explicitly investigates the meaning of `DAYS_EMPLOYED` and highlights the target imbalance, motivating later decisions like resampling and the use of precision/recall-focused evaluation rather than relying on accuracy alone.

3. **Data cleaning and feature engineering**  
   Some transformations are applied to make the data more meaningful and model-ready. These actions include, e.g., handling missing values for `OCCUPATION_TYPE` (split into “Unemployed” vs “Other” using `DAYS_EMPLOYED`, then refined using `NAME_INCOME_TYPE`) and consolidating extremely rare categories (e.g., grouping high counts in `CNT_CHILDREN` or `CNT_FAM_MEMBERS` into buckets like “3+” / “5+”).

4. **Encoding categorical features and scaling**  
   Binary flags (e.g., `CODE_GENDER`, `FLAG_OWN_CAR`) are converted into numeric form and one-hot encoding is applied to multi-class categorical variables (e.g. `NAME_INCOME_TYPE`, `OCCUPATION_TYPE`), producing a fully numeric feature matrix that can be handlled by a wide range of estimators. Features are standardized to support models sensitive to feature scale.

5. **Train/test split and imbalance handling**  
   A stratified train/test split is created to preserve the real-world class ratio. To address imbalance during training, alternate training set variants are built using combined oversampling and undersampling to reach balanced class distributions. The test set is kept untouched to ensure a realistic evaluation.

6. **Model training and evaluation**  
   Multiple classifiers (SGD, Random Forest and MLP) are trained across the original and resampled training sets. They are then evaluated and compared using confusion matrices and different classification metrics (accuracy, precision, recall, F1, ROC-AUC, and PR-AUC). Lastly, trade-offs between identifying creditworthy customers (recall for class `1`) and controlling operational cost (precision and false-positive rates) are discussed.
   
