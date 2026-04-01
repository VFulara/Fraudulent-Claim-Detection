# Fraudulent Claim Detection — Case Study 2

**Team Members:** Vaibhav Fulara, Tarun Garg, Banetta Nedunchezhiyan, Sharath Hosakote, Karthik Reddy

---

## What is this about?

This is a case study assignment where we worked on a real-world-style problem for a fictional insurance company called **Global Insure**. The company was losing money because of fraudulent claims slipping through their manual review process. Our job was to build a machine learning model that could look at historical claim data and flag which claims are likely fraudulent — before they get approved and paid out.

The dataset had **1,000 insurance claims** with 40 columns covering everything from customer demographics and policy details to incident information and claim amounts. The target column was `fraud_reported` (Y/N).

---

## How we approached it

We followed a step-by-step process as outlined in the assignment:

### 1. Data Preparation & Cleaning
First, we loaded the dataset and took a good look at it — checked the shape, data types, and missing values. A few columns had question marks (`?`) acting as missing value placeholders, which we cleaned up. We also dropped columns that weren't going to be useful for modelling (like unique identifiers, dates, and a meaningless `_c39` column). Missing values in categorical columns were filled using the mode.

### 2. Train-Validation Split
We split the data **70-30** (training vs validation) right after cleaning, and made sure all EDA was done only on the training set. This was important so we don't accidentally let future/validation data influence our analysis.

### 3. Exploratory Data Analysis (EDA)
We did a fairly thorough EDA on the training data — both univariate and bivariate. Some of the interesting things we noticed:
- Claims with **Major Damage** showed a much higher fraud rate compared to minor or trivial damage
- People with certain hobbies (like chess, cross-fit, or skydiving) appeared more often in fraudulent cases
- **Capital loss being exactly 0** was associated with higher fraud rates
- Fraud cases tended to have **higher vehicle claim amounts** and more witnesses
- Most incidents happened in the afternoon hours

### 4. Feature Engineering
This was probably the most creative part. Instead of just using the raw columns, we created 10 new features that we thought would be better predictors of fraud. Things like:
- `is_high_risk_hobby` — flagging policyholders with hobbies correlated with fraud
- `vehicle_to_injury_ratio` — checking if vehicle damage is weirdly high compared to injury claims (classic staged accident pattern)
- `early_claim_flag` — whether a claim was filed too soon after the policy was taken
- `claim_to_premium_ratio` — how much the claim is relative to what was paid in premiums
- `damage_no_police_report` — property damage was claimed but no police report exists

We also encoded all categorical variables and scaled the numerical features before modelling.

### 5. Model Building

We built two models:

**Logistic Regression (GLM — Binomial)**
- Used RFECV (Recursive Feature Elimination with Cross-Validation) to select the most relevant features
- Checked for multicollinearity using VIF scores and removed high-VIF features iteratively
- Analysed p-values to keep only statistically significant variables
- Played around with the probability cutoff — instead of just defaulting to 0.5, we looked at the sensitivity-specificity tradeoff to find a better threshold

**Random Forest**
- Started with a default Random Forest, which gave 100% training accuracy — obviously overfitting
- Used 5-fold cross-validation to get a more honest accuracy estimate (~92%)
- Did hyperparameter tuning using GridSearchCV to find the best combination of tree depth, number of estimators, leaf nodes, etc.
- The tuned model brought training accuracy down to ~93% but generalised much better

---

## Results at a Glance

| Metric | Logistic Regression (Validation) | Random Forest Tuned (Validation) |
|---|---|---|
| Accuracy | 83.67% | **84.67%** |
| Sensitivity (Recall) | 81.08% | 81.08% |
| Specificity | 84.51% | **85.84%** |
| Precision | 63.16% | **65.22%** |
| F1-Score | 71.01% | **72.29%** |
| AUC-ROC | **84.24%** | 83.51% |

---

## What we concluded

The **Tuned Random Forest** performed slightly better on most classification metrics on the validation set.

The **single biggest fraud predictor** turned out to be the hobby risk profile of the insured person. Claims where the insured had certain hobbies were far more likely to be fraudulent. Incident severity was another strong signal — surprisingly, "Major Damage" claims were more fraud-prone than total loss or trivial damage cases, probably because fraudsters try to stay in the mid-range to avoid scrutiny.

Other useful signals included a high vehicle-to-injury ratio (staged accidents), claims with multiple witnesses plus property damage but no police report, and claims filed very early after policy activation.

If Global Insure were to implement this, we'd recommend starting with the Logistic Regression for its interpretability and flagging high-risk claims for manual review based on the probability score. A 0.5 threshold might not be the best choice in practice — depending on the cost of false positives vs false negatives, the cutoff can be adjusted.

---

## Files in this repo

| File | Description |
|---|---|
| `Fraudulent_Claim_Detection_Starter.ipynb` | Main notebook with all analysis, EDA, feature engineering, and model building |
| `insurance_claims.csv` | The dataset used for this assignment |
| `README.md` | This file |
