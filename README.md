# 📊 Data Science Projects

This repository contains two end-to-end **data-to-model pipeline** projects demonstrating both **Classification** and **Regression** techniques using Python (Pandas, NumPy, Scikit-learn).

---

## 📂 Repository Structure
Data_Science_Project/
│── Subsidy_question.py
│── Linear_Regression_cars.py
│── README.md

---

## 𝟭. 𝗖𝗹𝗮𝘀𝘀𝗶𝗳𝗶𝗰𝗮𝘁𝗶𝗼𝗻 𝗣𝗿𝗼𝗷𝗲𝗰𝘁: 𝗜𝗻𝗰𝗼𝗺𝗲-𝗕𝗮𝘀𝗲𝗱 𝗦𝘂𝗯𝘀𝗶𝗱𝘆 𝗣𝗿𝗲𝗱𝗶𝗰𝘁𝗶𝗼𝗻  
**File:** `Subsidy_question.py`

This project addressed a **classification problem** to predict whether an individual's income is **≤ $50,000** or **> $50,000**, which can be used for determining **subsidy eligibility**.

### 𝗗𝗮𝘁𝗮 𝗖𝗹𝗲𝗮𝗻𝗶𝗻𝗴 & 𝗣𝗿𝗲𝗽𝗿𝗼𝗰𝗲𝘀𝘀𝗶𝗻𝗴
- **Initial Cleanup:** Dataset initially showed no null values, but placeholder “?” values were found in `JobType` and `occupation`.  
- **Missing Value Handling:** Converted “?” to NaN and dropped missing rows (~1,816 records) → Final dataset: **30,162 records**.  
- **Feature Analysis:** Used `pd.crosstab` to analyze categorical features (`EdType`, `occupation`, `gender`) against the target `SalStat`.  
- **Encoding:** Mapped `SalStat` to binary (≤$50,000 → 0, >$50,000 → 1), applied **one-hot encoding** → dataset expanded to **94 input columns**.  

### 𝗠𝗼𝗱𝗲𝗹 𝗕𝘂𝗶𝗹𝗱𝗶𝗻𝗴 & 𝗘𝘃𝗮𝗹𝘂𝗮𝘁𝗶𝗼𝗻
- Train/Test split: **70% train / 30% test**.  
- **Logistic Regression:** Accuracy = **83.61%**  
- **K-Nearest Neighbors (KNN, k=5):** Accuracy = **83.92%**

---

## 𝟮. 𝗥𝗲𝗴𝗿𝗲𝘀𝘀𝗶𝗼𝗻 𝗣𝗿𝗼𝗷𝗲𝗰𝘁: 𝗣𝗿𝗲-𝗢𝘄𝗻𝗲𝗱 𝗖𝗮𝗿 𝗣𝗿𝗶𝗰𝗲 𝗣𝗿𝗲𝗱𝗶𝗰𝘁𝗶𝗼𝗻  
**File:** `Linear_Regression_cars.py`

This project used **Linear Regression** and **Random Forest Regressor** to predict the **continuous price** of pre-owned cars.

### 𝗗𝗮𝘁𝗮 𝗖𝘂𝗿𝗮𝘁𝗶𝗼𝗻 & 𝗙𝗲𝗮𝘁𝘂𝗿𝗲 𝗘𝗻𝗴𝗶𝗻𝗲𝗲𝗿𝗶𝗻𝗴
- **Initial Data:** 50,001 records across 19 columns.  
- **Column Dropping:** Removed 5 irrelevant columns (`dateCrawled`, `dateCreated`, `postalCode`, etc.).  
- **Deduplication:** Removed 470 duplicate records.  
- **Outlier Treatment:**  
  - `yearOfRegistration`: 1980–2025  
  - `price`: $100–$150,000  
  - `powerPS`: 10–500 PS  
  → Final dataset after cleaning: **42,417 records**.  
- **Feature Creation:** `Age = 2025 – yearOfRegistration – (monthOfRegistration / 12)`.  
- **Dropped Low-Impact Features:** `seller`, `offerType`, `abtest`.  
- **Final Preprocessing:** Removed remaining NaNs → **32,765 records**. Converted categorical features to dummy variables → **300 input columns**.  
- **Target Transformation:** Applied **logarithmic transformation** to `price`.  

### 𝗠𝗼𝗱𝗲𝗹 𝗧𝗿𝗮𝗶𝗻𝗶𝗻𝗴 & 𝗖𝗼𝗺𝗽𝗮𝗿𝗮𝘁𝗶𝘃𝗲 𝗥𝗲𝘀𝘂𝗹𝘁𝘀
- Train/Test Split: 70:30 → **22,872 train / 9,803 test**  
- **Baseline (Mean Prediction):** RMSE = 1.143  

**Models:**  
- **Linear Regression** → RMSE: **0.4863**, R²: **0.8191**  
- **Random Forest Regressor** → RMSE: **0.4877**, R²: **0.8181**  

### 𝗖𝗼𝗻𝗰𝗹𝘂𝘀𝗶𝗼𝗻
Linear Regression provided the best performance, explaining **~81.9% variance** in car price with lower RMSE.  

---

## 🛠️ Tech Stack
- Python (Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn)  
- Jupyter/Spyder IDE  
- Data preprocessing, feature engineering, model training & evaluation  

---

## 📌 Key Learnings
- Importance of **robust data cleaning** and handling missing values.  
- **Feature engineering** significantly improves model accuracy.  
- Target transformation (log scale) helps stabilize regression performance.  
- Comparative model analysis provides better decision-making.  

---
