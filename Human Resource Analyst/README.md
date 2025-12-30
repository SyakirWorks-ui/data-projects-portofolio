# Human Resource Analysis: Workforce Insights & Attrition Analytics

This project provides a comprehensive data analysis of human resource metrics to identify patterns in employee performance, engagement, and attrition. By leveraging data-driven insights, this analysis aims to support HR departments in making strategic decisions to improve employee retention and workforce productivity.

## 📌 Project Overview
The core objective of this analysis is to explore the factors influencing employee turnover and performance. This includes analyzing salary distributions, department-specific trends, and the correlation between employee satisfaction and retention.

## 📂 Repository Structure
The project is organized to ensure transparency and reproducibility of the analysis:

* **`data/`**: Divided into `raw/` for the original dataset and `processed/` for the cleaned versions used in the analysis.
* **`notebooks/`**: Contains the Jupyter Notebook (`Hr_analysis.ipynb`) documenting the step-by-step Exploratory Data Analysis (EDA).
* **`scripts/`**: Includes modular Python scripts for `feature_engineering` and data transformation.
* **`visualizations/`**: (Planned) Will store key charts such as attrition rates, salary heatmaps, and demographic distributions.
* **`reports/`**: (Planned) Will contain the executive summary and strategic recommendations for stakeholders.

## 🛠️ Data Processing Pipeline
1.  **Data Cleaning**: Handling missing values, removing duplicates, and standardizing categorical variables within the HR dataset.
2.  **Exploratory Data Analysis (EDA)**: Investigating key metrics such as average tenure, turnover rate by department, and gender pay equality.
3.  **Feature Engineering**: Derived new metrics to better quantify employee engagement and flight risk.
4.  **Insight Generation**: Identifying actionable trends to help reduce unwanted attrition.

### Key Insights from Employee Retention Analysis:
1. **Total Attrition**: Out of 311 employees, 104 have left the company (33% Attrition Rate).
2. **Departmental Impact**: The **Production** department experiences the highest turnover rate, while the **Sales** department maintains a relatively stable retention rate compared to other technical roles.
3. **Primary Exit Drivers**: The top reasons for employee termination include "Another position," "Unhappy," and "More money," as identified through the Top N analysis of termination reasons.
4. **Tenure Analysis**: Most employees who resigned had a specific tenure range, with recruitment sources like Google Search and LinkedIn showing different average employee lifespans.

## 💻 How to Use
1.  **Clone the repository**:
    ```bash
    git clone [https://www.kaggle.com/datasets/rhuebner/human-resources-data-set]
    ```
2.  **Explore the Analysis**: Navigate to the `notebooks/` folder to view the full analytical workflow.
3.  **Review the Data**: The cleaned datasets are available in `data/processed/` for further experimentation.

---
**Author**: [SyakirWorks-ui]  
**Tools Used**: Python (Pandas, Matplotlib, Seaborn), Jupyter Notebook.

