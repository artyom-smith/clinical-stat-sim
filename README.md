# Clinical Trial Data Simulator & Analyzer (clinical-stat-sim)

A lightweight Python module designed to demonstrate a full end-to-end data pipeline for clinical research. It allows users to generate synthetic dementia patient cohorts with customized anomalies, clean and preprocess messy data, and automatically run descriptive statistical analysis with visual plots.

Perfect for quick prototyping, testing statistical workflows, or simulating clinical trial environments.

---

## 🌟 Key Features

* **Data Simulation (`TrialDataset`)**: Generates randomized datasets of hypothetical dementia patients split into Control and Intervention groups. Supports corruption injection (missing values) and cross-laboratory data merging.
* **Robust Preprocessing (`DataPreprocessor`)**: Cleans gender strings, standardizes data types, handles missing values based on strictness modes, and performs feature engineering (e.g., education levels).
* **Automated Statistical Analysis (`DataAnalyzer`)**: 
  * Accepts multiple formats: `.csv`, `.xlsx`, `.xls`, `.db` (SQLite), or `pandas.DataFrame`.
  * Computes descriptive metrics and statistical significance (p-values) using parametric and non-parametric tests (Shapiro-Wilk, t-test, Mann-Whitney U, Chi-square, Fisher's exact).
  * Automatically generates data-driven visualizations.

---

## 🛠️ Tech Stack

* **Core**: Python 3.10+
* **Data Manipulation**: Pandas, NumPy
* **Statistical Analysis**: SciPy (Stats)
* **Data Visualization**: Matplotlib, Seaborn
* **Storage**: SQLite3

---

## 🚀 Quick Start & Usage

### 1. Generate Synthetic Clinical Data
```python
from clinical_pipeline import TrialDataset

# Generate a dataset of 150 patients with corrupted data for testing
generator = TrialDataset(patient_number=150, d_value=2.5, add_waste=True, add_other_labs=True)
df_raw = generator.generate_main_data()

# Save the dataset (supports 'sql', 'csv', 'excel' via file_type argument)
generator.save_to_file(file_type='sql', db_name='clinical_trials')
```

### 2. Clean and Preprocess Data
```python
from clinical_pipeline import DataPreprocessor

preprocessor = DataPreprocessor()
# Run the entire cleaning pipeline (types, gender alignment, missing values)
df_clean = preprocessor.process(df_raw) 
```

### 3. Analyze and Visualize
```python
from clinical_pipeline import DataAnalyzer

analyzer = DataAnalyzer()
# Describe cohort split between Intervention and Control groups
report = analyzer.describe_cohort(df_clean, columns=['Age', 'Cognitive_Score', 'Gender'])
print(report)

# Generate plots
analyzer.get_plot(df_clean, column='Cognitive_Score')
```

---

## 📊 Pipeline Architecture & Class Overview

### `TrialDataset`
Responsible for data simulation.
* `generate_main_data()`: Generates random baseline clinical metrics.
* `add_waste()`: Corrupts data to simulate real-world data entry flaws.
* `add_other_labs()`: Merges data simulating multi-center studies.
* `save_to_file(file_type='sql', db_name='clinical_trials')`: Exports the generated dataset. Supports multiple formats (`sql`, `csv`, `excel`) based on the `file_type` argument.

### `DataPreprocessor`
Responsible for data health and standardization.
* `clean_gender()`: Standardizes variations into strict `male`/`female`.
* `handle_missing()`: Fills or drops missing values according to strictness settings.
* `add_education_level()`: Example of engineering categorical features.
* `process()`: Runs the entire processing pipeline.

### `DataAnalyzer`
The analytical engine.
* `_load_data()`: Polymorphic loader for DataFrames, Excel, CSV, or SQLite files.
* `describe_cohort()`: Aggregates descriptive statistics and computes $p$-values to evaluate differences between the **Intervention** and **Control** groups.
* `get_plot()`: Renders context-appropriate Seaborn plots based on data types.
