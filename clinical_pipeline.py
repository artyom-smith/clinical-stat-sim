from scipy.stats import shapiro, ttest_ind, mannwhitneyu, chi2_contingency, fisher_exact, pearsonr, spearmanr
from typing import List, Dict, Union, Optional, Tuple, Any
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import sqlite3
import warnings
import inspect
import random
import re


class TrialDataset:
  '''
    Class TrialDataset makes a sample dataset of hypothetical patients with dementia subdivided into control and intervention groups to test statistical analyzers in future.

    Methods:
      generate_main_data: Makes fully random dataset of clinical data.
      add_waste: Inserts wasted data into pd.DataFrame.
      add_other_labs: Expands the table by adding random data from supposedly other laboratoriesю
      save_to_file: Converts pandas DataFrame into SQL database and saves into root folder.

    Arguments:
      patient_number: Size of dataset. Default = 100
      d_value: Effect size ($d$-value) to control the statistical difference. Default = 2.5
      add_waste: An indicator of whether the data should be corrupted or not. Default = True
      add_other_labs: An indicator of whether the dataыуе should be expanded with other labs' data or not. Default = True
  '''

  def __init__ (self,
                patient_number: int = 100,
                d_value: float = 2.5,
                add_waste: bool = True,
                add_other_labs: bool = True) -> None:
    self.df = None
    self.generate_main_data(patient_number, d_value)
    if add_waste: self.add_waste()
    if add_other_labs: self.add_other_labs()

  def generate_main_data (self,
                          patient_number: int,
                          d_value: float) -> None:
    '''
      Generates dataset.

      Arguments:
        patient_number: number of records in the dataset. Default = 100
        d_value: effect-size between intervention and control groups. Default = 2.5
    '''

    if patient_number%2 != 0: patient_number -= 1 # Makes even number of patients
    noise = np.random.normal(0, 1.5, size=patient_number)

    patient_id = np.random.randint(10000, 99999, size=patient_number) # Creates numpy array of the random ids and then converts into list

    age = np.random.normal(72, 6, size=patient_number).clip(60, 85) # Creates numpy array of the random numbers of years of age, limits values ​​within a range 60-85, converts it into list and round all ages to integers
    age = np.round(age).astype(int)

    gender = np.random.choice(['Male', 'Female'], size=patient_number) # Creates numpy array of randomly chosen gender and converts into list

    education = np.random.normal(14, 3, size=patient_number).clip(8, 20) # Creates numpy array of the random numbers of years of education, limits values ​​within a range 8-20, converts it into list and round all years to integers
    education = np.round(education).astype(int)

    comorbidities_count = np.random.poisson(lam = age/20).clip(0, 5).astype(int) # Creates Poisson-distributed numpy array of integers from 0 to 5
    is_intervention = np.concatenate((np.full(shape = (1, patient_number // 2), fill_value=0)[0], np.full(shape = (1, patient_number // 2), fill_value=1)[0])) # Creates a fully symmetrical numpy array of the 0 and 1

    moca_score = (26 + 0.3*education - 0.5*comorbidities_count - 0.1*age + d_value*is_intervention + noise).clip(20, 30) # Creates a random numpy array of the MoCA-test results
    moca_score = np.round(moca_score).astype(int)

    reaction_time = (200 + age*0.68 + comorbidities_count - 0.8*moca_score - noise).clip(0, 800) # Creates a random numpy array of the attention dot-test results

    amiloid_level = (0.12 - 0.001*moca_score).clip(0, 10) # Creates a random numpy array of the amiloid Aβ42/Aβ40 ration in peripheral blood

    core_lab = np.full(shape = (1, patient_number), fill_value = 'LabCorp')[0]
    site = np.full(shape = (1, patient_number), fill_value = 'Siberian Neurology Institute')[0]
    site_quality = np.full(shape = (1, patient_number), fill_value = 'Medium')[0]
    employment = np.random.choice(['Retired', 'Employed', 'Unemployed', 'Disabled', 'Homemaker'], size=patient_number)
    living = np.random.choice(['With family', 'Alone', 'Assisted facility', 'With spouse', 'Nursing home'], size=patient_number)

    self.df = pd.DataFrame({
      'Patient id': patient_id,
      'Age': age,
      'Gender': gender,
      'Education': education,
      'Comorbidities count': comorbidities_count,
      'Is intervention': is_intervention,
      'Moca score': moca_score,
      'Reaction time': reaction_time,
      'Amiloid Aβ42/Aβ40 ratio': amiloid_level,
      'Core lab': core_lab,
      'Clinical site': site,
      'Site quality': site_quality,
      'Employment status': employment,
      'Living status': living
    })

    self.df = self.df.sample(frac=1).reset_index(drop=True) # Shuffles all records in DataFrame

  def add_waste(self) -> None:
    '''
      Adds wasted data into dataset.
    '''

    waste = ['error', 'hdkn', '-', 'no data', 'no', 'denied', '', ' ', np.nan]
    gender_waste = ['Male', 'Male.', ' Male ', ' Male', 'Male ', 'male', 'male.', ' male ', ' male', 'male ', 'm', 'm.', ' m ', ' m', 'm ', 'Female', 'Female.', ' Female ', ' Female', 'Female ', 'female', 'female.', ' female ', ' female', 'female ', 'f', 'f.', ' f ', ' f', 'f ']
    count = random.randint(0, round(len(self.df)*0.3))
    self.df = self.df.astype(object)

    for i in range(count): # Randomly add waste such as '-' or NaNs
      row = random.randint(0, self.df.shape[0] - 1)
      col = random.randint(0, self.df.shape[1] - 1)

      self.df.iat[row, col] = random.choice(waste)

    for i in range(count): # Randomly waste Gender column values
      row = random.randint(0, self.df.shape[0] - 1)
      col = self.df.columns.get_loc('Gender')

      self.df.iat[row, col] = random.choice(gender_waste)

    count = random.randint(0, round(len(self.df)*0.1))

    for i in range(count): # Add some dublicates
      row = random.randint(0, self.df.shape[0] - 1)
      col = random.randint(0, self.df.shape[1] - 1)

      self.df.iloc[row, :] = self.df.iloc[row + 1, :].values

  def add_other_labs(self) -> None:
    '''
      Expands the table by adding random data from supposedly other laboratories.
    '''

    res = self.df.copy()

    patient_id = np.random.randint(10000, 99999, size=len(self.df)) # Creates numpy array of the random IDs and then converts into list
    res['Patient id'] = patient_id

    core_labs = ['Quest Diagnostics', 'Mayo Clinic Labs', 'BioReference']
    res['Core lab'] = np.random.choice(core_labs, size=len(self.df)) # Creates numpy array of the random Labs, where a study was executed

    sites = ['Moscow City Hospital #1', 'Regional Clinic SPb', 'Volga Medical Center', 'Siberian Neurology Institute', 'Far East Medical Center', 'South Region Clinic']
    res['Clinical site'] = np.random.choice(list(sites), size=len(self.df)) # Creates numpy array of the clinical sites

    site_quality = ['High', 'Medium', 'Low']
    res['Site quality'] = np.random.choice(list(site_quality), size=len(self.df)) # Creates numpy array of the clinical sites quality

    employment = ['Retired', 'Employed', 'Unemployed', 'Disabled', 'Homemaker']
    res['Employment status'] = np.random.choice(employment, size=len(self.df), p=[0.65, 0.15, 0.1, 0.05, 0.05]) # Creates numpy array of the patient's employment status

    living = ['With family', 'Alone', 'Assisted facility', 'With spouse', 'Nursing home']
    res['Living status'] = np.random.choice(living, size=len(self.df), p=[0.35, 0.3, 0.15, 0.15, 0.05]) # Creates numpy array of the patient's living status

    self.df = pd.concat([self.df, res], ignore_index=True)

  def save_to_file(self,
                   file_type: str = 'sql',
                   db_name: str = 'clinical_trials') -> None:
    '''
    Converts pandas DataFrame into SQL database and saves into root folder.

    Arguments:
      file_type: Indicator of the format in which the data should be saved. Default = 'sql'
      db_name: the name of the database under which it will be saved. Default = clinical_trials.db
    '''

    if 'sql' in file_type:
      db_name += '.db'
      conn = sqlite3.connect(db_name)
      cursor = conn.cursor()
      self.df.to_sql('trials', conn, if_exists='replace', index=False)

      cursor.execute('CREATE INDEX idx_patient_id ON trials (`Patient id`)')
      conn.commit()
      conn.close()
      print(f'✅ Data saved to {db_name} and indexed by Patient id.')
      return

    elif 'xls' in file_type:
      db_name += '.xlsx'
      self.df.to_excel(db_name)
      print(f'✅ Data saved to {db_name} and indexed by Patient id.')
      return

    elif 'csv' in file_type:
      db_name += '.csv'
      self.df.to_csv(db_name)
      print(f'✅ Data saved to {db_name} and indexed by Patient id.')
      return

    else:
      raise ValueError(f"TrialDataset.save_to_file() doesn't support this file format -> {file_type}")


class DataPreprocessor:
  '''
    Responsible only for cleaning and preparing data for analysis

    Methods:
      clean_types: Converts columns to the correct types.
      clean_gender: Sets all values ​​in the "Gender" column to the standart "male" and "female".
      handle_missing: Fills in the blanks depending on the strict mode.
      add_education_level: Creates a categorical column for education level.
      sort_and_reset: Final sorting and index reset.
      process: Runs the entire processing pipeline.
  '''

  def __init__(self,
               df: pd.DataFrame,
               is_strict: bool = False) -> None:
    self.df = df.copy()
    self.is_strict = is_strict

  def clean_types(self) -> 'DataPreprocessor':
    '''
      Converts columns to the correct types
    '''

    for col in ['Patient id', 'Age', 'Education', 'Comorbidities count', 'Is intervention', 'Moca score', 'Reaction time', 'Amiloid Aβ42/Aβ40 ratio']:
      if col in ['Reaction time', 'Amiloid Aβ42/Aβ40 ratio']:
        self.df[col] = pd.to_numeric(self.df[col], errors='coerce', downcast='float')

      else:
        self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
        if col == 'Patient id': self.df[col] = self.df[col].astype('Int32')
        else: self.df[col] = self.df[col].astype('Int8')

    return self

  def clean_gender(self) -> 'DataPreprocessor':
    '''
      Sets all values ​​in the "Gender" column to the standart "male" and "female".
    '''

    self.df['Gender'] = self.df['Gender'].str.extract(r'(female|male)', flags=re.IGNORECASE, expand=False)
    self.df['Gender'] = self.df['Gender'].str.lower().astype('category')

    return self

  def handle_missing(self) -> 'DataPreprocessor':
    '''
      Fills in the blanks depending on the strict mode
    '''

    if self.is_strict: # Remove all NaNs
      self.df = self.df.dropna()

    else: # Remove only critical gaps
      self.df = self.df.dropna(axis=0, subset=['Is intervention'])

      # Fill the numerical columns with averages
      self.df['Age'] = self.df['Age'].fillna(round(self.df['Age'].mean()))
      self.df['Education'] = self.df['Education'].fillna(round(self.df['Education'].mean()))

      # The rest - using the ffill/bfill method
      for col in ['Comorbidities count', 'Moca score', 'Reaction time', 'Amiloid Aβ42/Aβ40 ratio']:
        self.df[col] = self.df[col].ffill().bfill()

    return self

  def add_education_level(self) -> 'DataPreprocessor':
    '''
      Creates a categorical column for education level
    '''

    education_level = []
    for i in self.df['Education']: # Standardizes the values ​​in a column "Education"
      if i < 12: education_level.append('school')
      elif i < 15: education_level.append('college')
      else: education_level.append('university')

    self.df.insert(4, 'Education level', education_level)
    self.df['Education level'] = self.df['Education level'].astype('category')

    return self

  def sort_and_reset(self) -> 'DataPreprocessor':
    '''
      Final sorting and index reset
    '''

    self.df = self.df.sort_values('Is intervention').reset_index(drop=True)
    return self

  def process(self) -> pd.DataFrame:
    '''
      Runs the entire processing pipeline
    '''

    return (self.clean_types()
                .clean_gender()
                .handle_missing()
                .add_education_level()
                .sort_and_reset()
                .df)


class DataAnalyzer:
  '''
    Class DataAnalyzer gets data in different formats (.pdDataFrame, .csv, .xlsx, .xls, .db) and describes it using descriptive statistics methods

    Methods:
      _load_data: Checks whether data format is suitable for working and then loads it into pd.Dataframe.
      _get_group_stats: Counts main descriptive statistical metrics for entered columns.
      _get_p_value: Counts statistical significance of differences between samples.
      describe_cohort: Returns main descriptive stats and p values for entered columns between Intervention and Control groups.
      get_plot: Generates suitable vizualisation for entred columns.
  '''

  def __init__(self,
               source: Union[str, pd.DataFrame],
               strict_load: bool = True,
               strict_for_nan: bool = False) -> None:
    '''
      Arguments:
        source: Pandas DataFrame or file path in string format
        strict_load: The severity level of loading without some columns - either raise KeyError (strict_load=True) or load an incomplete table (strict_load=False). Default = True
        strict_for_nan: The severity level of cleaning from NaNs - either complete cleaning (strict_for_nan=True) or replacement with medians (strict_for_nan=False). Default = False
    '''

    self.df = None
    query = '''
      SELECT `Patient id`, Age, Gender, Education, `Comorbidities count`, `Moca score`, `Reaction time`, `Amiloid Aβ42/Aβ40 ratio`
      FROM trials
      WHERE `Core lab` = 'LabCorp' AND Site = 'Siberian Neurology Institute' AND `Patient id` IS NOT NULL
      ORDER BY `Is intervention`
    '''
    required_cols=['Age', 'Gender', 'Education', 'Comorbidities count', 'Is intervention', 'Moca score', 'Reaction time', 'Amiloid Aβ42/Aβ40 ratio']

    self._load_data(source, query, required_cols, strict_load)
    self._validate_columns()

    preprocessor = DataPreprocessor(self.df, strict_for_nan)
    self.df = preprocessor.process()


  def _load_data(self,
                 source: Union[str, pd.DataFrame],
                 query: str,
                 required_cols: List[str],
                 strict_load: bool) -> None:
    '''
      Checks whether data format is suitable for working and then loads it into pd.Dataframe.

      Arguments:
        source: Pandas DataFrame or file path in string format
        query: Query for .db file in order to load only needed data
        required_cols: List of columns that are required to load
        strict_load: The severity level of loading without some columns - either raise KeyError (strict_load=True) or load an incomplete table (strict_load=False). Default = True
    '''

    if isinstance(source, pd.DataFrame): # Copy Pandas DataFrame
      self.df = source[required_cols].copy()
      return

    elif source.endswith('.db'): # Load .sql file
      with sqlite3.connect(source) as conn:
        self.df = pd.read_sql_query(query, conn)
      return

    elif source.endswith('.csv'): # Load .csv file
      existing_cols = pd.read_csv(source, nrows=0).columns.tolist()
      missing = [col for col in required_cols if col not in existing_cols] # Checks for necessary columns

      if missing:
        if strict_load:
          raise KeyError(f'Missing columns in data: {missing}. Unable to load.') # If not all necessary columns exist - raise Error (whether client needs strict data load)
        else:
          print(f"⚠️ WARNING: Missing columns: {missing}") # If not all necessary columns exist - raise Warning and continue working (whether client doesn't need strict data load)
          valid_cols = [col for col in required_cols if col in existing_cols]
          self.df = pd.read_csv(source, usecols=valid_cols)
          return
      else:
        self.df = pd.read_csv(source, usecols=required_cols)
        return

    elif source.endswith(('.xlsx', '.xls')): # Load Excel file
      existing_cols = pd.read_excel(source, nrows=0).columns.tolist()
      missing = [col for col in required_cols if col not in existing_cols] # Checks for necessary columns

      if missing:
        if strict_load:
          raise KeyError(f'Missing columns in data: {missing}. Unable to load.') # If not all necessary columns exist - raise Error (whether client needs strict data load)
        else:
          print(f"⚠️ WARNING: Missing columns: {missing}") # If not all necessary columns exist - raise Warning and continue working (whether client doesn't need strict data load)
          valid_cols = [col for col in required_cols if col in existing_cols]
          self.df = pd.read_excel(source, usecols=required_cols)
          return
      else:
        self.df = pd.read_excel(source, usecols=required_cols)
        return

    else:
      raise ValueError("Unsupported format")

  def _get_group_stats(self,
                       subset: pd.DataFrame,
                       col: str,
                       is_categorical: bool = False) -> Dict[str, List[Union[float, int, List[float]]]]:
    '''
      Counts main descriptive statistical metrics for entered columns.

      Arguments:
        subset: Slice from the main Pandas DataFrame
        col: The names of the columns needed to be analyzed
        is_categorical: Сategorical column indicator (nominative data). Default = False

      Return:
        dictionary with keys = column name, and value = list of mean and either SD or quartiles Q1-Q3
    '''

    res = {}

    if subset[col].dtype.name == 'category' or subset[col].nunique() < 5 or is_categorical:
        counts = dict(subset[col].value_counts())
        for i in counts:
          res[f'{col}_{i}'] = [int(counts[i]), float(counts[i] * 100 / sum(counts.values()))]
        return res

    data = subset[col].dropna()
    stat, p = shapiro(data)

    if p > 0.05:
        res = {col: [float(data.mean()), float(data.std())]}
        return res
    else:
        q1, q3 = data.quantile([0.25, 0.75])
        res = {col: [float(data.median()), [float(q1), float(q3)]]}
        return res

  def _get_p_value(self,
                   group1: pd.DataFrame,
                   group2: pd.DataFrame,
                   col: str,
                   is_categorical: bool = False) -> float:
    '''
      Counts statistical significance of differences between samples.

      Arguments:
       group1: First object of comparing
       group2: Second object of comparing
       col: Name of col needed to compare in two groups
       is_categorical: Сategorical column indicator (nominative data). Default = False

      Return:
        p-value as float
    '''

    g1 = group1[col].dropna()
    g2 = group2[col].dropna()

    if is_categorical:
      # Make a contingency table for 'Gender' or 'Education' categories
      contingency_table = pd.crosstab(self.df['Is intervention'], self.df[col])
      if contingency_table.shape == (2, 2) and contingency_table.values.min() < 5:
        _, p = fisher_exact(contingency_table)
      else:
        _, p, _, _ = chi2_contingency(contingency_table)
      return p

    # For numerical data, check the normality of distribution
    _, p_norm1 = shapiro(g1)
    _, p_norm2 = shapiro(g2)

    if p_norm1 > 0.05 and p_norm2 > 0.05:
      _, p = ttest_ind(g1, g2)
    else:
      _, p = mannwhitneyu(g1, g2)
    return p

  def describe_cohort(self,
                      columns: Union[str, List[str]]) -> pd.DataFrame:
    '''
      Returns main descriptive stats and p values for entered columns between Intervention and Control groups.

      Arguments:
        col: List of columns needed to describe

      Return:
        pandas DataFrame table of all parameters
    '''

    control = self.df[self.df['Is intervention'] == 0]
    interv = self.df[self.df['Is intervention'] == 1]

    res = {
        'Metric': ['Total Patients'],
        'Control': [len(control)],
        'Deviation': [0],
        'Intervention': [len(interv)],
        'Deviation': [0],
        'P-value': [0]
    }
    res = pd.DataFrame(res)

    if isinstance(columns, (list, tuple)): pass
    else: columns = [columns]

    for col in columns:
      # Calculate stats for each group
      val_ctrl = self._get_group_stats(control, col)
      val_intv = self._get_group_stats(interv, col)

      # Determine whether the column for the test is categorical
      is_cat = self.df[col].dtype.name == 'category' or self.df[col].nunique() < 5

      # Calculate p-value
      p_val = self._get_p_value(control, interv, col, is_categorical=is_cat)

      # Generating a table row
      for metric in list(val_ctrl.keys()):
        row = {
          'Metric': metric,
          'Control': val_ctrl[metric][0],
          'Deviation': str(val_ctrl[metric][1]),
          'Intervention': val_intv[metric][0],
          'Deviation': str(val_intv[metric][1]),
          'P-value': float(p_val)
        }
        res.loc[len(res)] = row # Pushing new row into Pandas DataFrame

    return res

  def get_plot(self,
               col: Union[str, List[str]],
               plot_type: Optional[str] = None,
               **kwargs) -> None:
    '''
      Generates suitable vizualisation for entred columns.

      Arguments:
        col: Name of cols needed to be visualized
        plot_type: Type of plot. Default = None
        **kwargs: Additional kwargs as plot settings for seaborn library
    '''

    if isinstance(col, (list, tuple)) and len(col) > 2:
      raise ValueError('The get_plot method supports a maximum of 2 columns (group comparison or x/y relationship).')

    plt.figure(figsize=(10, 6))

    # Case 1: Get two columns - so vizualizing relationship between two groups (Scatterplot)
    if isinstance(col, (list, tuple)) and len(col) == 2:
      c1, c2 = col
      # Generating scatterplot
      try: ax = sns.scatterplot(data=self.df, x=c1, y=c2, hue='Is intervention', **kwargs)
      except TypeError as e: raise TypeError(f'Error in design parameters: {e}. Check the spelling of **kwargs.')


      # Count correlation level (Srearmen, as it is robust to outliers)
      corr, p_val = spearmanr(self.df[c1].dropna(), self.df[c2].dropna())

      title = f'Relationship {c1}, and {c2}\n'
      stat_text = f'Correlation (Spearman): {corr:.3f}, p-value: {p_val:.4f}'

    # Case 2: Get one column - so compare 'Intervention', and 'Control' groups
    else:
      if isinstance(col, (list, tuple)): col = col[0] # If passed a list of one element

      is_cat = str(self.df[col].dtype) in ['category', 'object'] or self.df[col].nunique() < 5

      g1 = self.df[self.df['Is intervention'] == 0]
      g2 = self.df[self.df['Is intervention'] == 1]
      p_val = self._get_p_value(g1, g2, col, is_categorical=is_cat)

      if is_cat:
        # Plot for category
        try: ax = sns.countplot(data=self.df, x=col, hue='Is intervention', **kwargs)
        except TypeError as e: raise TypeError(f'Error in design parameters: {e}. Check the spelling of **kwargs.')

        title = f'Deviation {col} (Categorical)\n'
      else:
        # Numerical plot (Boxplot by default as it is better for p-value)
        current_plot = plot_type if plot_type else 'box'
        if current_plot == 'box':
          try: ax = sns.boxplot(data=self.df, x='Is intervention', y=col, **kwargs)
          except TypeError as e: raise TypeError(f'Error in design parameters: {e}. Check the spelling of **kwargs.')
        else:
          try: ax = sns.histplot(data=self.df, x=col, hue='Is intervention', kde=True, **kwargs)
          except TypeError as e: raise TypeError(f'Error in design parameters: {e}. Check the spelling of **kwargs.')
      title = f'Compare {col} by groups\n'

      stat_text = f"p-value: {p_val:.4f} ({'Significant' if p_val < 0.05 else 'Not significant'})"

    # Design
    color = 'red' if p_val < 0.05 else 'black'
    plt.title(title + stat_text, fontsize=12, color=color, fontweight='bold' if p_val < 0.05 else 'normal')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    print("--- Running Clinical Pipeline Demo ---")
    
    # 1. Generate data
    generator = TrialDataset(patient_number=50)
    df_raw = generator.generate_main_data()
    print("1. Synthetic data generated successfully.")
    
    # 2. Clean data
    preprocessor = DataPreprocessor()
    df_clean = preprocessor.process(df_raw)
    print("2. Data cleaned and preprocessed.")
    
    # 3. Analize data
    analyzer = DataAnalyzer()
    print("\n3. Statistical Cohort Report:")

    # 4. Generate a report
    report = analyzer.describe_cohort(df_clean, columns=['Age', 'Gender'])
    print(report)
