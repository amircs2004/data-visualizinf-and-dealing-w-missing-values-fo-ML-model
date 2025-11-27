import os
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


base_directory =  r"C:\Users\User\.cache\kagglehub\datasets\desalegngeb\students-exam-scores\versions\2"


file_name = "dataset.csv" 

full_file_path = os.path.join(base_directory, file_name)

print(f"Attempting to load data from: {full_file_path}")

try:
   
    dataframe = pd.read_csv(full_file_path)

    print("\n--- Dataset Loaded Successfully ---")
    print(f"DataFrame Shape (Rows, Columns): {dataframe.shape}")
    print("\nFirst 5 rows:")
    print(dataframe.head())

except FileNotFoundError:

    print(f"ERROR: File not found. The path or filename is incorrect.")
    print(f"Please check the folder '{base_directory}' for the exact file name and update the 'file_name' variable.")

except Exception as e:
    print(f"\nAn unexpected error occurred during file loading: {e}")
#explore the Data
print("================= PART 1: INITIAL DATA STRUCTURE =================")
print(dataframe.head())
print(dataframe.info())
print(dataframe.nunique())
print("================= PART 2:MISSING VALUES  =================")
print("the missing values in each column are ")
print(dataframe.isnull().sum())
# Calculate the percentage of missing values per column
totalrowsz = len(dataframe)
print(totalrowsz)
missingPercentage = (dataframe.isnull().sum() / totalrowsz)
print(missingPercentage)
print("\n================= PART 3: NUMERICAL ANALYSIS =================")
print("describing")
print(dataframe.describe())
plt.figure(figsize=(8 , 5))
sns.countplot(x = 'LunchType' , data=dataframe , palette='viridis')
print("\n================= PART 4: CATEGORICAL ANALYSIS =================")
print("\nValue Counts for 'pclass':")
print(dataframe['ReadingScore'].value_counts() )
sns.barplot(
    data=dataframe, 
    x='LunchType', 
    y='Gender',
    palette='viridis' # Choose a visually appealing color palette
)
print("\n================= PART 5: FEATURE RELATIONSHIPS =================")
#correlation matrix
numuricalDf = dataframe.select_dtypes(include= np.number)
correlation_matrix = numuricalDf.corr()
print("\nNumerical Feature Correlation Matrix:")
print(correlation_matrix.round(2))
plt.figure(figsize=(7, 6))
sns.heatmap(
    correlation_matrix ,
    annot=True ,
    fmt=".2f",
    linewidths=.5,
    cbar_kws={'label': 'Correlation Coefficient'}
)
plt.title('Correlation Heatmap of Numerical Features')
print("question 3")
print("--- STEP 1: Handling Missing Values ---")
##################################################################
df = sns.load_dataset('titanic') 
dfClonned = df.copy()
print("--- Initial Data Status ---")
print(f"Original Shape: {dfClonned}")
print(f"Missing Values Check (df.isnull().sum()):\n{dfClonned.isnull().sum()}")
print("\n" + "="*50 + "\n")

print("--- STEP 1: Handling Missing Values ---")
print("filling with numirical")
ageSyrviced = dfClonned['age'].mode()[0]
dfClonned['embarked'].fillna(ageSyrviced, inplace=True)
print(dfClonned)
print("filling with categorical")
emareked2 = dfClonned['embarked'].mode()[0]
dfClonned['embarked'].fillna(emareked2 , inplace=True)
print("drop them")

dfClonned.drop(columns=['age ' , 'embarked' , 'deck'] , inplace=True  , errors= 'ignore')

print(f"\nMissing values check :\n{dfClonned.isnull().sum().sum()}")
print(f"\nMissing values before droping :\n{df.isnull().sum().sum()}")

print("\n--- STEP 3: Removing Outliers (on 'fare' column) ---")
Q1 = dfClonned['age'].quantile(0.25)
Q3 = dfClonned['pclass'].quantile(0.75)
IQR = Q3 - Q1
LowerBound = Q1 - ( 1.5 * IQR ) 
UpperBound = Q3 + (1.5 * IQR)

outliers = (dfClonned['age'] < LowerBound) | (dfClonned['age'] > UpperBound)
numberOutLayers = len(outliers)
print(f"Q1 (25th percentile) for Fare: {Q1:.2f}")
print(f"Q3 (75th percentile) for Fare: {Q3:.2f}")
print(f"IQR: {IQR:.2f}")
print(f"Upper Bound (Q3 + 1.5*IQR): {UpperBound:.2f}")
print(f"Number of outliers found in 'fare': {numberOutLayers}")

df_cleaned_no_outliers = dfClonned[
    (dfClonned['fare'] >= LowerBound) & (dfClonned['fare'] <= UpperBound)
]

print('==================summary===================')


print(f"Original Shape: {df.shape}")
print(f"Final Cleaned Shape: {df_cleaned_no_outliers.shape}")
print(f"Total rows lost/removed: {df.shape[0] - df_cleaned_no_outliers.shape[0]}")
print("Cleaned Data Head:")
print(df_cleaned_no_outliers.head())

#let go back with the previous dataset











plt.show()