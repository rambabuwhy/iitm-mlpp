import pandas as pd

# Load the datasets
'''
Question 1 (MCQ): Dataset Dimensions 
Before building any models, it is crucial to understand the scale of the data. 
How many records (rows) are present in the train.csv and test.csv datasets, respectively?
'''

print("Loading datasets...")
train = pd.read_csv('train.csv', low_memory=False)
test = pd.read_csv('test.csv', low_memory=False)

# Display the number of records (rows) in each dataset
print("\n" + "="*50)
print("DATASET SCALE")
print("="*50)
print(f"train.csv: {len(train):,} rows")
print(f"test.csv: {len(test):,} rows")
print("="*50)

# Additional basic information
print("\n" + "="*50)
print("TRAIN DATASET INFO")
print("="*50)
print(f"Number of columns: {len(train.columns)}")
print(f"Shape: {train.shape}")
print("\nFirst few rows:")
print(train.head())

print("\n" + "="*50)
print("TEST DATASET INFO")
print("="*50)
print(f"Number of columns: {len(test.columns)}")
print(f"Shape: {test.shape}")
print("\nFirst few rows:")
print(test.head())


'''
Question 2 (MCQ): Target Variable Identification 
By doing a strict set difference between the columns in train.csv and test.csv, 
which column is correctly identified as the sole target variable we need to predict?
'''

# Find the column(s) in train.csv but not in test.csv
train_cols = set(train.columns)
test_cols = set(test.columns)
target_variable = train_cols - test_cols

print("\n" + "="*50)
print("TARGET VARIABLE IDENTIFICATION")
print("="*50)
print(f"Columns in train.csv: {len(train_cols)}")
print(f"Columns in test.csv: {len(test_cols)}")
print(f"\nColumns in train but NOT in test (target variable):")
print(target_variable)
print("="*50)

'''
Question 3 (Numeric): Feature Data Types
 In the train dataset, how many columns are strictly recognized as numerical features (int64 or float64) by default?
'''

# Count numerical features (int64 or float64)
numerical_cols = train.select_dtypes(include=['int64', 'float64']).columns
num_numerical_features = len(numerical_cols)

print("\n" + "="*50)
print("NUMERICAL FEATURES COUNT")
print("="*50)
print(f"Number of numerical columns (int64 or float64): {num_numerical_features}")
print(f"\nData types breakdown:")
print(train.dtypes.value_counts())
print(f"\nNumerical columns:")
print(list(numerical_cols))
print("="*50)


'''
Question 4 (MSQ): Extremely Sparse Columns 
Which of the following configuration columns exhibit a missing value rate higher than 90% (missing over 124,800 rows out of 138,701) in the training set and should likely be dropped or heavily grouped? (Select all that apply)
'''

# Check missing value rates for specific columns
columns_to_check = ['col18', 'col19', 'CabinType', 'col5']
total_rows = len(train)
threshold = 124800  # 90% of 138,701

print("\n" + "="*50)
print("EXTREMELY SPARSE COLUMNS (>90% missing)")
print("="*50)
print(f"Total rows in train: {total_rows:,}")
print(f"90% threshold: {threshold:,} missing rows\n")

sparse_columns = []
for col in columns_to_check:
    if col in train.columns:
        missing_count = train[col].isnull().sum()
        missing_rate = (missing_count / total_rows) * 100
        is_sparse = missing_count > threshold
        
        print(f"{col}:")
        print(f"  Missing values: {missing_count:,} ({missing_rate:.2f}%)")
        print(f"  Exceeds 90% threshold: {is_sparse}")
        
        if is_sparse:
            sparse_columns.append(col)
    else:
        print(f"{col}: Column not found in dataset")
    print()

print(f"Columns with >90% missing values: {sparse_columns}")
print("="*50)



'''
Question 5 (Numeric): Missing Operational Metrics 
The OperationalHoursMeter represents the lifetime active runtime of the machine. What is the approximate percentage of missing values for this critical column in the training dataset?
'''

# Calculate missing value percentage for OperationalHoursMeter
column_name = 'OperationalHoursMeter'
missing_count = train[column_name].isnull().sum()
total_rows = len(train)
missing_percentage = (missing_count / total_rows) * 100

print("\n" + "="*50)
print("OPERATIONAL HOURS METER - MISSING VALUES")
print("="*50)
print(f"Column: {column_name}")
print(f"Total rows: {total_rows:,}")
print(f"Missing values: {missing_count:,}")
print(f"Missing percentage: {missing_percentage:.2f}%")
print("="*50)

'''
Question 6 (MCQ): Target Value Distribution 
Heavy machinery prices are often right-skewed. What is the median of TargetValue (transaction price) in the dataset?
'''

# Calculate median and distribution statistics for TargetValue
median_price = train['TargetValue'].median()
mean_price = train['TargetValue'].mean()
min_price = train['TargetValue'].min()
max_price = train['TargetValue'].max()
std_price = train['TargetValue'].std()

print("\n" + "="*50)
print("TARGET VALUE DISTRIBUTION")
print("="*50)
print(f"Median (50th percentile): ${median_price:,.2f}")
print(f"Mean (average): ${mean_price:,.2f}")
print(f"Min: ${min_price:,.2f}")
print(f"Max: ${max_price:,.2f}")
print(f"Standard Deviation: ${std_price:,.2f}")
print(f"\nMean > Median: {mean_price > median_price} (indicates right-skewed distribution)")
print("="*50)


'''
Question 7 (Numeric): Manufacture Year Anomaly
 When analyzing the ManufactureYear, you will find a strange anomaly. What is the most frequently occurring (mode) ManufactureYear in the dataset, indicating a default placeholder value used by the data entry system?
'''

# Find the mode of ManufactureYear
manufacture_year_mode = train['ManufactureYear'].mode()[0]
manufacture_year_counts = train['ManufactureYear'].value_counts()

print("\n" + "="*50)
print("MANUFACTURE YEAR ANOMALY")
print("="*50)
print(f"Mode (most frequent value): {manufacture_year_mode}")
print(f"Frequency: {manufacture_year_counts.iloc[0]:,} occurrences")
print(f"\nTop 10 most common ManufactureYear values:")
print(manufacture_year_counts.head(10))
print(f"\nMin year: {train['ManufactureYear'].min()}")
print(f"Max year: {train['ManufactureYear'].max()}")
print("="*50)


'''
Question 8 (MCQ): Time Span of Transactions 
What is the earliest (minimum) TransactionDate recorded in the training dataset?
'''

# Find the earliest TransactionDate
# First check if TransactionDate exists in the dataset
if 'TransactionDate' in train.columns:
    # Convert to datetime if it's not already
    train['TransactionDate'] = pd.to_datetime(train['TransactionDate'])
    
    earliest_date = train['TransactionDate'].min()
    latest_date = train['TransactionDate'].max()
    
    print("\n" + "="*50)
    print("TRANSACTION DATE RANGE")
    print("="*50)
    print(f"Earliest (minimum) date: {earliest_date}")
    print(f"Latest (maximum) date: {latest_date}")
    print(f"Date range: {(latest_date - earliest_date).days} days")
    print("="*50)
else:
    print("\n" + "="*50)
    print("TransactionDate column not found in dataset")
    print("Available columns:", train.columns.tolist())
    print("="*50)


'''
Question 9 (Numeric): High Cardinality Identification
 Machine learning models struggle with high-cardinality categorical features. How many unique string classes exist within the Spec_BaseClass column?
'''

# Count unique values in Spec_BaseClass column
if 'Spec_BaseClass' in train.columns:
    unique_count = train['Spec_BaseClass'].nunique()
    total_count = train['Spec_BaseClass'].count()  # Non-null count
    missing_count = train['Spec_BaseClass'].isnull().sum()
    
    print("\n" + "="*50)
    print("SPEC_BASECLASS CARDINALITY")
    print("="*50)
    print(f"Number of unique classes: {unique_count}")
    print(f"Total non-null values: {total_count:,}")
    print(f"Missing values: {missing_count:,}")
    print(f"\nTop 10 most common classes:")
    print(train['Spec_BaseClass'].value_counts().head(10))
    print("="*50)
else:
    print("\n" + "="*50)
    print("Spec_BaseClass column not found in dataset")
    print("="*50)


    '''
    Question 10 (MCQ): Geographical Volume 
Which region accounts for the highest volume of machinery transactions (RegionCode) in this dataset?
    '''

# Find the region with the highest transaction volume
if 'RegionCode' in train.columns:
    region_counts = train['RegionCode'].value_counts()
    top_region = region_counts.index[0]
    top_region_count = region_counts.iloc[0]
    
    print("\n" + "="*50)
    print("GEOGRAPHICAL VOLUME - REGION CODE")
    print("="*50)
    print(f"Region with highest volume: {top_region}")
    print(f"Number of transactions: {top_region_count:,}")
    print(f"Percentage of total: {(top_region_count / len(train) * 100):.2f}%")
    print(f"\nTop 10 regions by transaction volume:")
    print(region_counts.head(10))
    print(f"\nTotal unique regions: {train['RegionCode'].nunique()}")
    print("="*50)
else:
    print("\n" + "="*50)
    print("RegionCode column not found in dataset")
    print("="*50)


    '''
    Question 11 (Numeric): Regional Pricing Insights 
For the most frequent region that is identified as Florida, what is the approximate average TargetValue for machines sold there?
    '''

# Calculate average TargetValue for Florida
if 'RegionCode' in train.columns and 'TargetValue' in train.columns:
    florida_data = train[train['RegionCode'] == 'Florida']
    florida_avg_price = florida_data['TargetValue'].mean()
    florida_median_price = florida_data['TargetValue'].median()
    florida_count = len(florida_data)
    
    # Overall average for comparison
    overall_avg_price = train['TargetValue'].mean()
    
    print("\n" + "="*50)
    print("REGIONAL PRICING - FLORIDA")
    print("="*50)
    print(f"Number of Florida transactions: {florida_count:,}")
    print(f"Average TargetValue in Florida: ${florida_avg_price:,.2f}")
    print(f"Median TargetValue in Florida: ${florida_median_price:,.2f}")
    print(f"\nOverall average (all regions): ${overall_avg_price:,.2f}")
    print(f"Difference from overall: ${florida_avg_price - overall_avg_price:,.2f}")
    print("="*50)
else:
    print("\n" + "="*50)
    print("Required columns not found for Florida pricing analysis")
    print("="*50)

    '''
    Question 12 (MCQ): Asset Utilization 
The UtilizationTier metadata column ranks how heavily an asset was used. Based on the value counts in the training set, what is the most common utilization tier?
    '''

# Find the most common UtilizationTier
if 'UtilizationTier' in train.columns:
    utilization_counts = train['UtilizationTier'].value_counts()
    most_common_tier = utilization_counts.index[0]
    most_common_count = utilization_counts.iloc[0]
    
    print("\n" + "="*50)
    print("ASSET UTILIZATION TIER")
    print("="*50)
    print(f"Most common utilization tier: {most_common_tier}")
    print(f"Count: {most_common_count:,}")
    print(f"Percentage: {(most_common_count / len(train) * 100):.2f}%")
    print(f"\nAll UtilizationTier value counts:")
    print(utilization_counts)
    print(f"\nTotal unique tiers: {train['UtilizationTier'].nunique()}")
    print(f"Missing values: {train['UtilizationTier'].isnull().sum():,}")
    print("="*50)
else:
    print("\n" + "="*50)
    print("UtilizationTier column not found in dataset")
    print("="*50)


    '''
    Question 13 (MCQ): Operational Correlation

 Intuitively, one might assume that machines with higher OperationalHoursMeter values would sell for less. What is the Pearson correlation coefficient between TargetValue and OperationalHoursMeter?
    '''

# Calculate Pearson correlation between TargetValue and OperationalHoursMeter
if 'TargetValue' in train.columns and 'OperationalHoursMeter' in train.columns:
    # Calculate correlation (this automatically handles missing values by excluding them)
    correlation = train['TargetValue'].corr(train['OperationalHoursMeter'])
    
    # Also get some additional statistics
    valid_pairs = train[['TargetValue', 'OperationalHoursMeter']].dropna()
    n_valid = len(valid_pairs)
    
    print("\n" + "="*50)
    print("OPERATIONAL CORRELATION ANALYSIS")
    print("="*50)
    print(f"Pearson correlation coefficient: {correlation:.4f}")
    print(f"Number of valid pairs (non-null): {n_valid:,}")
    print(f"\nInterpretation:")
    if correlation > 0:
        print(f"  Positive correlation: As OperationalHoursMeter increases,")
        print(f"  TargetValue tends to increase (contrary to intuition)")
    elif correlation < 0:
        print(f"  Negative correlation: As OperationalHoursMeter increases,")
        print(f"  TargetValue tends to decrease (matches intuition)")
    else:
        print(f"  No linear correlation detected")
    print("="*50)
else:
    print("\n" + "="*50)
    print("Required columns not found for correlation analysis")
    print("="*50)


    '''
    Question 14 (MCQ): Primary Equipment Types 
Based on the FunctionalClassification column, what is the description of the single most common exact machinery traded on this platform?
    '''

# Find the most common FunctionalClassification
if 'FunctionalClassification' in train.columns:
    functional_counts = train['FunctionalClassification'].value_counts()
    most_common_type = functional_counts.index[0]
    most_common_count = functional_counts.iloc[0]
    
    print("\n" + "="*50)
    print("PRIMARY EQUIPMENT TYPES")
    print("="*50)
    print(f"Most common machinery type: {most_common_type}")
    print(f"Count: {most_common_count:,}")
    print(f"Percentage: {(most_common_count / len(train) * 100):.2f}%")
    print(f"\nTop 10 FunctionalClassification values:")
    print(functional_counts.head(10))
    print(f"\nTotal unique classifications: {train['FunctionalClassification'].nunique()}")
    print(f"Missing values: {train['FunctionalClassification'].isnull().sum():,}")
    print("="*50)
else:
    print("\n" + "="*50)
    print("FunctionalClassification column not found in dataset")
    print("="*50)