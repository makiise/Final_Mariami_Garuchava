import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# cleaning data and handling missing values

def clean_process_data(df: pd.DataFrame):
    df_clean = df.copy()


    # 1. in my current working data, there are no missing values but handling implementation is neccessary
    # median is good for skewed data and outliers found in numerical variables 
    skewed_cols = ['age', 'ejection_fraction', 'serum_creatinine', 
                   'platelets', 'creatinine_phosphokinase', 'serum_sodium']

    # I will drop the 'time' variable during training process in case to be safe from data leakage, so I'm not
    #filling missing values for 'time' right now.
    
    # by using median,we Robust against outliers found in exploratory data analysis
    median = SimpleImputer(strategy='median')
    df_clean[skewed_cols] = median.fit_transform(df_clean[skewed_cols])

    # 2.for categorical/binary variables, missing values must be filled with mode
    binary_cols = ['high_blood_pressure', 'smoking', 'diabetes', 'anaemia', 'sex']
    mode = SimpleImputer(strategy='most_frequent')
    df_clean[binary_cols] = mode.fit_transform(df_clean[binary_cols])

    # Z-score normalization
    # we need this to treat all units equally 
    scaler = StandardScaler()
    
    # We exclude the target 'DEATH_EVENT' from scaling
    X = df_clean.drop('DEATH_EVENT', axis=1)
    y = df_clean['DEATH_EVENT']
    
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    
    # re-attach the excluded target
    df_processed = pd.concat([X_scaled, y.reset_index(drop=True)], axis=1)
    
    return df_processed

