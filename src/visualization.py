import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_continuous_distribution(df: pd.DataFrame, column: str, title: str = None):
    """
    plot KDE for any continuous variable.
    """
    plt.figure(figsize=(10, 5))
    sns.kdeplot(data=df[df['DEATH_EVENT'] == 0], x=column, fill=True, label='Survived', color='blue')
    sns.kdeplot(data=df[df['DEATH_EVENT'] == 1], x=column, fill=True, label='Died', color='red')
    plt.title(title if title else f'Distribution of {column} by Survival')
    plt.legend()
    plt.show()

def plot_categorical_ratio(df: pd.DataFrame, column: str):
    """
    binary/categorical variables.
    """

    plt.figure(figsize=(8, 5))
    # the percentage of deaths within each category
    sns.barplot(data=df, x=column, y='DEATH_EVENT', palette='viridis', ci=None)
    plt.title(f'Mortality Rate by {column}')
    plt.ylabel('Mortality Rate (Probability)')
    plt.show()

def plot_outlier_analysis(df: pd.DataFrame, column: str):
    """
    boxplot to show mathematical quartiles and outliers
    """
    plt.figure(figsize=(10, 5))
    sns.boxplot(data=df, x='DEATH_EVENT', y=column, palette='Set2')
    plt.title(f'Outlier Analysis for {column}')
    plt.show()

def plot_correlation_heatmap(df: pd.DataFrame):
    """
    correlation analysis
    """
    plt.figure(figsize=(12, 10))
    # I will use Spearman because non-linearity of some clinical features 
    
    sns.heatmap(df.corr(method='spearman'), annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Spearman Correlation Matrix of Clinical Features')
    plt.show()