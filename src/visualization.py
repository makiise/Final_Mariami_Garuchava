import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

SURVIVAL_PALETTE = {0: "royalblue", 1: "crimson"}


def plot_continuous_distribution(df: pd.DataFrame, column: str):
    """
    plot KDE for any continuous variable.
    """
    plt.figure(figsize=(10, 5))
    sns.kdeplot(data=df, x=column, hue='DEATH_EVENT', fill=True, 
                palette=SURVIVAL_PALETTE, alpha=0.5, common_norm=False)
    plt.title(f'Distribution of {column.replace("_", " ").title()}')
    plt.legend(title='Status', labels=['Dead', 'Survived'])
    plt.show()


#I come up with that function to see 2 pictures of survivors and dead patients at once. 
# e.g. (1) - survival rate if they had diabetes and, (2) - if they didn't.

def plot_categorical_ratio(df, column):
    plt.figure(figsize=(10, 6))
    ax = sns.countplot(data=df, x=column, hue='DEATH_EVENT', palette=SURVIVAL_PALETTE)
    plt.title(f'Survival vs Death count by {column.replace("_", " ").title()}', fontsize=14)
    plt.xlabel(column.replace("_", " ").title(), fontsize=12)
    plt.ylabel('Number of Patients', fontsize=12)
    if df[column].nunique() == 2:
        plt.xticks([0, 1], [f'No {column.replace("_", " ").title()}', f'Has {column.replace("_", " ").title()}'])
    
    plt.legend(title='Outcome', labels=['Survived', 'Dead'])
    plt.show()



def plot_outlier_analysis(df: pd.DataFrame, column: str):
    plt.figure(figsize=(10, 5))
    sns.boxplot(
        data=df, 
        x='DEATH_EVENT', 
        y=column, 
        hue='DEATH_EVENT', 
        palette=SURVIVAL_PALETTE, 
        legend=False        
    )
    
    plt.title(f'Outlier Analysis: {column.replace("_", " ").title()}')
    plt.xticks([0, 1], ['Survived', 'Dead'])
    plt.show()


def plot_violin_sodium(df):
    plt.figure(figsize=(10, 6))
    sns.violinplot(data=df, x='DEATH_EVENT', y='serum_sodium', palette=SURVIVAL_PALETTE, hue='DEATH_EVENT')
    plt.title('Sodium Concentration: Survival vs Death')
    plt.xticks([0, 1], ['Survived', 'Dead'])
    plt.show()

def plot_correlation_heatmap(df: pd.DataFrame, corr_method):
    plt.figure(figsize=(12, 10))
    sns.heatmap(df.corr(method=corr_method), annot=True, cmap='bwr', fmt='.2f', center=0)
    plt.title(f'{corr_method}Correlation Matrix of Clinical Features')
    plt.show()

