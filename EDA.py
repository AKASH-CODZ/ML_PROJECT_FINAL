import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_per_column_distribution(df, max_unique=50, ncols=3, max_plots=12):
    """
    Plot histogram/bar plot for columns with < max_unique unique values.
    """
   
    nunique = df.nunique()
    cols = [col for col in df.columns if 1 < nunique[col] < max_unique]
    cols = cols[:max_plots] 
    
    nrows = int(np.ceil(len(cols) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6*ncols, 4*nrows))
    axes = axes.flatten()
    
    for i, col in enumerate(cols):
        if df[col].dtype == 'object' or df[col].dtype.name == 'category':
            sns.countplot(x=col, data=df, ax=axes[i], palette="Set2")
            axes[i].set_ylabel("Count")
        else:
            sns.histplot(df[col], kde=True, ax=axes[i], color="skyblue")
            axes[i].set_ylabel("Frequency")
        axes[i].set_title(f"{col} (unique={nunique[col]})")
        axes[i].tick_params(axis='x', rotation=45)
    
   
    for j in range(i+1, len(axes)):
        fig.delaxes(axes[j])
        
    plt.tight_layout()
    plt.show()



def plot_correlation_matrix(df, figsize=(10, 8)):
    """
    Plot correlation heatmap for numeric columns.
    """
    df_num = df.select_dtypes(include=[np.number]).dropna(axis=1)
    df_num = df_num.loc[:, df_num.nunique() > 1]
    
    if df_num.shape[1] < 2:
        print("Not enough numeric columns for correlation matrix.")
        return
    
    corr = df_num.corr()
    plt.figure(figsize=figsize)
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", cbar=True, square=True)
    plt.title("Correlation Matrix", fontsize=14)
    plt.show()



def plot_scatter_matrix(df, max_cols=6, figsize=(12, 12)):
    """
    Scatterplot matrix for numeric columns with KDE on diagonals.
    """
    df_num = df.select_dtypes(include=[np.number]).dropna(axis=1)
    df_num = df_num.loc[:, df_num.nunique() > 1]
    
    cols = df_num.columns[:max_cols] 
    sns.pairplot(df_num[cols], diag_kind="kde", corner=True, plot_kws={'alpha':0.6})
    plt.suptitle("Scatter & Density Plots", y=1.02, fontsize=14)
    plt.show()



df = pd.read_csv('your_data.csv')

plot_per_column_distribution(df)
plot_correlation_matrix(df)
plot_scatter_matrix(df)
