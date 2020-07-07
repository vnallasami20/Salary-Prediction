#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from scipy import stats


# In[2]:


# function to create histogram, Q-Q plot and
# boxplot. We learned this in section 3 of the course


def diagnostic_plots(df, variable,col=None):
    # function takes a dataframe (df) and
    # the variable of interest as arguments

    # define figure size
    plt.figure(figsize=(16, 4))

    # histogram
    plt.subplot(1, 3, 1)
    sns.distplot(df[variable], bins=30,color=col,kde_kws={'bw':0.1})
    plt.title('Histogram')

    # Q-Q plot
    plt.subplot(1, 3, 2)
    stats.probplot(df[variable].astype(float), dist="norm", plot=plt)
    plt.ylabel('Variable quantiles')

    # boxplot
    plt.subplot(1, 3, 3)
    sns.boxplot(y=df[variable].astype(float),color=col)
    plt.title('Boxplot')

    plt.show()


# In[3]:


def monotomic_rel(df,target,var):
    
    pd.concat([df, target], axis=1).groupby([var])['SalePrice'].mean().plot()
    plt.title('Monotonic relationship between {} and {}'.format(var,'SalePrice'))
    plt.ylabel('SalePrice')
    
    plt.show()


# In[4]:


def cardinality_plot(df,cat_var_list):
    # plot number of categories per categorical variable

    df[cat_var_list].nunique().plot.bar(figsize=(10,6))
    plt.title('CARDINALITY: Number of categories in categorical variables')
    plt.xlabel('Categorical variables')
    plt.ylabel('Number of different categories')


# In[5]:


def find_non_rare_labels(df, variable, tolerance):
    
    temp = df.groupby([variable])[variable].count() / len(df)
    
    non_rare = [x for x in temp.loc[temp>tolerance].index.values]
    
    return non_rare


# In[6]:


def explore_data(df, pred=None): 
    obs = df.shape[0]
    types = df.dtypes
    counts = df.apply(lambda x: x.count())
    uniques = df.apply(lambda x: [x.unique()])
    nulls = df.apply(lambda x: x.isnull().sum())
    distincts = df.apply(lambda x: x.unique().shape[0])
    missing_ratio = (df.isnull().sum()/ obs) * 100
    skewness = df.skew()
    kurtosis = df.kurt() 
#     print('Data shape:', df.shape)
    
    if pred is None:
        cols = ['types', 'counts', 'distincts', 'nulls', 'missing ratio', 'uniques', 'skewness', 'kurtosis']
        str = pd.concat([types, counts, distincts, nulls, missing_ratio, uniques, skewness, kurtosis], axis = 1)

    else:
        corr = df.corr()[pred]
        str = pd.concat([types, counts, distincts, nulls, missing_ratio, uniques, skewness, kurtosis, corr], axis = 1, sort=False)
        corr_col = 'corr '  + pred
        cols = ['types', 'counts', 'distincts', 'nulls', 'missing_ration', 'uniques', 'skewness', 'kurtosis', corr_col ]
    
    str.columns = cols
#     dtypes = str.types.value_counts()
#     print('___________________________\nData types:\n',str.types.value_counts())
#     print('___________________________')
    return str


# In[7]:


def find_var_type(df):
    details = explore_data(df)
    
    cat_filt = (details.types =='object') # & (details.nulls > 0)
    num_filt = (details.types !='object') #& (details.nulls > 0)
    
    year_columns = [c for c in details.index if 'Yr' in c or 'Year' in c]
    cat_columns = [c for c in details[cat_filt].index]
    num_columns = [c for c in details[num_filt].index if c not in year_columns]
    
    discreate_columns = [c for c in details[num_filt].index if (details.loc[c].distincts <=20) & (c not in year_columns)]
    continues_columns = [c for c in details[num_filt].index if (c not in year_columns) & (c not in discreate_columns)]
    
    return cat_columns,discreate_columns,continues_columns,year_columns    


# In[8]:


def remove_duplicate_feature(df):
    dup_list = set()
    for i in range(0,len(df.columns)):
        col1 = df.columns[i]
        for col2 in df.columns[i+1:]:
            if df[col1].equals(df[col2]):
#                 print(col1,col2)
                dup_list.add(col2)
                
    df.drop(labels=dup_list,axis=1,inplace=True)
    return df,dup_list


# In[9]:


def correlation(dataset, threshold):
    col_corr = set()  # Set of all the names of correlated columns
    corr_matrix = dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > threshold: # we are interested in absolute coeff value
                colname = corr_matrix.columns[i]  # getting the name of column
                col_corr.add(colname)
    return col_corr


# In[ ]:


def transform_year(df,var,sold):
    df[var] = df[sold] - df[var]
    return df


