# Libraries
import pandas as pd
import numpy as np

#Load data function
def load_data(data):
    df = pd.read_csv(data)
    return df

#Summarise data
def summarise_dataframe(df):
    print("\n" + "="*50 + "\n")
    print(f"Summary for {df}:")
    print("\nHead:\n")
    print(df.head())
    
    print("\nInfo:\n")
    df.info()
    
    print("\nDescribe:\n")
    print(df.describe())
    print("\n" + "="*50 + "\n")

#Missing data
def missing_data(data):
    total = data.isnull().sum()
    percent = (data.isnull().sum()/data.isnull().count()*100)
    tt = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    types = []
    for col in data.columns:
        dtype = str(data[col].dtype)
        types.append(dtype)
    tt['Types'] = types
    df_missing = np.transpose(tt)
    return df_missing

#Most Frequent
def most_frequent(data):
    total = data.count()
    tt = pd.DataFrame(total)
    tt.columns = ['Total']
    items = []
    vals = []
    for col in data.columns:
        try:
            itm = data[col].value_counts().index[0]
            val = data[col].value_counts().values[0]
            items.append(itm)
            vals.append(val)
        except Exception as ex:
            print(ex)
            items.append(0)
            vals.append(0)
            continue
    tt['Most frequent item'] = items
    tt['Frequence'] = vals
    tt['Percent from total'] = np.round(vals / total * 100, 3)
    np.transpose(tt)
    return tt

#Unique values
def unique_values(data):
    total = data.count()
    tt = pd.DataFrame(total)
    tt.columns = ['Total']
    uniques = []
    for col in data.columns:
        unique = data[col].nunique()
        uniques.append(unique)
    tt['Uniques'] = uniques
    np.transpose(tt)
    return tt

### EDA


#Aggregate function
def agg_data(df1,df2):
    all_df = pd.concat([df1, df2], axis=0)
    all_df["set"] = "train"
    all_df.loc[all_df.Survived.isna(), "set"] = "test" #since the test data does not have a survived column as often this is what we would try to predict
    return all_df