# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def convert(df, col):
    temp_series = pd.Series([], dtype='int64')
    for i in df[col]:
        temp_1 = int(i[:i.find('yrs')])
        temp_2 = int(i[i.find(' ')+1:i.find('mon')])
        temp_3 = (temp_1*12) + temp_2
        temp_series = temp_series.append(pd.Series(temp_3), ignore_index = True)
    return temp_series

# Load dataset
dataset = pd.read_csv('data.csv')
dataset = dataset.dropna()
dataset = dataset.drop(['Date.of.Birth', 'DisbursalDate'], axis=1)

dataset['AVERAGE.ACCT.AGE'] = convert(dataset, 'AVERAGE.ACCT.AGE')
dataset['CREDIT.HISTORY.LENGTH'] = convert(dataset, 'CREDIT.HISTORY.LENGTH')

"""
dataset = dataset[['disbursed_amount', 'asset_cost', 'ltv', 'Employment.Type', 'PERFORM_CNS.SCORE', 'PRI.ACTIVE.ACCTS',
                   'PRI.CURRENT.BALANCE', 'PRI.SANCTIONED.AMOUNT', 'CREDIT.HISTORY.LENGTH', 'loan_default']]
"""

dataset.to_csv('train.csv', index=False)
print("File Saved")
