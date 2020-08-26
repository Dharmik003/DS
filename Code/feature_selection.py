# Kernel SVM

# Importing the libraries
import pandas as pd
import numpy as np

# Importing the dataset
dataset = pd.read_csv('train.csv')
dataset = dataset.dropna()
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder
labelencoder_X = LabelEncoder()
X[:, 8] = labelencoder_X.fit_transform(X[:, 8])
X[:, 18] = labelencoder_X.fit_transform(X[:, 18])

# Building the optimal model using Backward Elimination
# import statsmodels.formula.api as sm
import statsmodels.api as sm
# This statement adds a new column containg only 1s for b0 variable of the equation
X = np.append(arr=np.ones((217991,1)).astype(int), values=X, axis=1)
X_opt = X.astype('int64')
y = y.astype('int64')
coef_labels = ['Constant', 'UniqueID', 'disbursed_amount', 'asset_cost', 'ltv', 'branch_id', 'supplier_id',
               'manufacturer_id', 'Current_pincode_ID', 'Employment.Type', 'State_ID', 'Employee_code_ID',
               'MobileNo_Avl_Flag', 'Aadhar_flag', 'PAN_flag', 'VoterID_flag', 'Driving_flag', 'Passport_flag',
               'PERFORM_CNS.SCORE', 'PERFORM_CNS.SCORE.DESCRIPTION', 'PRI.NO.OF.ACCTS', 'PRI.ACTIVE.ACCTS',
               'PRI.OVERDUE.ACCTS', 'PRI.CURRENT.BALANCE', 'PRI.SANCTIONED.AMOUNT', 'PRI.DISBURSED.AMOUNT',
               'SEC.NO.OF.ACCTS', 'SEC.ACTIVE.ACCTS', 'SEC.OVERDUE.ACCTS', 'SEC.CURRENT.BALANCE', 'SEC.SANCTIONED.AMOUNT',
               'SEC.DISBURSED.AMOUNT', 'PRIMARY.INSTAL.AMT', 'SEC.INSTAL.AMT', 'NEW.ACCTS.IN.LAST.SIX.MONTHS',
               'DELINQUENT.ACCTS.IN.LAST.SIX.MONTHS', 'AVERAGE.ACCT.AGE', 'CREDIT.HISTORY.LENGTH', 'NO.OF_INQUIRIES']

print(type(X_opt))
print(type(y))
while(True):
    regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
    print(regressor_OLS.summary(xname=coef_labels))
    print()
    p_values = list(regressor_OLS.pvalues)
    
    p_max = max(p_values)
    p_max_idx = p_values.index(p_max)
    
    if(p_max > 0.05):
        X_opt = np.delete(X_opt, [p_max_idx], axis=1)
        del coef_labels[p_max_idx]
    else:
        break
