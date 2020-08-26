# Importing the libraries
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('train.csv')
features = ['disbursed_amount', 'asset_cost', 'ltv', 'Employment.Type',
            'MobileNo_Avl_Flag', 'Aadhar_flag', 'PAN_flag', 'Driving_flag', 'Passport_flag',
            'PERFORM_CNS.SCORE', 'PRI.NO.OF.ACCTS', 'PRI.ACTIVE.ACCTS',
            'PRI.OVERDUE.ACCTS', 'PRI.CURRENT.BALANCE', 'DELINQUENT.ACCTS.IN.LAST.SIX.MONTHS',
            'AVERAGE.ACCT.AGE', 'NO.OF_INQUIRIES', 'loan_default']

dataset = dataset[features]
dataset = dataset.dropna()
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder
labelencoder_X = LabelEncoder()
X[:, 3] = labelencoder_X.fit_transform(X[:, 3])

# Fitting the classifier to the Training set
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
classifier.fit(X, y)

# Saving the model
import _pickle as cPickle
with open('model.pkl', 'wb') as f:
    cPickle.dump(classifier, f)
print("Model Saved.")
