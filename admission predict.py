# Importing essential libraries
import pandas as pd
import pickle

# Loading the dataset
df = pd.read_csv('admission_predict.csv')

# Renaming DiabetesPedigreeFunction as DPF
df = df.rename(columns={'AdmissionPedigreeFunction':'APF'})

# Renaming the columns with appropriate names
df = df.rename(columns={'Chance of Admit ': 'Probability'})
# Removing the serial no, column
df.drop('Serial No.', axis='columns', inplace=True)

# Splitting the dataset in features and label
X = df.drop('Probability', axis='columns')
y = df['Probability']


# Model Building
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

from sklearn.linear_model import LinearRegression
model = LinearRegression(normalize=True)
model.fit(X_train,y_train)


# Creating a pickle file for the classifier

pickle.dump(model, open('model.pkl', 'wb'))

model = pickle.load(open('model.pkl', 'rb'))
