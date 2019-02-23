
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


dataset = pd.read_csv('emp.csv')
X = dataset.iloc[:,1:10]
y = dataset.iloc[:,10]

print(X)
print(y)

from sklearn.preprocessing import LabelEncoder
labelencoder_y=LabelEncoder()
y=labelencoder_y.fit_transform(y)
X['dept']=labelencoder_y.fit_transform(X['dept'])
X['salary']=labelencoder_y.fit_transform(X['salary'])

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=0)


d=dataset.describe(include=['object'])
print(d)
dataset['underperformer'] = ((dataset['last_evaluation'] < 0.6)).astype(int)

dataset['unhappy'] = (dataset['satisfaction'] < 0.2).astype(int)

dataset['overachiever'] = \
((dataset['last_evaluation'] > 0.8) & (dataset['satisfaction'] > 0.7)).astype(int)

print(dataset[['underperformer', 'unhappy', 'overachiever']].mean())



from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(X_train,y_train)

y_pred=regressor.predict(X_test)
df=pd.DataFrame({'Actual':y_test, 'Predicted':y_pred})
print(df)
from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

from sklearn import metrics
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))