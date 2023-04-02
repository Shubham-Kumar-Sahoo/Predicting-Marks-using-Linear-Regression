# Importing the libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Creating the dataset

from numpy.random import rand

df  = pd.DataFrame(rand( 50 , 3)*20, columns = 'Hours_studied Hours_slept iQ '.split())
df.head(2)

from random import choice

location = []
for i in range(50):
    location.append(choice(["New Delhi" , "Pune" , "Bangalore"]))
 

df['Location'] = location

df.head()

df['Marks'] = (1.73 + (3.34*df['Hours_studied']) + (2.45*df['Hours_slept']) + (1.83*df['iQ']  ))
df['Marks'] =  df['Marks'] +np.random.rand(50)*20
               
df.to_csv('Students.csv',index=False)

df.head()

# Importing the dataset

dataset = pd.read_csv('Students.csv')

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

print('Input:-\n Hours studied\t\t Hours slept\t iQ\t\t Location')
print(X)

print('\nOutput:-\nMarks obtained')
print(y)

# Encoding categorical data (Location)

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])],  remainder='passthrough')
X = np.array(ct.fit_transform(X))

print('\n\nInput after encoding Location:-\n Location\tHours studied\t Hours slept\t   iQ')
print(X)

# Splitting the dataset into the Training set and Test set

from sklearn.model_selection import train_test_split
X_train, X_test, y_train , y_test = train_test_split(X, y, test_size = 0.06, random_state = 0)
print('\nTraining Dataset :-\n')
print('Training Input :-\n Location\tHours studied\t Hours slept\t   iQ')
print(X_train)
print('\nTraining Output :-\nMarks obtained')
print(y_train)

print('\nTesting Dataset :-\n')
print('Testing Input :-\n Location\tHours studied\t Hours slept\t   iQ')
print(X_test)
print('\nTesting Output :-\nMarks obtained')
print(y_test)

# Training the Multiple Linear Regression model on the Training set

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results

y_pred = regressor.predict(X_test)

np.set_printoptions(precision=2)
print('\nAfter training the model:-\nPredicted Marks vs Actual Marks')
print("\n\n\nPredicted  Actual")
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

# Finding The Accuracy

from sklearn.metrics import mean_absolute_error
accuracy_score=100-mean_absolute_error(y_test, y_pred)
print('\nAccuracy Score=',accuracy_score)

# Visulaize training data

plt.title("Predicted (Red) vs Actual (Blue)")
plt.xlabel("Imput")
plt.ylabel("Output")
plt.plot(X_test, y_pred,color='red')  
plt.plot(X_test, y_test,color='blue')
plt.show()




