from dataclasses import dataclass
import pandas as pd
df = pd.read_csv("/content/content/supermarket_sales - Sheet1.csv", encoding= 'unicode_escape')
df.head()

df.isnull().any()

print(df.apply(lambda col: col.unique()))

df.head()

df.rename(columns = {'Invoice ID':'new_col'}, inplace = True)
df.head()

print(df.nunique())

from sklearn.preprocessing import LabelEncoder
labelEn = LabelEncoder()
df['Gender'] = labelEn.fit_transform(df['Gender'])
df['Branch'] = labelEn.fit_transform(df['Branch'])
df['Customer type'] = labelEn.fit_transform(df['Customer type'])
df['Payment'] = labelEn.fit_transform(df['Payment'])
df.head()

df1 = df.drop(["new_col","City","Product line","Date","Time"],axis=1)
df1

df1['Unit price'] = df1['Unit price'].astype('int')
df1['Tax 5%'] = df1['Tax 5%'].astype('int')
df1['Total'] = df1['Total'].astype('int')
df1['gross margin percentage'] = df1['gross margin percentage'].astype('int')
df1['cogs'] = df1['cogs'].astype('int')
df1['gross income'] = df1['gross income'].astype('int')
df1['Rating'] = df1['Rating'].astype('int')
df1


from pandas import DataFrame
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot

data = df1.values[:, :-1]
# perform a robust scaler transform of the dataset
trans = MinMaxScaler()
data = trans.fit_transform(data)
# convert the array back to a dataframe
nrmldata = DataFrame(data)
# summarize
print(nrmldata.describe())

df1.info()

df1.head(1)

#split the data set into independent (X) and dependent (Y) data sets
x= nrmldata.iloc[:, [2,3]].values  
y= nrmldata.iloc[:, 1].values 

# Splitting the dataset into training and test set.  
from sklearn.model_selection import train_test_split  
x_train, x_test, y_train, y_test= train_test_split(x, y, test_size= 0.25, random_state=0)  

# Training the K-NN model on the Training set
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 2, metric = 'euclidean', p = 2)
classifier.fit(x_train, y_train)

y_pred = classifier.predict(x_test)

y_test

y_pred

import numpy as np

error = []
# Calculating error for K values between 1 and 40
for i in range(1, 40):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(x_train, y_train)
    pred_i = knn.predict(x_test)
    error.append(np.mean(pred_i != y_test))
    
    
    
import matplotlib.pyplot as plt
plt.figure(figsize=(13,8))
plt.plot(range(1,40), error, linestyle = 'dotted', marker = 'o',color = 'g')
plt.xlabel('K value')
plt.ylabel('Error Rate')
plt.title('K value Vs Error Rate')
plt.show() 



from sklearn.metrics import confusion_matrix,accuracy_score
cm = confusion_matrix(y_test, y_pred)
ac = accuracy_score(y_test,y_pred)


cm

ac


#KNN on the basis of manhattan distance measure

from math import sqrt
 
# calculate manhattan distance
def manhattan_distance(a, b):
  return sum(abs(e1-e2) for e1, e2 in zip(a,b))
 
# define data
row1=df["Gender"]
row2=df["Total"]
# calculate distance
dist = manhattan_distance(row1, row2)
print(dist)





#KNN on the basis of euclidean distance measure

from math import sqrt
 
# calculate euclidean distance
def euclidean_distance(a, b):
  return sqrt(sum((e1-e2)**2 for e1, e2 in zip(a,b)))
 
# define data
row1 = df["Gender"]
row2 = df["Total"]
# calculate distance
dist = euclidean_distance(row1, row2)
print(dist)



from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(x_train, y_train)
 
# making predictions on the testing set
y_pred = gnb.predict(x_test)
 
# comparing actual response values (y_test) with predicted response values (y_pred)
from sklearn import metrics
print("Gaussian Naive Bayes model accuracy(in %):", metrics.accuracy_score(y_test, y_pred)*100)



