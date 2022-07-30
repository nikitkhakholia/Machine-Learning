#Multiple Linear Regression 
from sklearn.linear_model import LinearRegression 
from numpy.linalg import lstsq 
import numpy as np 
x = [[674188, 2015], [532170, 2015], [8631090, 2016], [6413550, 2016], [66046600, 2017]]
y = [[45486894.24], [42399573.5], [335088727.4], [359310563.1], [8317148601]]
model = LinearRegression()
model.fit(x,y)
x1 = [[8, 2], [9, 0], [11, 2], [16, 2], [12, 0]]
y1 = [[11], [8.5], [15], [18], [11]]
predictions = model.predict([[8, 2], [9, 0], [12, 0]])
print ("values of Predictions: ",predictions)
print ("values of β1, β2: ",lstsq(x, y, rcond=None)[0])
#least-squares solution to find best regression line
#It is a cut-off ratio for smaller singular values of  x and Y 
print ("Score = ",model.score(x1, y1)) 
