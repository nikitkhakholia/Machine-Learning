#R-squared Score
from sklearn.linear_model import LinearRegression 
import numpy as np 
from numpy.linalg import inv,lstsq 
from numpy import dot, transpose
model = LinearRegression()
model.fit(volume_x,marketcap_y)
volume_x_test = [[4637030],[2554360],[3550790],[1942830],[1485680]]
marketcap_y_test = [[110672321.8],[102303608.5],[94901005.35],[87295366.5],[78868413.08]]
print ("Score = ",model.score(volume_x_test, marketcap_y_test))
