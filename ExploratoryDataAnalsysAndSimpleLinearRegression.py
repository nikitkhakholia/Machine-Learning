#Program for simple linear regression
from pandas import *
import matplotlib.pyplot as plt 
from sklearn.linear_model import LinearRegression 
# reading CSV file
data = read_csv("coin_Ethereum.csv",parse_dates=True)

data['Year'] = (pandas.to_datetime(data["Date"].str.strip(), format='%Y-%m-%d %H:%M:%S')).dt.year
year = data['Year'].tolist()
print(year)

volume= data['Volume'].tolist()
marketcap = data['Marketcap'].tolist()

#EDA
mean_volume = data['Volume'].mean()
mean_marketcap = data['Marketcap'].mean()
median_volume = data['Volume'].median()
median_marketcap = data['Marketcap'].median()
print('Mean Volume:',mean_volume)
print('Mean Marketcap:',mean_marketcap)
print('Median Volume:',median_volume)
print('Median Marketcap:',meadian_marketcap)

# volume.sort(reverse=True)
# print(volume)
volume_x = []
marketcap_y = []

for i in volume:
  volume_x.append([i])

for i in marketcap:
  marketcap_y.append([i])


model = LinearRegression()
model.fit(volume_x,marketcap_y) 
plt.figure()
plt.title('Ethereum Market History')
plt.xlabel('Volume')
plt.ylabel('Marketcap ($)')
plt.plot(volume_x,marketcap_y,'.')
plt.plot(volume_x,model.predict(volume_x),'--')

plt.grid(True) 
print ("Predicted marketcap = ",model.predict([[90000000000]])) # 22.467

plt.show()
