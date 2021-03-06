#importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#importing dataset
df = pd.read_csv("Admission_Prediction.csv")
#print(df)

#extracting x and y
x = df.iloc[:,1:-1].values
y = df.iloc[:,-1].values
          
#performing feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x = sc.fit_transform(x)

#creating and training random forest model
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state =65)

from sklearn.ensemble import RandomForestRegressor
rfr = RandomForestRegressor(n_estimators=100,random_state=0)
rfr.fit(x,y)
y_pred = rfr.predict(x_test)
rfr_score = (rfr.score(x_test, y_test))*100 
print("Regressor score: ", rfr_score)


#to check mse rmse and r2_score
from sklearn.metrics import r2_score , mean_squared_error
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2_sc = r2_score(y_test, y_pred)

print("mean squared error: " , mse)
print("root mean square error is: " , rmse)
print("r2_score" , r2_sc) 


plt.scatter(x_test[:,0],y_test,c='red')
plt.plot(x_test[:,0],y_pred,c='blue')
plt.title("GRE vs CHANCE OF ADMIT")
plt.xlabel("GRE")
plt.ylabel("CHANCE OF ADMIT")
plt.show()

plt.scatter(x_test[:,1],y_test,c='red')
plt.plot(x_test[:,1],y_pred,c='blue')
plt.title("TOEFL vs CHANCE OF ADMIT")
plt.xlabel("TOEFL")
plt.ylabel("CHANCE OF ADMIT")
plt.show()

plt.scatter(x_test[:,2],y_test,c='red')
plt.plot(x_test[:,2],y_pred,c='blue')
plt.title("UNIVERSITY RATING vs CHANCE OF ADMIT")
plt.xlabel("UNIVERSITY RATING")
plt.ylabel("CHANCE OF ADMIT")
plt.show()

plt.scatter(x_test[:,3],y_test,c='red')
plt.plot(x_test[:,3],y_pred,c='blue')
plt.title("SOP vs CHANCE OF ADMIT")
plt.xlabel("SOP")
plt.ylabel("CHANCE OF ADMIT")
plt.show()

plt.scatter(x_test[:,4],y_test,c='red')
plt.plot(x_test[:,4],y_pred,c='blue')
plt.title("LOR vs CHANCE OF ADMIT")
plt.xlabel("LOR")
plt.ylabel("CHANCE OF ADMIT")
plt.show()

plt.scatter(x_test[:,5],y_test,c='red')
plt.plot(x_test[:,5],y_pred,c='blue')
plt.title("CGPA vs CHANCE OF ADMIT")
plt.xlabel("CGPA")
plt.ylabel("CHANCE OF ADMIT")
plt.show()

plt.scatter(x_test[:,6],y_test,c='red')
plt.plot(x_test[:,6],y_pred,c='blue')
plt.title("RESEARCH vs CHANCE OF ADMIT")
plt.xlabel("RESEARCH")
plt.ylabel("CHANCE OF ADMIT")
plt.show()



























