#importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Importing dataset
df = pd.read_csv("Admission_Prediction.csv")
print(df.head())


#data types of features
print(df.dtypes)
#to find missing values if any
print("Sum of missing values:\n ",df.isna().sum())

#extracting x and y
x = df.iloc[:,1:-1].values
y = df.iloc[:,-1].values

#performing feature scaling on the extracted columns
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x = sc.fit_transform(x)

#performing linear regression without backward elimination to check accuracy of model if trained with all the features
#splitting x and y into training and test data
from sklearn.model_selection import train_test_split
x_tr,x_te,y_tr,y_te = train_test_split(x,y,test_size=0.2,random_state = 65) 

#creating and training linear regression model
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x_tr,y_tr)
y_pr = lr.predict(x_te)

#to check mse rmse and r2_score
from sklearn.metrics import r2_score , mean_squared_error 
mse = mean_squared_error(y_te, y_pr)
rmse = np.sqrt(mse)
r2_sc = r2_score(y_te, y_pr)
reg_score = (lr.score(x_te, y_te))*100 
print("Regressor score before Backward Elimination ", reg_score)
print("mean squared error before backward elimination: " , mse)
print("root mean square error before backward elimination: " , rmse)
print("r2_score before backward elimination: " , r2_sc) 



x_1 = x[:,:]
#adding constant column(b)
const = np.ones((500,1)).astype(int)
x_1 = np.append(arr=const,values = x_1,axis = 1)
print(x_1)

#Performing Backward Elimination
import statsmodels.api as sm
x_opt = np.array(x_1[:,[0,1,2,3,4,5,6,7]],dtype = float)
model = sm.OLS(endog = y , exog = x_opt).fit()
print(model.summary())

#removing 4th column
x_opt = np.array(x_1[:,[0,1,2,3,5,6,7]],dtype = float)
model = sm.OLS(endog = y , exog = x_opt).fit()
print(model.summary())

#removing 3rd column
x_opt = np.array(x_1[:,[0,1,2,5,6,7]],dtype = float)
model = sm.OLS(endog = y , exog = x_opt).fit()
print(model.summary())

#removing 2nd column
x_opt = np.array(x_1[:,[0,1,5,6,7]],dtype = float)
model = sm.OLS(endog = y , exog = x_opt).fit()
print(model.summary())


#significant columns obtained, now training dataset
#split x_opt,y to training set and test set

from sklearn.model_selection import train_test_split
x_train , x_test , y_train , y_test = train_test_split(x_opt,y,test_size = 0.2, random_state = 65)

#creating and training model and predicting value
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)
regressor_score = (regressor.score(x_test, y_test))*100 
print("Regressor score: ", regressor_score)

#plotting graphs
#plotting graph of chance of admit vs gre
plt.scatter(x_test[:,1],y_test,color='red')
plt.plot(x_test[:,1],y_pred,color='blue')
plt.title("CHANCE OF ADMIT vs GRE")
plt.xlabel("GRE")
plt.ylabel("CHANCE OF ADMIT")
plt.show()

#graph of lor vs chance of admit
plt.scatter(x_test[:,2],y_test,color='red')
plt.plot(x_test[:,2],y_pred,color='blue')
plt.title("CHANCE OF ADMIT vs LOR")
plt.xlabel("LOR")
plt.ylabel("CHANCE OF ADMIT")
plt.show()

#graph of chance of admit vs cgpa
plt.scatter(x_test[:,3],y_test,color='red')
plt.plot(x_test[:,3],y_pred,color='blue')
plt.title("CHANCE OF ADMIT vs CGPA")
plt.xlabel("CGPA")
plt.ylabel("CHANCE OF ADMIT")
plt.show()

#graph of chance of admit vs research
plt.scatter(x_test[:,4],y_test,color='red')
plt.plot(x_test[:,4],y_pred,color='blue')
plt.title("CHANCE OF ADMIT vs RESEARCH")
plt.xlabel("RESEARCH")
plt.ylabel("CHANCE OF ADMIT")
plt.show()

#to check mse rmse and r2_score
from sklearn.metrics import r2_score , mean_squared_error 
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2_sc = r2_score(y_test, y_pred)
print("mean squared error: " , mse)
print("root mean square error is: " , rmse)
print("r2_score" , r2_sc) 




