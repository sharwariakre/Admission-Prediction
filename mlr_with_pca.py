#importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Importing dataset
df = pd.read_csv("Admission_Prediction.csv")
#extracting x and y
x = df.iloc[:,1:-1].values
y = df.iloc[:,-1].values

#to find missing values if any
print("Sum of missing values:\n ",df.isna().sum())

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x = sc.fit_transform(x)

#apply pca
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
x= pca.fit_transform(x)
print(pca.explained_variance_ratio_)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=65)

#creating and training model and predicting value
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)
reg_score = (regressor.score(x_test, y_test))*100 
print("Regressor score: ", reg_score)

#to check mse rmse and r2_score
from sklearn.metrics import r2_score , mean_squared_error 
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2_sc = r2_score(y_test, y_pred)

print("mean squared error: " , mse)
print("root mean square error is: " , rmse)
print("r2_score" , r2_sc) 


#plotting graphs of significant columns vs chance of admit
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







