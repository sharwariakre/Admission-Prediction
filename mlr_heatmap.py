#importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='white')
sns.set(style='whitegrid' , color_codes=True)

#Importing dataset
df = pd.read_csv("Admission_Prediction.csv")
print(df.head())

#extracting x and y
x = df.iloc[:,1:-1].values
y = df.iloc[:,-1].values

df=df.rename(columns={'Serial No.':'no','GRE Score':'gre','TOEFL Score':'toefl','University Rating':'rating','SOP':'sop','LOR ':'lor',
                           'CGPA':'gpa','Research':'research','Chance of Admit ':'chance'})

#to find missing values if any
print("Sum of missing values:\n ",df.isna().sum())

#for heatmap
df.drop(labels='no', axis=1, inplace=True)
fig,ax = plt.subplots(figsize=(10,10))
sns.heatmap(df.corr(), annot=True, cmap='Blues')
plt.show()

# from the heatmap we can infer that the GRE, TOEFL and CGPA are the important features
x_opt = x[:,(0,1,5)]
print(x_opt)

#performing feature scaling on x_opt
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_opt = sc.fit_transform(x_opt)
np.set_printoptions(suppress=True)
print(x_opt)

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

#plotting graph of cgpa vs chance of admit
plt.scatter(x_test[:,2],y_test,color='red')
plt.plot(x_test[:,2],y_pred,color='blue')
plt.title("CGPA vs CHANCE OF ADMIT")
plt.xlabel("CGPA")
plt.ylabel("CHANCE OF ADMIT")
plt.show()

#plotting graph of gre vs chance of admit
plt.scatter(x_test[:,0],y_test,color='red')
plt.plot(x_test[:,0],y_pred,color='blue')
plt.title("GRE vs CHANCE OF ADMIT")
plt.xlabel("GRE")
plt.ylabel("CHANCE OF ADMIT")
plt.show()

#plotting graph of toefl vs chance of admit
plt.scatter(x_test[:,1],y_test,color='red')
plt.plot(x_test[:,0],y_pred,color='blue')
plt.title("TOEFL vs CHANCE OF ADMIT")
plt.xlabel("TOEFL")
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



