#!/usr/bin/env python
# coding: utf-8

# In[10]:


import pandas as pd


# In[11]:


import os


# In[12]:


print(os.getcwd())


# In[13]:


import numpy as np
import sklearn
import sklearn


# In[14]:


bigmart_train=pd.read_csv('Train.csv')


# In[15]:


#shape of the data set
bigmart_train.shape


# In[16]:


#first 5 raws of data 
bigmart_train.head()


# In[17]:


bigmart_train.tail()


# In[18]:


#checking data set info
bigmart_train.info()


# In[19]:


bigmart_train.describe().T


# In[20]:


bigmart_train.describe()


# In[21]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[22]:


bigmart_train.hist(figsize=(15,12))


# In[23]:


bigmart_train.isnull().sum()


# In[24]:


bigmart_train.nunique()


# In[25]:


bigmart_train['Outlet_Size'] = bigmart_train['Outlet_Size'].map({"Small":1,"Medium":2,"High":3})


# In[26]:


bigmart_train['Item_Weight']=bigmart_train['Item_Weight'].fillna(bigmart_train['Item_Weight'].median())


# In[27]:


bigmart_train['Outlet_Size'].fillna(bigmart_train['Outlet_Size'].mode()[0], inplace=True)


# In[28]:


bigmart_train.isnull().sum()


# In[29]:


bigmart_train.head() 


# In[30]:


import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['figure.figsize'] = (10,10)
plt.hist(bigmart_train["Item_Outlet_Sales"],bins = 100)
# plt.show()


# In[31]:


# plt.rcParams['figure.figsize'] = (10,10)
plt.hist(bigmart_train["Item_MRP"],alpha = 0.3,bins = 150)
plt.show()


# In[32]:


bigmart_train.drop(labels = ["Outlet_Establishment_Year"],inplace = True,axis =1)


# In[33]:


fig,axes=plt.subplots(1,1,figsize=(12,8))
sns.scatterplot(x='Item_MRP',y='Item_Outlet_Sales',hue='Item_Fat_Content',size='Item_Weight',data=bigmart_train)


# In[34]:


from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection  import cross_val_score
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor


# In[35]:


feat = ['Outlet_Size','Outlet_Type','Outlet_Location_Type','Item_Fat_Content',"Item_Type"]
X = pd.get_dummies(bigmart_train[feat])
bigmart_train = pd.concat([bigmart_train,X],axis=1)


# In[36]:


bigmart_train.head()


# In[37]:


X_train = bigmart_train.drop(labels = ["Item_Outlet_Sales"],axis=1)
y_train = bigmart_train["Item_Outlet_Sales"]
X_train.shape,y_train.shape


# In[38]:


X_train.drop(labels = ["Outlet_Size",'Outlet_Location_Type',"Outlet_Type",'Item_Fat_Content','Outlet_Identifier','Item_Identifier',"Item_Type"],axis=1,inplace = True)


# In[39]:


X_train.head()


# In[40]:


from sklearn import preprocessing
x = X_train.values #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled_train = min_max_scaler.fit_transform(x)
df_train = pd.DataFrame(x_scaled_train)


# In[41]:


df_train.head()


# In[42]:


#returns a numpy array
Sales_norm = (y_train-np.min(y_train))/(np.max(y_train)-np.min(y_train))


# In[43]:


Sales_norm.head()


# In[44]:


y_train= Sales_norm


# In[45]:


from sklearn.model_selection import train_test_split


# In[46]:


X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.20)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)


# In[47]:


X_train = np.nan_to_num(X_train)
X_test = np.nan_to_num(X_test)


# In[44]:


from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


# In[45]:


mlr = LinearRegression()  
mlr.fit(X_train, y_train)


# In[46]:


print("Intercept: ", mlr.intercept_)
print("Coefficients:")
list(zip(x, mlr.coef_))


# In[47]:


#Prediction of test set
y_pred_mlr= mlr.predict(X_test)
#Predicted values
print("Prediction for test set: {}".format(y_pred_mlr))


# In[48]:


mlr_diff = pd.DataFrame({'Actual value': y_test, 'Predicted value': y_pred_mlr})
mlr_diff.head()


# In[49]:


from sklearn import metrics
meanAbErr = metrics.mean_absolute_error(y_test, y_pred_mlr)
meanSqErr = metrics.mean_squared_error(y_test, y_pred_mlr)
rootMeanSqErr = np.sqrt(metrics.mean_squared_error(y_test, y_pred_mlr))
print('Mean Absolute Error:', meanAbErr)
print('Mean Square Error:', meanSqErr)
print('Root Mean Square Error:', rootMeanSqErr)


# In[50]:


print('R squared: {:.2f}',y_test,y_pred_mlr)


# In[51]:


r2_error = np.mean((y_test / y_pred_mlr)*100)
r2_error


# In[73]:


# training model wioth XG boost algorithem
import xgboost as xgb
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


# In[74]:


# Define different learning rates and max_depth parameters
learning_rates = [0.3, 0.5, 0.6, 0.7, 0.8]
max_depths = [11, 20, 21, 25, 31]


# In[75]:


# Dictionary to store the results
results = {}


# In[76]:


# Iterate over different combinations of learning rates and max_depth parameters
for lr in learning_rates:
    for md in max_depths:
        # Create an XGBoost regression model with specific learning rate and max_depth
        model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=lr, max_depth=md)

        # Train the model
        model.fit(X_train, y_train.ravel())

        # Make predictions on the test data
        y_pred = model.predict(X_test)

        # Calculate the RMSE
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        # Calculate the R² score
        r2 = r2_score(y_test, y_pred)

        # Store the results
        results[(lr, md)] = (rmse, r2)

        print(f"Learning Rate: {lr}, Max Depth: {md}, RMSE: {rmse}, R² Score: {r2}")


# In[54]:


# Find the best combination based on R² score
best_params = max(results, key=lambda k: results[k][1])
best_rmse, best_r2 = results[best_params]

print(f"\nBest Parameters -> Learning Rate: {best_params[0]}, Max Depth: {best_params[1]}")
print(f"Best RMSE: {best_rmse}, Best R² Score: {best_r2}")


# In[55]:


# Plotting the results
learning_rates, max_depths, rmses, r2_scores = zip(*[(lr, md, res[0], res[1]) for (lr, md), res in results.items()])

plt.figure(figsize=(14, 7))

# Plot R² scores
plt.subplot(1, 2, 1)
plt.scatter(max_depths, r2_scores, c=learning_rates, cmap='viridis')
plt.colorbar(label='Learning Rate')
plt.xlabel('Max Depth')
plt.ylabel('R² Score')
plt.title('R² Score for different Max Depth and Learning Rate')


# In[56]:


# Plot RMSE values
plt.subplot(1, 2, 2)
plt.scatter(max_depths, rmses, c=learning_rates, cmap='viridis')
plt.colorbar(label='Learning Rate')
plt.xlabel('Max Depth')
plt.ylabel('RMSE')
plt.title('RMSE for different Max Depth and Learning Rate')

plt.tight_layout()
plt.show()


# In[57]:


#selceted model
# Create an XGBoost regression model
model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1, max_depth=3)


# In[58]:


# Train the model
model.fit(X_train, y_train.ravel())

# Make predictions on the test data
y_pred = model.predict(X_test)


# In[59]:


# Calculate the Root Mean Square Error
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

# Calculate the R² score
r2 = r2_score(y_test, y_pred)

print(f"Root Mean Square Error: {rmse}")
print(f"R² Score: {r2}")


# In[61]:


Accuracy = r2 * 100
print(f"Best model accuracy: {Accuracy}")


# In[2]:





# In[ ]:





# In[93]:


# Plotting the results
plt.figure(figsize=(14, 7))

# Scatter plot of true vs predicted values
plt.subplot(1, 2, 1)
plt.scatter(y_test, y_pred, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel('True Values')
plt.ylabel('Predicted Values')
plt.title('True vs Predicted Values')


# In[94]:


# Residual plot
plt.subplot(1, 2, 2)
residuals = y_test - y_pred
plt.scatter(y_pred, residuals, color='red')
plt.hlines(0, y_pred.min(), y_pred.max(), colors='k', linestyles='dashed')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residuals vs Predicted Values')

plt.tight_layout()
plt.show()


# In[ ]:


# random forest algorithem 


# In[62]:


# Define different n_estimators and max_features parameters
n_estimators_list = [50, 100, 150, 200, 250]
max_features_list = [0.3, 0.5, 0.7, 0.9, 1.0]

# Dictionary to store the results
results = {}


# In[63]:


# Iterate over different combinations of n_estimators and max_features parameters
for n_estimators in n_estimators_list:
    for max_features in max_features_list:
        # Create a Random Forest regression model with specific n_estimators and max_features
        model = RandomForestRegressor(n_estimators=n_estimators, max_features=max_features, random_state=42)

        # Train the model
        model.fit(X_train, y_train)

        # Make predictions on the test data
        y_pred = model.predict(X_test)

        # Calculate the RMSE
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        # Calculate the R² score
        r2 = r2_score(y_test, y_pred)

        # Store the results
        results[(n_estimators, max_features)] = (rmse, r2)

        print(f"n_estimators: {n_estimators}, max_features: {max_features}, RMSE: {rmse}, R² Score: {r2}")


# In[64]:


# Find the best combination based on R² score
best_params = max(results, key=lambda k: results[k][1])
best_rmse, best_r2 = results[best_params]

print(f"\nBest Parameters -> n_estimators: {best_params[0]}, max_features: {best_params[1]}")
print(f"Best RMSE: {best_rmse}, Best R² Score: {best_r2}")


# In[65]:


# Plotting the results
n_estimators_list, max_features_list, rmses, r2_scores = zip(*[(ne, mf, res[0], res[1]) for (ne, mf), res in results.items()])

plt.figure(figsize=(14, 7))

# Plot R² scores
plt.subplot(1, 2, 1)
sc = plt.scatter(n_estimators_list, r2_scores, c=max_features_list, cmap='viridis')
plt.colorbar(sc, label='max_features')
plt.xlabel('n_estimators')
plt.ylabel('R² Score')
plt.title('R² Score for different n_estimators and max_features')


# In[66]:


# Plot RMSE values
plt.subplot(1, 2, 2)
sc = plt.scatter(n_estimators_list, rmses, c=max_features_list, cmap='viridis')
plt.colorbar(sc, label='max_features')
plt.xlabel('n_estimators')
plt.ylabel('RMSE')
plt.title('RMSE for different n_estimators and max_features')

plt.tight_layout()
plt.show()


# In[101]:


# Random forest regesser selcted model with parameters 


# In[ ]:





# In[67]:


# Create a Random Forest regression model
model = RandomForestRegressor(n_estimators=150,max_features = 0.5,random_state=42)


# In[68]:


# Train the model
model.fit(X_train, y_train)


# In[70]:


# Calculate the R-squared value
# Calculate the R-squared value
r2 = r2_score(y_test, y_pred)

print(f"Root Mean Square Error: {rmse}")
print(f"R-squared: {r2}")


# In[71]:


Accuracy = r2 * 100
print(f"Best model accuracy: {Accuracy}")


# 

# In[91]:


# Plotting the results
plt.figure(figsize=(14, 7))

# Scatter plot of true vs predicted values
plt.subplot(1, 2, 1)
plt.scatter(y_test, y_pred, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel('True Values')
plt.ylabel('Predicted Values')
plt.title('True vs Predicted Values')


# In[92]:


# Residual plot
plt.subplot(1, 2, 2)
residuals = y_test - y_pred
plt.scatter(y_pred, residuals, color='red')
plt.hlines(0, y_pred.min(), y_pred.max(), colors='k', linestyles='dashed')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residuals vs Predicted Values')

plt.tight_layout()
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[51]:


import seaborn as sns
import matplotlib.pyplot as plt


# In[ ]:


numerical_features = ['Item_Weight', 'Item_Visibility', 'Item_MRP', 'Outlet_Establishment_Year']
numerical_bigmart_train = bigmart_train[numerical_features]


# In[ ]:





# In[ ]:





# In[ ]:





# In[52]:


# Plot covariance heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(covariance_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Covariance Heatmap')
plt.show()


# In[47]:


# Extract numerical features (excluding 'output_sales_value')
numerical_features = ['Item_Weight', 'Item_Visibility', 'Item_MRP', 'Outlet_Establishment_Year']
numerical_bigmart_train = bigmart_train[numerical_features]


# In[49]:


# Calculate covariance matrix
covariance_matrix = numerical_bigmart_train.cov()


# In[ ]:





# In[28]:


# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[29]:


rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)


# In[30]:


# Train the model
rf_regressor.fit(X_train, y_train)
y_pred = rf_regressor.predict(X_test)


# In[32]:


from sklearn.metrics import mean_squared_error


# In[33]:


# Calculate Mean Squared Error (MSE)
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)


# In[36]:


X_test


# In[35]:


y_pred


# In[ ]:




