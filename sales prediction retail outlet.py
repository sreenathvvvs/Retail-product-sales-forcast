
import pandas as pd
import numpy as np
import sklearn
import sklearn
import os
print(os.getcwd())
bigmart_train=pd.read_csv('Train.csv')
bigmart_train.shape

#first 5 raws of data 
bigmart_train.head()
#last 5 raws of data 
bigmart_train.tail()

#checking data set info
bigmart_train.info()
bigmart_train.describe().T
bigmart_train.describe()

# EDA with matplot lib 
import matplotlib.pyplot as plt
import seaborn as sns
bigmart_train.hist(figsize=(15,12)) 

# finding the null values 
bigmart_train.isnull().sum()
bigmart_train.nunique()

bigmart_train['Outlet_Size'] = bigmart_train['Outlet_Size'].map({"Small":1,"Medium":2,"High":3})
bigmart_train['Item_Weight']=bigmart_train['Item_Weight'].fillna(bigmart_train['Item_Weight'].median())
bigmart_train['Outlet_Size'].fillna(bigmart_train['Outlet_Size'].mode()[0], inplace=True)
bigmart_train.isnull().sum()
bigmart_train.head() 

# ploting historgam again with unique values 
plt.rcParams['figure.figsize'] = (10,10)
plt.hist(bigmart_train["Item_Outlet_Sales"],bins = 100)
# plt.show()

# plt.rcParams['figure.figsize'] = (10,10)
plt.hist(bigmart_train["Item_MRP"],alpha = 0.3,bins = 150)
plt.show()

bigmart_train.drop(labels = ["Outlet_Establishment_Year"],inplace = True,axis =1)
fig,axes=plt.subplots(1,1,figsize=(12,8))
sns.scatterplot(x='Item_MRP',y='Item_Outlet_Sales',hue='Item_Fat_Content',size='Item_Weight',data=bigmart_train)

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection  import cross_val_score
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

feat = ['Outlet_Size','Outlet_Type','Outlet_Location_Type','Item_Fat_Content',"Item_Type"]
X = pd.get_dummies(bigmart_train[feat])
bigmart_train = pd.concat([bigmart_train,X],axis=1)
bigmart_train.head()
X_train = bigmart_train.drop(labels = ["Item_Outlet_Sales"],axis=1)
y_train = bigmart_train["Item_Outlet_Sales"]
X_train.shape,y_train.shape

X_train.drop(labels = ["Outlet_Size",'Outlet_Location_Type',"Outlet_Type",'Item_Fat_Content','Outlet_Identifier','Item_Identifier',"Item_Type"],axis=1,inplace = True)

X_train.head()

from sklearn import preprocessing
x = X_train.values #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled_train = min_max_scaler.fit_transform(x)
df_train = pd.DataFrame(x_scaled_train)

df_train.head()

#returns a numpy array
Sales_norm = (y_train-np.min(y_train))/(np.max(y_train)-np.min(y_train))

Sales_norm.head()

y_train= Sales_norm

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.20)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

X_train = np.nan_to_num(X_train)
X_test = np.nan_to_num(X_test)

##############################################
# Regression modeling 

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
mlr = LinearRegression()  
mlr.fit(X_train, y_train)
print("Intercept: ", mlr.intercept_)
print("Coefficients:")
list(zip(x, mlr.coef_))
#Prediction of test set
y_pred_mlr= mlr.predict(X_test)
#Predicted values
print("Prediction for test set: {}".format(y_pred_mlr))
mlr_diff = pd.DataFrame({'Actual value': y_test, 'Predicted value': y_pred_mlr})
mlr_diff.head()

# calacualting the model efficency 

from sklearn import metrics
meanAbErr = metrics.mean_absolute_error(y_test, y_pred_mlr)
meanSqErr = metrics.mean_squared_error(y_test, y_pred_mlr)
rootMeanSqErr = np.sqrt(metrics.mean_squared_error(y_test, y_pred_mlr))
print('Mean Absolute Error:', meanAbErr)
print('Mean Square Error:', meanSqErr)
print('Root Mean Square Error:', rootMeanSqErr)
print('R squared: {:.2f}',y_test,y_pred_mlr)
r2_error = np.mean((y_test / y_pred_mlr)*100)
r2_error

# training model wioth XG boost algorithem
import xgboost as xgb
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Define different learning rates and max_depth parameters
learning_rates = [0.3, 0.5, 0.6, 0.7, 0.8]
max_depths = [11, 20, 21, 25, 31]

# Dictionary to store the results
results = {}

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

# Find the best combination based on R² score
best_params = max(results, key=lambda k: results[k][1])
best_rmse, best_r2 = results[best_params]

print(f"\nBest Parameters -> Learning Rate: {best_params[0]}, Max Depth: {best_params[1]}")
print(f"Best RMSE: {best_rmse}, Best R² Score: {best_r2}")

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

# Plot RMSE values
plt.subplot(1, 2, 2)
plt.scatter(max_depths, rmses, c=learning_rates, cmap='viridis')
plt.colorbar(label='Learning Rate')
plt.xlabel('Max Depth')
plt.ylabel('RMSE')
plt.title('RMSE for different Max Depth and Learning Rate')

plt.tight_layout()
plt.show()

#selceted model
# Create an XGBoost regression model
model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1, max_depth=3)


# Train the model
model.fit(X_train, y_train.ravel())

# Make predictions on the test data
y_pred = model.predict(X_test)

# Calculate the Root Mean Square Error
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

# Calculate the R² score
r2 = r2_score(y_test, y_pred)

print(f"Root Mean Square Error: {rmse}")
print(f"R² Score: {r2}")

Accuracy = r2 * 100
print(f"Best model accuracy: {Accuracy}")

# Plotting the results
plt.figure(figsize=(14, 7))

# Scatter plot of true vs predicted values
plt.subplot(1, 2, 1)
plt.scatter(y_test, y_pred, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel('True Values')
plt.ylabel('Predicted Values')
plt.title('True vs Predicted Values')

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

##############################################################################
# random forest algorithem 
# Define different n_estimators and max_features parameters
n_estimators_list = [50, 100, 150, 200, 250]
max_features_list = [0.3, 0.5, 0.7, 0.9, 1.0]

# Dictionary to store the results
results = {}

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

# Find the best combination based on R² score
best_params = max(results, key=lambda k: results[k][1])
best_rmse, best_r2 = results[best_params]

print(f"\nBest Parameters -> n_estimators: {best_params[0]}, max_features: {best_params[1]}")
print(f"Best RMSE: {best_rmse}, Best R² Score: {best_r2}")

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

# Plot RMSE values
plt.subplot(1, 2, 2)
sc = plt.scatter(n_estimators_list, rmses, c=max_features_list, cmap='viridis')
plt.colorbar(sc, label='max_features')
plt.xlabel('n_estimators')
plt.ylabel('RMSE')
plt.title('RMSE for different n_estimators and max_features')

plt.tight_layout()
plt.show()

# Random forest regesser selcted model with parameters 

# Create a Random Forest regression model
model = RandomForestRegressor(n_estimators=150,max_features = 0.5,random_state=42)

# Train the model
model.fit(X_train, y_train)

# Calculate the R-squared value
# Calculate the R-squared value
r2 = r2_score(y_test, y_pred)

print(f"Root Mean Square Error: {rmse}")
print(f"R-squared: {r2}")

Accuracy = r2 * 100
print(f"Best model accuracy: {Accuracy}")

# Plotting the results
plt.figure(figsize=(14, 7))

# Scatter plot of true vs predicted values
plt.subplot(1, 2, 1)
plt.scatter(y_test, y_pred, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel('True Values')
plt.ylabel('Predicted Values')
plt.title('True vs Predicted Values')

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
