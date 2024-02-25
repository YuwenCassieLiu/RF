# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV 
from sklearn.decomposition import PCA

stock_df = pd.read_csv('data base.csv')
stock_df['date'] = pd.to_datetime(stock_df['date'])

column_name  = stock_df.columns[1:].to_list()

char_list = list(set([col.split("__")[0] for col in column_name]))
char_list.remove('TICKER')
char_list.remove('RET')

ticker_list = list(set([col.split("__")[1] for col in column_name]))

data_base = stock_df
for name in column_name:
    data_base[name].replace(float(0), method='ffill', inplace=True)
    
exclude_strings = ['TICKER', 'RET', 'date']

# Apply the lag operation to columns not containing the exclude strings
for col in data_base.columns:
    if not any(excl_str in col for excl_str in exclude_strings):
        data_base[col] = data_base[col].shift(1)

data_base.dropna(inplace=True)

time_range = data_base['date']

def Rf_prediction(train_data, test_data, stock, n_components = 10):
    # Prepare the (x_train, y_train) and x_test
    y_train = train_data[f"RET__{stock}"]
    x_train = train_data.drop(f"RET__{stock}", axis = 1)
    x_test = test_data.drop(f"RET__{stock}", axis = 1)
    
    # Standardize the features
    scaler = StandardScaler()

    # Fit on training set only
    scaler.fit(x_train)
    
    # Apply transform to both the training set and the test set
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)

    # Make an instance of the PCA, if n_components is not set, it will keep all components
    pca = PCA(n_components = n_components)
    pca.fit(x_train)

    # Apply PCA transform to both the training and test set
    x_train = pca.transform(x_train)
    x_test = pca.transform(x_test)

    # Set ranges of model parameters for Cross-validation
    n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1000, num = 10)] 
    max_features = [0.2, 0.4, 0.6, 0.8]
    max_depth = [int(x) for x in np.linspace(1, 10, num = 10)] 
    max_depth.append(None)
    min_samples_split = [2, 5, 10]
    min_samples_leaf = [1, 2, 4]
    
    param_grid = {
        'n_estimators': n_estimators,
        'max_features': max_features,
        'max_depth': max_depth,
        'min_samples_split': min_samples_split,
        'min_samples_leaf': min_samples_leaf
    }

    # Define Random Forest model
    rf = RandomForestRegressor(random_state=610)
    # TimeSeriesSplit
    tscv = TimeSeriesSplit(n_splits = 5)
    print(tscv)
    cv_rf = RandomizedSearchCV(estimator = rf, param_distributions = param_grid, 
                       cv = tscv, scoring='neg_mean_squared_error', random_state = 610, n_jobs = -1)

    # Fit the random forest model with cross-validation
    cv_rf.fit(x_train, y_train)

    # Best model
    # best_parameters = cv_rf.best_params_
    best_rf = cv_rf.best_estimator_
    
    # Predict
    y_pred = best_rf.predict(x_test)

    # Return the one-step forecast adjusted stock return
    return (y_pred[0])


# Define a list to record portfolio return and a dictionary to record OOS residuals of the model
portfolio_returns = []
residuals_dict = {}
for stock in ticker_list[0:100]:
    residuals_dict[stock] = []
Dates = []

# Define the numer of forecast (for example, one year)
for month in range(121,134):
    # Seperate the train and test data set
    train_data = data_base[(month-121):month]
    test_data = data_base[month:month+1]
    Dates.append(test_data.iloc[0,0])

    # Define two lists to predicted and realised return of each stock on the same month
    predictions = []
    realized = []

    # Define the number of stocks that we taken into consideration
    for stock in ticker_list[0:100]:
        # print(stock)
        # For each stock, find the columns in df related to this stock in both train and test data set
        updated_char_list = [f"{col}__{stock}" for col in char_list]
        updated_list = [f"RET__{stock}"] + updated_char_list
        stock_train_data = train_data[updated_list]
        stock_test_data = test_data[updated_list]
        
        # Call the Rf_prediction() function to get the one-step forward predicted return for this stock and record it
        stock_pred = Rf_prediction(train_data = stock_train_data, test_data = stock_test_data, stock = stock)
        predictions.append(stock_pred)

        # Rcord the realised return of this stock, compute 
        # and record the OOS residual of this stock into a dictionary with key being the ticker
        y_test = float(stock_test_data[f"RET__{stock}"])
        realized.append(y_test)
        residuals_dict[stock].append(y_test - stock_pred)
                                           
    # Rank stocks based on predicted returns
    ranked_stocks = pd.Series(predictions).rank()

    # Select top 5 and bottom 5 (index)
    top_5 = ranked_stocks.nlargest(5).index
    bottom_5 = ranked_stocks.nsmallest(5).index

    # Calculate portfolio return as the average of selected stocks' actual returns next month
    top_5_realised = [realized[i] for i in top_5]
    bottom_5_realised = [realized[i] for i in bottom_5]
    next_day_returns = sum(top_5_realised)/len(top_5_realised) - sum(bottom_5_realised)/len(bottom_5_realised)
    portfolio_returns.append(next_day_returns)

sum_mse = 0
for stock in ticker_list[0:10]:
    residuals = np.array(residuals_dict[stock])
    mse = np.mean(residuals**2)
    sum_mse += mse
    print(mse)
average_mse = sum_mse/10
print("Mse:", average_mse)

print("portfolio_returns:", portfolio_returns)
print("average return:", np.mean(portfolio_returns)*12)
print("vol:", np.std(portfolio_returns)*np.sqrt(12))

Interest_rate_df = pd.read_csv("Interest rate.csv")
Interest_rate_df['date'] = pd.to_datetime(Interest_rate_df['date'], dayfirst=True)
Interest_rate_df = pd.DataFrame(Interest_rate_df.drop("Unnamed: 0",axis = 1))
Interest_rate_df = Interest_rate_df[Interest_rate_df['date'].isin(Dates)]
Interest_rate_df = Interest_rate_df.reset_index().drop("index", axis=1)
diff = []
for i in range(0,len(Interest_rate_df)-1):
    diff.append(portfolio_returns[i] - Interest_rate_df.value[i]/100)

SR = np.mean(diff)/np.std(diff) * np.sqrt(12)
print("SR:", SR)

returns_series = pd.Series(portfolio_returns)
cumulative_return = (1 + returns_series).cumprod()
plt.figure(figsize=(10, 6))
cumulative_return.plot()
plt.title('Cumulative Return Over Time (Random Forest)')
plt.xlabel('Time Period')
plt.ylabel('Cumulative Return')
plt.grid(True)

# Save the figure to a file
plt.savefig('/mnt/data/cumulative_return_plot610.png')
plt.close()  # Close the figure window


