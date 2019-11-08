# import packages for data manipulation
import pandas as pd
import numpy as np

# Plotting libraries
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import janitor
import pickle
from itertools import product
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.model_selection import GroupKFold, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_squared_log_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings('ignore')

import daiquiri,logging
daiquiri.setup(level=logging.INFO)
logger = daiquiri.getLogger()

import argparse

parser = argparse.ArgumentParser(description="Type of avocado to run model for")
parser.add_argument("avocado_type", type=str, help="Enter the avocado type: conventional or organic")
parser.add_argument("region", type=str, help="Enter the region you need prediction plotting for")
args = parser.parse_args()
avocado_type = args.avocado_type

#Loading data
logger.info("Reading the data file...")
df = pd.read_csv('../data/avocado.csv')

logger.info("Quick preprocessing of data...")
# Removing index column
df.drop('Unnamed: 0', axis=1, inplace=True)
# Removing records with TotalUS region, assuming it is nust the average of all other regions
df = df.loc[df.region!='TotalUS'].reset_index(drop=True)
# Making date to datetime and sorting chrinologically
df['Date'] = pd.to_datetime(df['Date'])
df = df.clean_names()

logger.info("Making Future DataFrame...")
future_dates = ['2018-04-01','2018-04-08','2018-04-15','2018-04-22']
regions = list(set(df.region))
types = list(set(df.type))
future_df = pd.DataFrame(list(product(future_dates, regions, types)), columns=['date', 'region', 'type'])
future_df.date = pd.to_datetime(future_df.date)
df = df.append(future_df)

# Dataset creation
logger.info("Create one month lag DataFrame")
df_list = []
for avocado_type in types:
    for region in regions:
        #print(region)
        temp = df.loc[(df.region==region)&(df.type==avocado_type)].sort_values('date').reset_index(drop=True)
        for col in ['total_volume','4046','4225','4770','total_bags','small_bags','large_bags','xlarge_bags','averageprice']:
            temp[f'one_month_lag_{col}'] = temp[col].shift(5)
            if col!='averageprice':
                temp.drop(col,axis=1,inplace=True)
        temp = temp.loc[temp.one_month_lag_total_volume.notnull()].reset_index(drop=True)
        #print(temp.shape)
        df_list.append(temp)
final_train = pd.concat(df_list)

logger.info('Saving dataframe for later use...')
pickle.dump(final_train,open('../data/data_with_one_month_lag.p','wb'))

logger.info("Filter dataframe for the desired avocado type...")
final_train = final_train.loc[final_train.type==avocado_type]
final_train = final_train.sort_values(['date']).reset_index(drop=True)

# Adding month and daya variable for capturing seasonality
logger.info('Adding time realted features....')
final_train['month']=final_train['date'].apply(lambda x:x.month)
final_train['day']=final_train['date'].apply(lambda x:x.day)

for lag in range(1,4):
    final_train[f'one_month_lag_lag_{lag}'] = final_train.groupby(['region','type'])['one_month_lag_averageprice'].shift(lag)
    
final_train['long_term_moving_average'] = final_train.groupby(['region','type'])['one_month_lag_averageprice'].transform(lambda x: x.rolling(window=52,min_periods=1).mean())
final_train['short_term_moving_average'] = final_train.groupby(['region','type'])['one_month_lag_averageprice'].transform(lambda x: x.rolling(window=12,min_periods=1).mean())
final_train['is_SMA_greater'] = (final_train['short_term_moving_average'] > final_train['long_term_moving_average'])


# Splitting data into train and test followed by preprocessing
logger.info("Splitting data into train and test...")
test = final_train.loc[(final_train.date>='2018-01-01')&(final_train.date<'2018-04-01')].reset_index(drop=True)
train = final_train.loc[final_train.date<'2018-01-01'].reset_index(drop=True)
future = final_train.loc[final_train.date>='2018-04-01'].reset_index(drop=True)


logger.info("Feature Processing...")

def preprocessing(df, train=True):
    
    num_columns = ['one_month_lag_total_volume','one_month_lag_4046','one_month_lag_4225',
                   'one_month_lag_4770','one_month_lag_total_bags','one_month_lag_small_bags', 
                   'one_month_lag_large_bags', 'one_month_lag_xlarge_bags']

    if train:
        sc = StandardScaler()
        scaled_columns = sc.fit_transform(df[num_columns])
        scaled_df = pd.DataFrame(scaled_columns)
        scaled_df.columns = num_columns
        df = df.drop(num_columns,axis=1).join(scaled_df)
        pickle.dump(sc,open('../data/one_month_lag_scaler.p','wb'))
        
        le = LabelEncoder()
        label_encoding = le.fit_transform(df['region'])
        df['region'] = label_encoding
        pickle.dump(le,open('../data/region_label_encoding.p','wb'))
        
    else:
        sc = pickle.load(open('../data/one_month_lag_scaler.p','rb'))
        scaled_columns = sc.transform(df[num_columns])
        scaled_df = pd.DataFrame(scaled_columns)
        scaled_df.columns = num_columns
        df = df.drop(num_columns,axis=1).join(scaled_df)
        
        le = pickle.load(open('../data/region_label_encoding.p','rb'))
        df['region'] = le.transform(df['region'])
        
    
    return df
logger.info("Preparing for modelling...")
train = preprocessing(train,train=True)
test = preprocessing(test,train=False)
future = preprocessing(future,train=False)

y = train['averageprice']
date = train['date']
X = train.drop(['date','averageprice','type'],axis=1)

# Modeling
logger.info("Defining the XGBoost model...")
params = {
			'learning_rate':0.05, 
			'n_estimators': 200,
			'colsample_bytree': 0.8, 
			'gamma': 0.3, 'max_depth': 7, 
			'min_child_weight': 4, 
			'subsample': 0.6
			}
model = XGBRegressor(**params)

logger.info('Fitting model...')
model.fit(X,y)
pickle.dump(model,open(f'../models/xgb_model_one_month_lag_{avocado_type}.p','wb'))

# Prediction
logger.info('Making predictions for Denver region...')
preds = model.predict(test[X.columns])
logger.info('Saving the plot actual vs prediction...')
test['predicted_price'] = preds
test['error'] = test.averageprice - test.predicted_price
mape = np.round(np.mean(np.abs(100*test.error/test.averageprice)), 2) 
rmse = np.round(np.sqrt(mean_squared_error(test.averageprice,test.predicted_price)),2)
r2 = np.round(r2_score(test.averageprice,test.predicted_price),2)

logger.info(f'Actual vs Predicted Avocado price for Denver \n RMSE: {rmse} \n MAPE: {mape}% \n R2: {r2}')

logger.info('Plotting the result on holdout set...')

# set the region you need ploting for
region=args.region

le = pickle.load(open('../data/region_label_encoding.p','rb'))
train.region = le.inverse_transform(train.region)
test.region = le.inverse_transform(test.region)
# Getting the time series from train and test dataframes
train_dates = train.loc[(train.region==region), 'date']
test_dates = test.loc[(test.region==region), 'date']
train_values = train.loc[(train.region==region), 'averageprice']
test_values = test.loc[(test.region==region), 'averageprice']
test_predictions = test.loc[(test.region==region), 'predicted_price']

# Plotting the predictions
fig, ax = plt.subplots(1, 1, figsize=(15, 8));
ax.plot(train_dates,train_values, color='blue', label='Training Data');
ax.plot(test_dates, test_predictions, color='green', marker='o',label='Predicted Price');

ax.plot(test_dates, test_values, color='red', label='Actual Price');
ax.set_title(f'{region} region -  Avocado Prices Prediction - {avocado_type} \nRMSE: {rmse}');
ax.set_xlabel('Dates');
ax.set_ylabel('Prices');
ax.legend();
plt.savefig(f'../plots/{region}_Price_prediction_one_month_lag_{avocado_type}.png')

logger.info('Future predictions...')
future_preds = model.predict(future[X.columns])
future['predicted_price'] = future_preds

logger.info('Saving future predictions...')
future.loc[:, ['date','region','type','predicted_price']].to_csv(f'../results/next_month_predictions_{avocado_type}.csv', index=False)