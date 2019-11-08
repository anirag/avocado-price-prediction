# import packages for data manipulation
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import janitor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error, mean_squared_log_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor

#logging package
import daiquiri,logging
daiquiri.setup(level=logging.INFO)
logger = daiquiri.getLogger()


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
df = df.sort_values(['region','Date'])
df = df.clean_names()

# Adding month and daya variable for capturing seasonality
logger.info("Adding additional varaibles...")
df['month']=df['date'].apply(lambda x:x.month)
df['day']=df['date'].apply(lambda x:x.day)
for lag in range(1,11):
	df[f'lag_{lag}'] = df.groupby(['region','type'])['averageprice'].shift(lag)
df['long_term_moving_average'] = df.groupby(['region','type'])['averageprice'].transform(lambda x: x.rolling(window=52,min_periods=1).mean())
df['short_term_moving_average'] = df.groupby(['region','type'])['averageprice'].transform(lambda x: x.rolling(window=12,min_periods=1).mean())
df['is_SMA_greater'] = (df['short_term_moving_average'] > df['long_term_moving_average'])

# Splitting data into train and test followed by preprocessing
logger.info("Splitting data into train and test...")
test = df.loc[df.region=='Denver'].reset_index(drop=True)
train = df.loc[df.region!='Denver'].reset_index(drop=True)

logger.info("Feature Processing...")

def preprocessing(df, train=True):
    
    df = pd.get_dummies(df, columns = ['type'], prefix=['type'])
    num_columns = ['total_volume','4046','4225','4770','total_bags','small_bags', 'large_bags', 'xlarge_bags']

    if train:
        sc = StandardScaler()
        scaled_columns = sc.fit_transform(df[num_columns])
        scaled_df = pd.DataFrame(scaled_columns)
        scaled_df.columns = num_columns
        df = df.drop(num_columns,axis=1).join(scaled_df)
        pickle.dump(sc,open('../data/scaler.p','wb'))
        
    else:
        sc = pickle.load(open('../data/scaler.p','rb'))
        scaled_columns = sc.transform(df[num_columns])
        scaled_df = pd.DataFrame(scaled_columns)
        scaled_df.columns = num_columns
        df = df.drop(num_columns,axis=1).join(scaled_df)
    
    return df

# Modeling
logger.info("Preparing for modelling...")
train = preprocessing(train,train=True)
test = preprocessing(test,train=False)
y = train['averageprice']
date = train['date']
regions = train['region']
X = train.drop(['date','averageprice','region'],axis=1)

logger.info("Defining the XGBoost model...")
params = {
			'learning_rate':0.02, 
			'n_estimators': 650,
			'colsample_bytree': 0.8, 
			'gamma': 0.3, 'max_depth': 7, 
			'min_child_weight': 4, 
			'subsample': 0.6
			}
model = XGBRegressor(**params)

logger.info('Fitting model...')
model.fit(X,y)
pickle.dump(model,open(f'../models/xgb_model_interpolation.p','wb'))

# Prediction
logger.info('Making predictions for Denver region...')
preds = model.predict(test[X.columns])

logger.info('Saving the plot actual vs prediction...')
test['predicted_price'] = preds
test['error'] = test.averageprice - test.predicted_price
mape = np.round(np.mean(np.abs(100*test.error/test.averageprice)), 2) 
rmse = np.round(np.sqrt(mean_squared_error(test.averageprice,test.predicted_price)),2)
r2 = np.round(r2_score(test.averageprice,test.predicted_price),2)


# Plotting
logger.info(f'Actual vs Predicted Avocado price for Denver \n RMSE: {rmse} \n MAPE: {mape}% \n R2: {r2}')
plt.figure(figsize=(10,5))
sns.lineplot(x='date', y='value', hue='variable', data=pd.melt(test[['date','averageprice','predicted_price']], ['date']), err_style=None)
plt.title(f'Actual vs Predicted Avocado price for Denver \n RMSE: {rmse} \n MAPE: {mape}% \n R2: {r2}')
plt.tight_layout()
plt.savefig('../plots/Denver_Price_prediction.png')

logger.info("Saving the results as csv file")
test[['date','region','averageprice','predicted_price','error']].to_csv('../results/denver_price_prediction.csv',index=False)