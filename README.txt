INTRODUCTION
------------

This repository is for analyzing and forecasting avocado prices in US region.
Python 3.7.3 was used
requirements.txt file contains all the packages used in the analysis. One can run `pip install -r requirements.txt` to install all of them


FOLDERS
-------

src: Contains two python files
	
	model_interpolation.py: (see notebooks->  Predicting Denver prices (interpolation) )

		Details
		* Trains a XGBoost model based on all data and predicts prices for Denver region
		* Cannot be used for future prediction
		* Cross validation, tuning and feature importances are in the notebook

		Output
		* model file
		* Chart showing the actual price vs predicted price for Denver region with evaluation metrics in the title.
		* csv file containing the date,type,region,actual price and predicted price

	model_one_month_lag.py: (see notebooks ->  one month lag)

		Details

		* Assume your avocado data collection is on a 1 month delay. When predicting the price of an avocado at time 			x, you can only use historical data from up until 1 month prior to x)
		* Trains a XGBoost model based on all data but features used are based on one month prior at any given point 			of time. 
		* Needs user input for which 'avocado type' to run the model for and 'region' for which the prediction plot 			to be made
		* Can be used for future prediction. default is for next month
		* Cross validation, tuning and feature importances are in the notebook

		Output
		* model file
		* Chart showing the actual price vs predicted price for chosen region with evaluation metrics in the title.
		* csv file containing the date,type,region and predicted price for the next month (April 2018) for all 			  regions

notebooks: contains 6 interactive notebooks
	
	 interactive plot.ipynb

		Details

		* Plots an interactive time series plot for desired region,type and year
		* Plots an interactive year-over-year plot for desired region and type

	 Exploratory Data Analysis.ipynb

		Details

		* Exploratory analysis done on the given data and summarizes the finding that is being used in the modeling 			procedures

	 Single Time Series.ipynb

		Details

		* Experimentation based on prophet and ARIMA models for per region.
		* Summary and findings included in the notebooks

	 Predicting Denver prices (interpolation) and  one month lag.ipynb

		Details

		* Counterpart notebooks for python scripts
		* Contains grid search, cross validation and feature importance analysis 

	 tsfresh features.ipynb

		Details

		* Experimentations with tsfresh package to generate extensive time series features
		* Not currently used in the models
		* Outputs a pickle file that can be merged with given data using region,type,date as keys

data - contains raw and intermediary data files needed for analysis and modeling
results - contains results files out of modeling
plots - contains plots from the analysis
models - model files
html - All notebooks have been run and converted into html

EXECUTION
---------

cd to src folder and run the following command

1.

python model_interpolation.py

eg output:
2019-11-07 19:13:09,572 [93120] INFO     root: Reading the data file...
2019-11-07 19:13:09,620 [93120] INFO     root: Quick preprocessing of data...
2019-11-07 19:13:09,654 [93120] INFO     root: Adding additional varaibles...
2019-11-07 19:13:10,345 [93120] INFO     root: Splitting data into train and test...
2019-11-07 19:13:10,375 [93120] INFO     root: Feature Processing...
2019-11-07 19:13:10,375 [93120] INFO     root: Preparing for modelling...
2019-11-07 19:13:10,869 [93120] INFO     root: Defining the XGBoost model...
2019-11-07 19:13:10,870 [93120] INFO     root: Fitting model...

2.

python model_one_month_lag.py conventional Denver

eg output:

2019-11-07 19:16:07,924 [93219] INFO     root: Reading the data file...
2019-11-07 19:16:07,954 [93219] INFO     root: Quick preprocessing of data...
2019-11-07 19:16:07,970 [93219] INFO     root: Making Future DataFrame...
2019-11-07 19:16:07,979 [93219] INFO     root: Create one month lag DataFrame
2019-11-07 19:16:09,662 [93219] INFO     root: Saving dataframe for later use...
2019-11-07 19:16:09,667 [93219] INFO     root: Filter dataframe for the desired avocado type...
2019-11-07 19:16:09,672 [93219] INFO     root: Adding time realted features....
2019-11-07 19:16:09,798 [93219] INFO     root: Splitting data into train and test...
2019-11-07 19:16:09,804 [93219] INFO     root: Feature Processing...
2019-11-07 19:16:09,804 [93219] INFO     root: Preparing for modelling...
2019-11-07 19:16:10,215 [93219] INFO     root: Defining the XGBoost model...
2019-11-07 19:16:10,215 [93219] INFO     root: Fitting model...
2019-11-07 19:16:12,510 [93219] INFO     root: Making predictions for Denver region...
2019-11-07 19:16:12,516 [93219] INFO     root: Saving the plot actual vs prediction...
2019-11-07 19:16:12,524 [93219] INFO     root: Actual vs Predicted Avocado price for Denver 


