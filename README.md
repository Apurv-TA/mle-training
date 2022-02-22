# Median housing value prediction

The housing data can be downloaded from https://raw.githubusercontent.com/ageron/handson-ml/master/. The script has codes to download the data. We have modelled the median house value on given housing data. 

The following techniques have been used: 

 - Linear regression
 - Decision Tree
 - Random Forest

## Steps performed
 - We prepare and clean the data. We check and impute for missing values.
 - Features are generated and the variables are checked for correlation.
 - Multiple sampling techinuqies are evaluated. The data set is split into train and test.
 - All the above said modelling techniques are tried and evaluated. The final metric used to evaluate is R2 error.

## To excute the script with default inputs
python < scriptname.py >

## You can also used add arguments which can be observed by using
python < scriptname.py > -h

## An example of run and the options is given below
python < scriptname.py > [-h] [--log-level LOG_LEVEL] [--log-path LOG_PATH] [--no-console-log] [--data DATA] [--save SAVE] [-v {1,2,3}]