"""Module created to train the model"""
# IMPORTING LIBRARIES
import argparse
import os
import warnings

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--data", nargs= "?" , const= "../../data/processed/train.csv")
parser.add_argument("-s", "--save", nargs= "?", const= "../../artifacts")

data_path = parser.parse_args().data
data_sav = parser.parse_args().save

# MODELLING

# BASIC MODEL
rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room=True):  # no *args or **kargs
        self.add_bedrooms_per_room = add_bedrooms_per_room
        
    def fit(self, X, y=None):
        return self  # nothing else to do
        
    def transform(self, X):
        rooms_per_household = X[:, rooms_ix] / X[:, households_ix]
        population_per_household = X[:, population_ix] / X[:, households_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]

def eval_matrics(model, train_x, train_y):
    # function to get r2 score using cross_val_score
    scores = cross_val_score(model, train_x, train_y, scoring="r2", cv=10)
    return scores.mean()

def basic_modeling(train):
    housing = train.drop("median_house_value", axis=1)
    housing_labels = train["median_house_value"].copy()

    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="median")), 
        ('attribs_adder', CombinedAttributesAdder()), 
        ('std_scaler', StandardScaler()), 
        ])

    full_pipeline = ColumnTransformer([
        ("num", num_pipeline, list(housing.drop("ocean_proximity", axis=1))), 
        ("cat", OneHotEncoder(), ["ocean_proximity"]),
        ])

    housing_prepared = full_pipeline.fit_transform(housing)

    models = {
        "Linear_reg": LinearRegression(), 
        "Decision_tree": DecisionTreeRegressor(), 
        "Random_forest": RandomForestRegressor()
        }

    eval_dict = {}
    for model in models:
        eval_dict[model] = eval_matrics(models[model], housing_prepared, housing_labels)
    print(eval_dict)
    print(max(eval_dict))

#    for model in models:
 #       print(f"{model}_R2_Score", eval_matrics(models[model], housing_prepared, housing_labels))    
    return housing_prepared, housing_labels, full_pipeline


# FINE TUNING
def model_search(housing_prepared, housing_labels):
    params_grid = [
        {"n_estimators": [3, 10, 30, 100, 300], "max_features": [2, 4, 6, 8, 10]},
        {"bootstrap": [0], "n_estimators": [3, 10, 30, 100], "max_features": [2, 3, 4, 6]}
        ]

    forest_reg = RandomForestRegressor(random_state=42)
    grid_search = GridSearchCV(
        forest_reg, params_grid, cv=2, scoring="r2", return_train_score=True, verbose=3
    )

    grid_search.fit(housing_prepared, housing_labels)

    print(grid_search.best_estimator_)
    print(grid_search.best_params_)
    print(grid_search.best_score_)

    return grid_search.best_estimator_, grid_search.best_score_



if __name__ == "__main__":
    train = pd.read_csv(data_path)
    housing_prepared, housing_labels, full_pipeline = basic_modeling(train)
#    model_search(housing_prepared= housing_prepared, 
 #   housing_labels= housing_labels)
