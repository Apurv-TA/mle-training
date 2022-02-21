# -*- coding: utf-8 -*-
"""utils.py docstring

This module contain the code for creating Custom transformer class
'CombinedAttributesAdder'


Attributes
----------
rooms_ix: int
    The index number of 'rooms' feature in the housing DataFrame.
bedrooms_ix: int
    The index number of 'bedrooms' feature in the housing DataFrame.
population_ix: int
    The index number of 'population' feature in the housing DataFrame.
households_ix: int
    The index number of 'households' feature in the housing DataFrame.
"""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6


class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    """Class to add some extra features to the dataframe

    The features created are 'rooms_per_household' and
    'population_per_household'

    Attributes
    ----------
    add_bedrooms_per_room : bool, optional
        Public attribute to decide whether to create a feature named
        'bedrooms_per_room' or not
    """

    def __init__(self, add_bedrooms_per_room=True):
        self.add_bedrooms_per_room = add_bedrooms_per_room

    def fit(self, X, y=None):
        """Method to fit the dataframe

        Parameters
        ----------
        X: pandas Dataframe
            Dataframe on which the operation is performed
        y: pd.Dataframe
        """

        return self

    def transform(self, X):
        """Method to transform the dataframe

        This method is used to transform the dataframe and add extra
        features to it. The features added are: 'rooms_per_household',
        'population_per_household' and 'bedrooms_per_room'

        Parameters
        ----------
        X: pd.DataFrame
            The dataframe on which the tranform operation will be performed

        Returns
        --------
        Updated dataframe
        """

        rooms_per_household = X[:, rooms_ix] / X[:, households_ix]
        population_per_household = X[:, population_ix] / X[:, households_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[
                X,
                rooms_per_household,
                population_per_household,
                bedrooms_per_room
            ]
        return np.c_[X, rooms_per_household, population_per_household]
