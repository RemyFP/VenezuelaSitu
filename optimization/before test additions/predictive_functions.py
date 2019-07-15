# -*- coding: utf-8 -*-

r"""
High level description of the module.
"""

# TODO:
#   * Fix docstrings.
#   * Re-factor to remove repetition.

import numpy as np
import sklearn.linear_model as sklm


def lin_reg(goal_series, series_collection):
    r"""Returns a predictive function using least_squares

    Returns Ax* where x* = argmin_{x}( || Ax - b ||_2 )
    The matrix, A, is obtained by putting the lists from series_collection
    into a column of A.  The vector b is simply goal_series

    Parameters
    ----------
    goal_series : list
        the vector b
    series_collection : list
        list of column vectors of A

    Returns
    -------
    output : list
        least squares coefficients and a predictive time series

    """
    data_matrix = np.array(series_collection).T
    coefficients = np.linalg.lstsq(data_matrix, goal_series)[0]
    output = coefficients, np.dot(data_matrix, coefficients)
    return output


def logistic_reg(goal_series, series_collection):
    r"""Returns a logistic regression fit

    Parameters
    ----------
    goal_series : list
        the vector b
    series_collection : list
        list of column vectors of A

    Returns
    -------
    output : tuple
        coefficients of logistic regression

    """
    data_matrix = np.array(series_collection).T
    lr_fit = sklm.LogisticRegression().fit(data_matrix, goal_series)
    output = (lr_fit, lr_fit.predict(data_matrix),
              lr_fit.predict_proba(data_matrix))
    return output
