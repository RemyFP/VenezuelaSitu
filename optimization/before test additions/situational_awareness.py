# -*- coding: utf-8 -*-

r"""Functions for situational awareness.



"""

import scipy
import numpy as np
# from sklearn.cross_validation import KFold
from sklearn.model_selection import KFold
from functools import partial
import data_manipulation as dm
import train_test as tt


def lin_reg(goal_datum, data_sources):
    r"""Determine the coefficients of linear regression using training data.

    Parameters
    ----------
    goal_datum : dictionary
        The gold standard data source, which is the dependent variable in the
        linear regression.
    data_sources : list
        List of data sources that will be evaluated. Each data source is a
        dictionary, which is an independent variable in the linear
        regression model.

    Returns
    -------
    coefficients : list
        Coefficients of the linear regression model.

    """
    goal_series = goal_datum['data']['values']
    series_list = [d['data']['values'] for d in data_sources]
    series_matrix = np.array(series_list).T #transpose
    coefficients = np.linalg.lstsq(series_matrix, goal_series)[0]
    return coefficients


def lin_pred(data_sources, coefficients):
    r"""Predict dependent variable values based on independent variables and
    coefficients.

    Parameters
    ----------
    data_sources : list
        List of data sources that will be evaluated. Each data source is a
        dictionary, which is an independent variable in the linear
        regression model.
    coefficients : list
        Coefficients of the linear regression model.

    Returns
    -------
    pred_series : list
        Gold standard values predicted from independent variables (data
        sources).

    """
    series_list = [d['data']['values'] for d in data_sources]
    series_matrix = np.array(series_list).T
    pred_series = np.dot(series_matrix, coefficients)
    return pred_series


def pred_CV(goal_datum, data_sources, n_folds=1, bootstrap=False):
    r"""Test the performance of the linear regression model using
    cross-validation.

    Parameters
    ----------
    goal_datum : dictionary
        The gold standard data source, which is the dependent variable in the
        linear regression.
    data_sources : list
        List of data sources that will be evaluated. Each data source is a
        dictionary, which is an independent variable in the linear
        regression model.
    n_folds : integer
        Number of fold in k-fold cross-validation
    bootstrap : boolean
        Whether or not to use bootstrap data sets, which are created by sampling
        with replacement and the same size as the original training dataset.

    Returns
    -------
    pred_series_CV : dictionary
        Gold standard values predicted from independent variables (data
        sources).

    """
    if n_folds > 1:
        # kf = KFold(dm.length(goal_datum), n_folds)
        kf_p = KFold(n_folds)
        kf = list(kf_p.split(range(dm.length(goal_datum))))
    else:
        v = range(dm.length(goal_datum))
        kf = [(v,v)]
    pred_series_CV_list = []
    for train, test in kf:
        if bootstrap:
            train = np.random.choice(train, len(train))
        train_f, test_f = tt.index_to_filter(train), tt.index_to_filter(test)
        coef = tt.train_on_filter(
                lin_reg, goal_datum, data_sources, train_f)
        pred_series = tt.test_on_filter(
                partial(lin_pred, coefficients=coef), data_sources, test_f)
        pred_series_CV_list.append(pred_series)
    pred_series_CV = {}
    pred_series_CV['data'] = {}
    pred_series_CV['data']['values'] = np.concatenate(
            pred_series_CV_list, axis=0)
    pred_series_CV['data']['times'] = goal_datum['data']['times']
    return pred_series_CV


def R_squared(problem, subset):
    r"""Situational awareness objective function.

    A standard goodness-of-fit measure for determining how much variance
    in the gold standard data source can be explained by given data sources.

    Parameters
    ----------
    problem : surveillance optimization problem
        See Class `surveillance_optimization`.
    subset : list
        List of data sources on which to evaluate objective function. Each data
        source is a dictionary.

    Returns
    -------
    rsquared : float
        :math:`R^2` of cross-validated predictive series generated from subset
        of data sources by linear regression. Measure is relative to the gold
        standard data source.

    """
    pred_series_CV = pred_CV(problem.gold_standard, subset, problem.n_folds,
                                problem.bootstrap)['data']['values']
    goal_series = problem.gold_standard['data']['values']
    numerator = scipy.stats.tvar(goal_series - pred_series_CV)
    denominator = float(scipy.stats.tvar(goal_series))
    rsquared = 1 - numerator/denominator
    return rsquared
