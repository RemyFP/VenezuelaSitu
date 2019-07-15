# -*- coding: utf-8 -*-

r"""Objective functions for situational awareness and early detection.

"""

# TODO:
#   * Fix docstrings.
#   * Remove unused code.
#   * Re-factor code.

import numpy as np
import scipy
#from sklearn import cross_validation
from sklearn import model_selection
import sklearn.metrics as skmt
import situational_awareness as sa


def ED(problem, subset, **kwargs):
    r"""(Unfinished) Cross-validated early detection objective function.

    Parameters
    ----------
    problem : FS_problem
        Object on which to run optimization.
    subset : list
        List of data sources on which to evaluate objective function.

    Returns
    -------
    ave_timeliness : float
        Average timeliness of first alarms conditional on triggering alarm
        within outbreak interval.

    Other Parameters
    ----------------
    val : type

    """
    # series_list = [datum['data']['values'] for datum in subset]
    # goal_series = problem.goal['data']['values']
    #
    # return ave_timeliness


def R_squared(problem, subset, **kwargs):
    r"""Cross-validated situational awareness objective function.

    A standard goodness-of-fit measure for determining how much variance
    in the goal time-series can be explained by given data sources while
    training and testing on different subsets in time.

    Parameters
    ----------
    problem : FS_problem
        Object on which to run optimization.
    subset : list
        List of data sources on which to evaluate objective function.

    Returns
    -------
    rsquared : float
        :math:`R^2` of cross-validated predictive series generated from subset
        of data sources by linear regression. Measure is relative to goal
        time-series.

    Other Parameters
    ----------------
    var : type

    """
    pred_series_CV = sa.pred_CV(problem.goal, subset, problem.n_folds,
                                problem.bootstrap)['data']['values']
    goal_series = problem.goal['data']['values']
    numerator = scipy.stats.tvar(goal_series - pred_series_CV)
    denominator = float(scipy.stats.tvar(goal_series))
    rsquared = 1 - numerator/denominator
    return rsquared


# Deprecated objective functions

def R_squared_old(problem, subset, **kwargs):
    r"""Deprecated: Situational awareness objective function.

    Returns the coefficient of determination :math:`R^2` of linear regression
    of data sources in given subset (possibly including their derivatives).

    Parameters
    ----------
    problem : FS_problem
        Object on which to run optimization.
    subset : list
        Description
    kwargs : dict
        Must include the key 'include_derivatives' with a boolean value.

    Returns
    -------
    R_squared : float
        The coefficient of determination.

    References
    ----------
    http://en.wikipedia.org/wiki/Coefficient_of_determination

    """
    series_list = [datum['data']['values'] for datum in subset]
    if kwargs['include_derivatives']:
        series_list += [datum['data']['derivative'] for datum in subset]
    goal_series = problem.goal['data']['values']
    coefficients, predictive = problem.predictive_function(goal_series,
                                                           series_list)
    numerator = scipy.stats.tvar(goal_series - predictive)
    denominator = float(scipy.stats.tvar(goal_series))
    rsquared = 1 - numerator/denominator
    return rsquared


def R_squared_cv(problem, subset, **kwargs):
    r"""Deprecated: Returns averaged cross-validated :math:`R^2`.

    Only the first 80% of the data is used in this fit.
    Random three-fold cross-validation is performed on a linear regression
    of the data.  Average l2 norm between the predictor and the gold standard
    is then returned.

    Parameters
    ----------
    problem : type
        problem being solved
    subset : list
        data for prediction
    kwargs : dictionary
        Dictionary of parameters that must include:
            1. 'include_derivatives' given by a boolean value
            2. 'num_shuffles' given by a positive integer
            3. 'test_size' must be a float between 0 and 1

    Returns
    -------
    average : float
        l2 distance between predictor and test gold_standard data

    """
    length = len(subset[0]['data']['values'])
    cutoff = int(1.0*length)
    series_list = [datum['data']['values'] for datum in subset]
    if kwargs['include_derivatives']:
        series_list += [datum['data']['derivative'] for datum in subset]
    goal_series = problem.goal['data']['values'][:cutoff]
    # partitions = cross_validation.ShuffleSplit(cutoff,
                                               # n_iter=kwargs['num_shuffles'],
                                               # test_size=kwargs['test_size'])
    partitions = cross_validation.model_selection(n_iter=kwargs['num_shuffles'],
                                                  test_size=kwargs['test_size'])
    Rsquared_values = []
    for train_set, test_set in partitions.split(range(cutoff)):
        # Select train and test sets
        train_series_list = [[datum[i] for i in train_set] for
                             datum in series_list]
        test_series_list = [[datum[i] for i in test_set] for
                            datum in series_list]
        train_goal_series = [goal_series[i] for i in train_set]
        test_goal_series = [goal_series[i] for i in test_set]
        # Use coefficients from training set to generate predictor on test set
        coefficients, _ = problem.predictive_function(train_goal_series,
                                                      train_series_list)
        test_matrix = np.array(test_series_list).T
        test_predictive = np.dot(test_matrix, coefficients)
        # Compare predictor against goa on test set
        numerator = scipy.stats.tvar(test_goal_series - test_predictive)
        denominator = scipy.stats.tvar(test_goal_series)
        Rsquared_values.append(1-numerator/float(denominator))
    average = np.mean(Rsquared_values)
    return average


def AUC_scaled(problem, subset, **kwargs):
    r"""Deprecated: Returns an area under the curve (AUC) measure scaled to
    take values between 0 and 1.

    A goodness-of-fit measure for assessing performance in classification
    tasks, such as when using logistic regression.

    Parameters
    ----------
    problem : type
        problem being solved
    subset : list
        data for prediction
    kwargs : dictionary
        dictionary of parameters
        kwargs must have
        1. ?

    Returns
    -------
    AUC : float
        mapped to [0,1]

    """
    series_list = []
    for datum in subset:
        series_list += np.asarray(datum['data']['lagged']).T.tolist()
    goal_series = problem.goal['data']['values']

    lr_fit, predictive_class, predictive_prob = problem.predictive_function(
            goal_series, series_list)

    predictive_prob_class1 = map(lambda x: x[1], predictive_prob)
    auc = skmt.roc_auc_score(goal_series, predictive_prob_class1)

    # AUC is in [1/2,1], so scale to [0,1]
    AUC = 2*auc - 1
    return AUC


def AUC_scaled_cv(problem, subset, **kwargs):
    r"""Deprecated: Returns averaged cross-validated AUC (scaled).

    A goodness-of-fit measure for assessing performance in classification
    tasks, such as when using logistic regression.

    Parameters
    ----------
    problem : type
        problem being solved
    subset : list
        data for prediction
    kwargs : dictionary
        dictionary of parameters
        kwargs must have
        1. 'num_shuffles' given by a positive integer
        2. 'test_size' must be a float between 0 and 1

    Returns
    -------
    auc_mean : float
        AUC mapped to [0,1]

    """
    length = len(subset[0]['data']['values'])
    cutoff = int(1.0*length)
    series_list = []
    for datum in subset:
        series_list += np.asarray(datum['data']['lagged']).T.tolist()
    goal_series = problem.goal['data']['values'][:cutoff]
    # partitions = cross_validation.ShuffleSplit(cutoff,
                                               # n_iter=kwargs['num_shuffles'],
                                               # test_size=kwargs['test_size'])
    partitions = cross_validation.model_selection(n_splits=kwargs['num_shuffles'],
                                                  test_size=kwargs['test_size'])
    
    auc_values = []
    for train_set, test_set in partitions.split(range(cutoff)):
        # Select train and test sets
        train_series_list = [[datum[i] for i in train_set] for
                             datum in series_list]
        test_series_list = [[datum[i] for i in test_set] for
                            datum in series_list]
        train_goal_series = [goal_series[i] for i in train_set]
        test_goal_series = [goal_series[i] for i in test_set]
        # Use coefficients from training set to generate predictor on test set
        lr_fit, _, _ = problem.predictive_function(train_goal_series,
                                                   train_series_list)
        test_matrix = np.array(test_series_list).T
        test_predictive_prob_class1 = map(lambda x: x[1],
                                          lr_fit.predict_proba(test_matrix))
        # Compare predictor against goa on test set
        auc = skmt.roc_auc_score(test_goal_series, test_predictive_prob_class1)
        # AUC is in [1/2,1], so scale to [0,1]
        auc_values.append(2*auc - 1)
    auc_mean = np.mean(auc_values)
    return auc_mean
