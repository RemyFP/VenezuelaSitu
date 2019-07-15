# -*- coding: utf-8 -*-

r"""
High level description of the module.
"""

# TODO:
#   * Fix docstrings.
#   * Re-factor to remove repetition.

import copy
import math
import numpy
import numpy.random
from sklearn.linear_model import (Ridge, RidgeCV, Lasso, LassoCV, ElasticNet,
                                  ElasticNetCV, OrthogonalMatchingPursuit,
                                  OrthogonalMatchingPursuitCV)


def forward_selection(problem, **kwargs):
    #TODO:
    # - when connect with front end, need to make sure that req_data will not
    #   be included in data.
    r"""Forward selection for specified objective function

    Parameters
    ----------
    problem : type
        Surveillance objective problem to be solved.
    kwargs : dictionary
        ['choose'] must be a positive integer.

    Returns
    -------
    output : tuple
        (optimal subset, optimal objective value, objective trace, objective
        value for each single datum)
        (list of data dictionaries, float, list)

    References
    ----------
    "Optimizing Provider Recruitment for Influenza Surveillance Networks"

    """
    data = copy.copy(problem.data)
    req_data = copy.copy(problem.req_data)
    choose = kwargs['choose'] - len(req_data)
    threshold_optim = kwargs['threshold_optim']
    optimum = [] + req_data
    optimum_OOS_R_sq = []
    # initialize objective_trace list based on if req_data list is blank or not.
    if (len(req_data) != 0):
        objective_trace = [problem.objective_function(req_data, **kwargs)]
    else:
        objective_trace = [0]
    for i in range(choose):
        objective_values = []
        for datum in data:
            temp_optimum = optimum + [datum]
            objective_values.append(problem.objective_function(temp_optimum, **kwargs))
        if i == 0:
            objective_values_single_datum = objective_values
        argmax = numpy.argmax(objective_values)
        interim_optimum = data.pop(argmax)
        
        # Continue only if addition of data source increases the R squared
        # as much as the chosen threshold (if given as input)
        if (threshold_optim is not None) and (i>0) and (max(objective_values)>0.15):
            if max(objective_values) < objective_trace[-1] + threshold_optim:
                break
        
        optimum = optimum + [interim_optimum]
        objective_trace.append(max(objective_values))
        
        # Test set of optimum series out of sample
        #print(problem.test_OOS(optimum))
        optimum_OOS_R_sq.append(problem.test_OOS(optimum))
        

        
    output = (optimum, objective_trace[-1], objective_trace,
            objective_values_single_datum,optimum_OOS_R_sq)
    return output


def backward_selection(problem, **kwargs):
    # TODO:
    # - how to show results on lower right graph? The current solution is to
    # rank datums based on objective values calculated after the datum is
    # removed. The graph will show the top N data sources (N is the size of
    # optimized size).
    r"""Backward selection for specified objective function

    Parameters
    ----------
    problem : type
        Surveillance objective problem to be solved.
    kwargs : dictionary
        ['choose'] must be a positive integer.

    Returns
    -------
    output : tuple
        (optimal subset, optimal objective value, objective trace, objective
        value for each single datum)
        (list of data dictionaries, float, list)

    """
    data = copy.copy(problem.data)  # do not include req_data
    req_data = copy.copy(problem.req_data)
    choose = kwargs['choose'] - len(req_data)
    optimum = []
    objective_trace = [problem.objective_function(data+req_data, **kwargs)]
    while len(data) > 1:
        objective_values = []
        for index, datum in enumerate(data):
            #  temp_optimum = copy.deepcopy(optimum)
            temp_optimum = copy.copy(data)
            temp_optimum.pop(index)
            objective_values.append(problem.objective_function(\
                temp_optimum+req_data, **kwargs))
        argmax = numpy.argmax(objective_values)
        optimum = optimum + [data.pop(argmax)]
        objective_trace.append(max(objective_values))
    if (len(req_data) != 0):
        objective_trace.append(problem.objective_function(req_data, **kwargs))
    else:
        objective_trace.append(0)
    optimum.reverse()
    optimum = req_data + optimum
    objective_trace.reverse()
    # compute objective value for each datum
    objective_values_single_datum = []
    for datum in problem.data:
        objective_values_single_datum.append(problem.objective_function([datum],
            **kwargs))
    output = (optimum[ :kwargs['choose']], objective_trace[-1], objective_trace[
        :choose+1], objective_values_single_datum)
    return output


def random_selection(problem, **kwargs):
    # TODO:
    # - Same questions/issues as in backward selection. Can this be used to
    #   a ranked list?
    r"""Random selection for specified objective function

    Choose enough random subsets until a subset is in the
    problem.upper_percentile percent of solutions with probability
    problem.rand_selection_confidence

    Parameters
    ----------
    problem : type
        problem to be solved
    kwargs : dictionary
        kwargs['rand_selection_confidence'] must be a probability value
        kwargs['upper_percentile'] must be between 0 and 1
        kwargs['choose'] must be a positive integer

    Returns
    -------
    output : tuple
        (optimal subset, optimal objective value, objective trace)
        (list of data dictionaries, float, list)

    """
    maximum = -10000000
    choose = max(2, min(kwargs['choose'], len(problem.data)))
    objective_trace = []
    numerator = math.log(1 - kwargs['rand_selection_confidence'])
    denominator = math.log(kwargs['upper_percentile'])
    num_selections = int(math.ceil(numerator/denominator))
    for i in range(num_selections):
        random_subset = list(numpy.random.choice(problem.data, choose,
                                                 replace=False))
        # data_list = [datum['data']['values'] for datum in random_subset]
        objective_value = problem.objective_function(random_subset, **kwargs)
        if objective_value > maximum:
            maximum = objective_value
            optimum = random_subset
            objective_trace.append(objective_value)
    # return (optimum, maximum)
    output = (optimum, objective_trace[-1], objective_trace)
    return output


# The following algorithms are from Scikit Learn
# They call the appropriate function from Scikit Learn and only keep
# coefficients above tolerance given by kwargs['coef_tolerance']
def ridge_regression(problem, **kwargs):
    r"""High level description.

    Parameters
    ----------
    problem : type
        Description
    kwargs : dictionary
        kwargs['ridge_reg_coef'] must be a nonnegative float.  This is the
        multiplier for the penalty term
        kwargs['coef_tolerance'] must be a nonnegative float

    Returns
    -------
    output : tuple
        (optimum, maximum)

    """
    data_list = [datum['data']['values'] for datum in problem.data]
    data = numpy.array(data_list)
    ridge = Ridge(kwargs['ridge_reg_coef'])
    ridge.fit(data.T, problem.goal['data']['values'])
    ridge_regression_coefficients = ridge.coef_
    optimum = [problem.data[index] for index,element in
               enumerate(ridge_regression_coefficients)
               if abs(element) > kwargs['coef_tolerance']]
    maximum = ridge.score(data.T, problem.goal['data']['values'])
    output = (optimum, maximum)
    return output


def ridge_regression_cv(problem, **kwargs):
    r"""High level description.

    Parameters
    ----------
    problem : type
        Description
    kwargs : dictionary
        kwargs['ridge_reg_coefs'] must be a list of nonnegative float.  These
        are the multipliers for the penalty term in cross-validation of ridge
        regression
        kwargs['coef_tolerance'] must be a nonnegative float

    Returns
    -------
    output : tuple
        (optimum, maximum)

    """
    data_list = [datum['data']['values'] for datum in problem.data]
    data = numpy.array(data_list)
    ridge = RidgeCV(kwargs['ridge_reg_coefs'])
    ridge.fit(data.T, problem.goal['data']['values'])
    ridge_regression_coefficients = ridge.coef_
    optimum = [problem.data[index] for index,element in
               enumerate(ridge_regression_coefficients)
               if abs(element) > kwargs['coef_tolerance']]
    maximum = ridge.score(data.T, problem.goal['data']['values'])
    output = (optimum, maximum)
    return output


def LASSO(problem, **kwargs):
    r"""High level description.

    Parameters
    ----------
    problem : type
        Description
        kwargs['LASSO_reg_coef'] must be a nonnegative float.  This is the
        multiplier for the penalty term
        kwargs['coef_tolerance'] must be a nonnegative float

    Returns
    -------
    output : tuple
        (optimum, maximum)

    """
    data_list = [datum['data']['values'] for datum in problem.data]
    data = numpy.array(data_list)
    lasso = Lasso(alpha=kwargs['LASSO_reg_coef'])
    lasso.fit(data.T, problem.goal['data']['values'])
    lasso_coefficients = lasso.coef_
    optimum = [problem.data[index] for index,element in
               enumerate(lasso_coefficients)
               if abs(element) > kwargs['coef_tolerance']]
    maximum = lasso.score(data.T, problem.goal['data']['values'])
    output = (optimum, maximum)
    return output


def LASSO_cv(problem, **kwargs):
    r"""High level description.

    Parameters
    ----------
    problem : type
        Description
        kwargs['LASSO_reg_coefs'] must be a nonnegative float.  These are the
        multipliers for the penalty term in cross-validation of LASSO
        kwargs['coef_tolerance'] must be a nonnegative float

    Returns
    -------
    output : tuple
        (optimum, maximum)

    """
    data_list = [datum['data']['values'] for datum in problem.data]
    data = numpy.array(data_list)
    lasso = LassoCV(alphas=kwargs['LASSO_reg_coefs'])
    lasso.fit(data.T, problem.goal['data']['values'])
    lasso_coefficients = lasso.coef_
    optimum = [problem.data[index] for index,element in
               enumerate(lasso_coefficients)
               if abs(element) > kwargs['coef_tolerance']]
    maximum = lasso.score(data.T, problem.goal['data']['values'])
    output = (optimum, maximum)
    return output


def elastic_net(problem, **kwargs):
    r"""High level description.

    Parameters
    ----------
    problem : type
        Description
        kwargs['elastic_net_reg_coef'] must be a nonnegative float.  This is
        the multiplier for the penalty term
        kwargs['elastic_net_ratio'] must be between 0 and 1
        kwargs['coef_tolerance'] must be a nonnegative float

    Returns
    -------
    output : tuple
        (optimum, maximum)

    """
    data_list = [datum['data']['values'] for datum in problem.data]
    data = numpy.array(data_list)
    elastic_net = ElasticNet(alpha=kwargs['elastic_net_reg_coef'],
                             l1_ratio=kwargs['elastic_net_ratio'])
    elastic_net.fit(data.T, problem.goal['data']['values'])
    elastic_net_coefficients = elastic_net.coef_
    optimum = [problem.data[index] for index,element in
               enumerate(elastic_net_coefficients)
               if abs(element) > kwargs['coef_tolerance']]
    maximum = elastic_net.score(data.T, problem.goal['data']['values'])
    output = (optimum, maximum)
    return output


def elastic_net_cv(problem, **kwargs):
    r"""High level description.

    Parameters
    ----------
    kwargs['elastic_net_reg_coefs'] must be a list of nonnegative float.  These
    are the multiplier for the penalty term in cross-validation of EN

    kwargs['elastic_net_ratio'] must be between 0 and 1

    kwargs['coef_tolerance'] must be a nonnegative float

    Returns
    -------
    output : tuple
        (optimum, maximum)

    """
    data_list = [datum['data']['values'] for datum in problem.data]
    data = numpy.array(data_list)
    elastic_net = ElasticNetCV(alphas=kwargs['elastic_net_reg_coefs'],
                               l1_ratio=kwargs['elastic_net_ratio'])
    elastic_net.fit(data.T, problem.goal['data']['values'])
    elastic_net_coefficients = elastic_net.coef_
    optimum = [problem.data[index] for index,element in
               enumerate(elastic_net_coefficients)
               if abs(element) > kwargs['coef_tolerance']]
    maximum = elastic_net.score(data.T, problem.goal['data']['values'])
    output = (optimum, maximum)
    return output


def OMP(problem, **kwargs):
    r"""High level description.

    Parameters
    ----------
    problem : type
        Description
    kwargs : dictionary
        kwargs['choose'] must be a positive integer
        kwargs['coef_tolerance'] must be a nonnegative float

    Returns
    -------

    """
    data_list = [datum['data']['values'] for datum in problem.data]
    data = numpy.array(data_list)
    OMP = OrthogonalMatchingPursuit(n_nonzero_coefs=kwargs['choose'])
    OMP.fit(data.T, problem.goal['data']['values'])
    OMP_coefficients = OMP.coef_
    optimum = [problem.data[index] for index,element in
               enumerate(OMP_coefficients)
               if abs(element) > kwargs['coef_tolerance']]
    maximum = OMP.score(data.T, problem.goal['data']['values'])
    return (optimum, maximum)


def OMP_cv(problem, **kwargs):
    r"""High level description.

    Requirements
    ------------
    kwargs['choose'] must be a positive integer

    kwargs['coef_tolerance'] must be a nonnegative float

    Returns
    -------
    output : tuple
        (optimum, maximum)

    """
    data_list = [datum['data']['values'] for datum in problem.data]
    data = numpy.array(data_list)
    OMP = OrthogonalMatchingPursuitCV(max_iter=kwargs['choose'])
    OMP.fit(data.T, problem.goal['data']['values'])
    OMP_coefficients = OMP.coef_
    optimum = [problem.data[index] for index,element in
               enumerate(OMP_coefficients)
               if abs(element) > kwargs['coef_tolerance']]
    maximum = OMP.score(data.T, problem.goal['data']['values'])
    output = (optimum, maximum)
    return output
