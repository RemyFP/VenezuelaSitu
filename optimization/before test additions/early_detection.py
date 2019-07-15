# -*- coding: utf-8 -*-

r"""Functions for early detection.


"""

import copy
import datetime
import numpy as np
import scipy.stats as stats
# from sklearn.cross_validation import KFold
from sklearn.model_selection import KFold
from functools import partial
import data_manipulation as dm
import filter_selection as fs
import train_test as tt


def null_dist(goal_datum, data_sources, threshold):
    r"""Generate null distributions each of which only includes observations
    below the threshold from each data source (non-outbreak period).

    Parameters
    ----------
    goal_datum : dictionary
        The gold standard data source.
    data_sources : list
        List of data sources that will be evaluated. Each data source is a
        dictionary.
    threshold : float
        Event baseline. When the value is larger than the threshold, there is
        an event. Otherwise, there is no event.

    Returns
    -------
    empirical_dist_0 : list
        List of arrays. Each array contains observations below the
        threshold at time t from all data sources.
    mu_0 : array
        Elements in the array are mean values of m observations below the
        threshold from one data source.
    inv_Sigma_0 : array
        Inverse of array Sigma_0. Elements in Sigma_0 are covariances of each
        two data sources (only including observations below the threshold).

    """
    f = fs.filter_on_values(goal_datum, below=threshold)
    series_list_null = [fs.select(d, f)['data']['values']
                        for d in data_sources]
    empirical_dist_0 = [np.array(series_list_null)[:,t]
                        for t in range(len(series_list_null[0]))]
    mu_0 = np.mean(series_list_null, axis=1)
    Sigma_0 = np.cov(series_list_null)
    # Uses pseudoinverse in case of singular matrix from missing data
    if len(data_sources) == 1:
        if Sigma_0:
            inv_Sigma_0 = 1/Sigma_0
        else:
            inv_Sigma_0 = 0
    else:
        inv_Sigma_0 = np.linalg.pinv(Sigma_0)
    return empirical_dist_0, mu_0, inv_Sigma_0


def qform(A, x):
    r"""Calculate dot product of two arrays.

    Format of the calculation: x.T*A*x.

    Parameters
    ----------
    A : array
        A one-dimension array.
    x : array
        A one-dimension array.

    Returns
    -------
    Dot product of two arrays.

    """
    return np.dot(x.T, np.dot(A,x))


def vpos(x):
    r"""Substitute elements with 0 if non positive.

    Parameters
    ----------
    x : list
        A list of numbers.

    Returns
    -------
    A list of numbers in which negative numbers are substituted with 0.

    """
    return np.array([y*(y>0) for y in x])


def EED(data_sources, alg, h, auto_reset=False, len_outbreak=8):
    r"""Generalized Early Event Detection (EED) method with chose update method
    and optional reset.

    Parameters
    ----------
    data_sources : list
        List of data sources that will be evaluated. Each data source is a
        dictionary.
    alg : string
        An early event detection method, including MEWMA, cCUSUM, MCUSUM.
    h : number
        A value below which an alarm will not be triggered (no outbreak).
    auto_reset : boolean, optional
        If the updated S statistic should be reset. The default is False.
    len_outbreak : integer, optional
        The length of the outbreak. The default length is 8.

    Returns
    -------
    alarm : array (boolean data type)
        Indicate which time an EED method signals.
    S : list
        Elements in the list are arrays. Each element is the updated S statistic
        at each time step.
    E : list
        Elements in the list are the EED method test statistic E values at each
        time step.

    """
    S, E = [], []
    S_prev = np.zeros(len(data_sources))
    for t in range(dm.length(data_sources[0])):
        series_list = [d['data']['values'] for d in data_sources]
        Y_t = np.array(series_list)[:,t]
        S_next, E_next = alg(S_prev, Y_t)
        S.append(S_next)
        E.append(E_next)
        S_prev = S_next
        if auto_reset:
            len_regression = int(round(len_outbreak/2))
            slope, _, _, p_value, _ = stats.linregress(
                    range(len(E[-len_regression:])),E[-len_regression:])
            reset = (E[-1] > h)*(slope < 0)*(p_value < 0.05)
            S_prev = (not reset)*S_prev
    alarm = (np.array(E) >= h)
    return alarm, S, E


def MEWMA(s, y, mu_0, inv_Sigma_0, l):
    r"""Multivariate Exponentially Weighted Moving Average (MEWMA).

    Parameters
    ----------
    s : array
        The previous S statistic.
    y : array
        The current observation.
    mu_0 : array
        Elements in the array are mean values of m observations below the
        threshold from one data source.
    inv_Sigma_0 : array
        Inverse of array Sigma_0. Elements in Sigma_0 are covariances of each
        two data sources (only including observations below the threshold).
    l : number
        The smoothing parameter. When l is approaching 0, this method is similar
        to MCUSUM. When l=1, the method is equivalent to Hotelling's
        chi-squared.

    Returns
    -------
    s_next : array
        The updated S statistic.
    e_next : number
        The MEWMA test statistic E value. If E >= threshold h, the MEWMA signals.

    """
    s_next = vpos(l*(y - mu_0) + (1-l)*s)
    #e_next = qform(np.linalg.inv((l/(2-l))*Sigma_0), s_next)
    e_next = qform(((2-l)/l)*inv_Sigma_0, s_next)
    return s_next, e_next

# NOTE: NOT IN USE
def cCUSUM(s, y, mu_0, Sigma_0, k):
    r"""Cumulative Sum(CUSUM) method.

    combined univariate CUSUM (does not take into account correlations between
    data sources).

    """
    sigma_0 = np.diag(Sigma_0)
    l = (y - mu_0)/sigma_0 - k*np.ones(len(s))
    s_next = vpos(s + l)
    e_next = np.sum(s_next)
    return s_next, e_next


# NOTE: NOT IN USE
def MCUSUM(s, y, mu_0, inv_Sigma_0, k):
    r"""Multivariate Cumulative Sum (MCUSUM).

    """
    s_star = s + y - mu_0
    d = np.sqrt(qform(inv_Sigma_0, s_star))
    s_next = vpos(s_star*(1 - k/np.max([k, d])))
    e_next = np.sqrt(qform(inv_Sigma_0, s_next))
    return s_next, e_next


def ATFS(dist_0, alg, h, num_samples=100, maxiter=1000):
    r"""Calculate the average time between false signals (ATFS).

    Parameters
    ----------
    dist_0 : list
        List of arrays. Each array contains observations below the
        threshold (event baseline) at time t from all data sources.
    alg : string
        An early event detection method, including MEWMA, cCUSUM, MCUSUM.
    h : number
        A value below which an alarm will not be triggered (no event).
    num_samples : integer
        The number of samples. The default is 100.
    maxiter : integer
        Maximum number of iterations. The default is 1000.

    Returns
    -------
    ATFS_samples : list
        The time between false signals in one sample.
    FPR : float
        False positive rate.

    """
    ATFS_samples = []
    for i in range(num_samples):
        S, E = [], []
        S_prev = np.zeros(len(dist_0[0]))
        alarm = False
        while not alarm and len(E) <= maxiter:
            Y_t = dist_0[np.random.choice(len(dist_0))]
            S_next, E_next = alg(S_prev, Y_t)
            S.append(S_next)
            E.append(E_next)
            S_prev = S_next
            alarm = (E[-1] >= h)
        ATFS_samples.append(len(E))
        ATFS_avg = np.mean(ATFS_samples)
        #  FPR = 1/np.mean(ATFS_samples)
    #  return ATFS_samples, FPR
    return ATFS_samples, ATFS_avg

# Secant method with positivity enforced
def secant(f, x0, x1, tol=1, maxiter=10):
    r""" Find a root of a function *f* using the secant method.

    Parameters
    ----------
    f : given function
        The function *f* whose root will be found in the algorithm.
    x0 : number
        Start value. One of the end point value.
    x1 : number
        Start value. The second end point value.
    tol : number
        Tolerance value. The default is 1.
    maxiter : integer
        The max number of iterations. The default is 10.

    Returns
    -------
    x1 : number
        The root of the function *f*.

    """
    for i in range(maxiter):
        if f(x1)-f(x0) == 0:
            return x1
        x_temp = x1 - (f(x1)*(x1-x0)*1.0)/(f(x1)-f(x0))
        #x_temp = min(x_temp, 100)
        x_temp = max(x_temp, 0)
        #print x_temp
        x0 = x1
        x1 = x_temp
        if abs(x1-x0) < tol:
            return x1
    return x1


def pred_CV_candidate(goal_datum, data_sources, threshold, l, h,
        auto_reset=False, len_outbreak=8, n_folds=1, bootstrap=False):
    r"""Cross-validated train/test function for early detection.

    Parameters
    ----------
    goal_datum : dictionary
        The gold standard data source.
    data_sources : list
        List of data sources that will be evaluated. Each data source is a
        dictionary.
    threshold : float
        Event baseline. When the value is larger than the threshold, there is
        an event. Otherwise, there is no event.
    l : float
        The smoothing parameter. A parameter in EED method MEWMA.
    h : number
        A value below which an alarm will not be triggered (no event).
    auto_reset : boolean
        If the updated S statistic should be reset. The default is False.
    len_outbreak : integer
        The length of the outbreak. The default length is 8.
    n_folds : integer
        Number of fold in k-fold cross-validation
    bootstrap : boolean
        Whether or not to use bootstrap data sets, which are created by sampling
        with replacement and has the same size as the original training dataset.

    Returns
    -------
    alarm_CV : dictionary
        Including date and whether or not alarm is triggered at that date.
    S_CV : ndarray
        Elements in the array are lists. Each element includes the updated S
        statistic for each fold.
    E_CV : array
        Elements in the array are the EED method test statistic E values for
        each fold.

    """
    if n_folds > 1:
        # kf = KFold(dm.length(goal_datum), n_folds)
        kf_p = KFold(n_folds)
        kf = list(kf_p.split(range(dm.length(goal_datum))))
    else:
        v = range(dm.length(goal_datum))
        kf = [(v,v)]
    alarm_CV_list, S_CV_list, E_CV_list = [], [], []
    for train, test in kf:
        if bootstrap:
            train = np.random.choice(train, len(train))
        train_f, test_f = tt.index_to_filter(train), tt.index_to_filter(test)
        _, mu_0, inv_Sigma_0 = tt.train_on_filter(partial(null_dist,
            threshold=threshold), goal_datum, data_sources, train_f)
        EED_MEWMA = partial(EED, alg=partial(MEWMA, mu_0=mu_0,
            inv_Sigma_0=inv_Sigma_0, l=l),
            h=h, auto_reset=auto_reset, len_outbreak=len_outbreak)
        alarm, S, E = tt.test_on_filter(EED_MEWMA, data_sources, test_f)
        alarm_CV_list.append(alarm)
        S_CV_list.append(S)
        E_CV_list.append(E)
    S_CV = np.concatenate(S_CV_list, axis=0)
    E_CV = np.concatenate(E_CV_list, axis=0)
    alarm_CV = {}
    alarm_CV['data'] = {}
    alarm_CV['data']['values'] = np.concatenate(alarm_CV_list, axis=0)
    alarm_CV['data']['times'] = goal_datum['data']['times']
    return alarm_CV, S_CV, E_CV


#  def candidate_params(goal_datum, data_sources, threshold, log_FPR_threshold,
        #  num_samples=100, dl=0.1):
def candidate_params(goal_datum, data_sources, threshold, log_ATFS_threshold,
        num_samples=100, dl=0.1):
    r"""Choose the optimized smoothing parameter l and the threshold h.

    Parameters
    ----------
    goal_datum : dictionary
        The gold standard data source.
    data_sources : list
        List of data sources that will be evaluated. Each data source is a
        dictionary.
    threshold : float
        Event baseline. When the value is larger than the threshold, there is
        an event. Otherwise, there is no event.
    log_FPR_threshold : float
        Natural log of the False Positive Rate(FPR) threshold. The FPR is the
        inverse of Average Time Between False signals(ATFS).
    num_samples : integer, optional
        The number of samples. The default is 100.
    dl : number, optional
        Spacing between values for a series of smoothing parameters l. The
        default distance between two adjacent values l is 0.1.

    Returns
    -------
    param_tuples : tuple
        Elements in the tuple are combinations of alarm threshold h and
        smoothing parameter l for specified False Positive Rate(FPR).

    """
    # compute empirical null dist to find level set of chosen log FPR
    empirical_dist_0, mu_0, inv_Sigma_0 = null_dist(goal_datum,
            data_sources, threshold)
    L = np.arange(dl, 1, dl)
    param_tuples = []
    h_star = 1
    #if len(data_sources) == 1:
    #    inv_Sigma_0 = 1/Sigma_0
    #else:
    #    inv_Sigma_0 = np.linalg.inv(Sigma_0)
    for l in L:
        MEWMA_l = partial(MEWMA, mu_0=mu_0, inv_Sigma_0=inv_Sigma_0, l=l)
        #  f = lambda h: np.log(ATFS(empirical_dist_0, MEWMA_l, h=h,
            #  num_samples=num_samples)[1]) - log_FPR_threshold
        f = lambda h: np.log(ATFS(empirical_dist_0, MEWMA_l, h=h,
            num_samples=num_samples)[1]) - log_ATFS_threshold
        h_star = secant(f, h_star+1, h_star, tol=0.5, maxiter=10)
        param_tuples.append((l, h_star))
    return param_tuples


def start(alarm):
    r"""Determine the first alarm in a cluster of alarms.

    Parameters
    ----------
    alarm : dict
        Including date and whether or not alarm is triggered at that date.

    Returns
    -------
    alarm_start : dict
        Including date and whether the alarm at that date is the first alarm in
        a cluster of alarms.

    """
    b = alarm['data']['values']
    db = b - np.concatenate(([0], b[:-1]), axis=0)
    alarm_start = copy.deepcopy(alarm)
    alarm_start['data']['values'] = (db > 0)
    return alarm_start


def ED_obj(alarm, goal_datum, threshold=2.0,
        min_time_above_threshold=datetime.timedelta(weeks=3),
        min_time_between_events=datetime.timedelta(weeks=8)):
    r"""Objective function for Early Event Detection.

    Parameters
    ----------
    alarm : dict
        Including date and whether or not alarm is triggered at that date.
    goal_datum : dictionary
        The gold standard data source.
    threshold : float
        Event baseline. When the value is larger than the threshold, there is
        an event. Otherwise, there is no event.
    min_time_above_threshold : datetime.timedelta
        Minimum amount of time above threshold required post-crossing to qualify
        as event.
    min_time_between_events : datetime.timedelta
        Minimum buffer between potential events.

    Returns
    -------
    average_obj : number
        The average of the objective value.
    weeks_ahead_list : list
        How many weeks ahead in an event the alarm is triggered.

    """
    events = find_events(goal_datum, threshold=threshold,
            min_time_above_threshold=min_time_above_threshold,
            min_time_between_events=min_time_between_events)
    total_obj = 0
    weeks_ahead_list = []
    alarm0 = start(alarm)
    total_alarm_times = [t for t,v in zip(alarm0['data']['times'],
        alarm0['data']['values']) if v==True]
    for event in events:
        f = fs.filter_on_times(alarm0,
                after=event-min_time_between_events/2,
                before=event+min_time_between_events/2)
        event_alarm = fs.select(alarm0, f)
        alarm_times = [t for t,v in zip(event_alarm['data']['times'],
            event_alarm['data']['values']) if v==True]
        if alarm_times:
            first_alarm = min(alarm_times)
            diff = (event - min_time_between_events/2 - first_alarm).days/7
            weeks_ahead = (first_alarm - event).days/7
        else:
            diff = -np.inf
            weeks_ahead = 'NA'
        total_obj = total_obj + np.exp(diff)
        weeks_ahead_list.append(weeks_ahead)
    if len(events) > 0:
        average_obj = total_obj/len(events)
    else:
        average_obj = 0
    return average_obj, weeks_ahead_list
    #  return average_obj, weeks_ahead_list, alarm, alarm0


def argmax(keys, func):
    r"""Choose the input that can maximize the output of the function.

    Parameters
    ----------
    keys : tuple
        Elements in the tuple are inputs of the function.
    func : string
        Input function.

    Returns
    -------
    tuple
    The first element is the output of the function. The second element is the
    input maximizing the function.

    """
    return max((func(*key), key) for key in keys)


def ed_objective(problem, subset, **kwargs):
    r"""Early detection objective function.

    The objective function for old version.

    Parameters
    ----------
    problem : feature selection problem
        See Class `problem`.
    subset : list
        List of data sources on which to evaluate objective function. Each data
        source is a dictionary.

    Returns
    -------
    ED_optimization_value : number
        The objective value for early detection.

    """
    #  param_tuples = candidate_params(problem.goal, subset, problem.threshold,
            #  problem.log_FPR_threshold)
    param_tuples = candidate_params(problem.goal, subset, problem.threshold,
            problem.log_ATFS_threshold)
    alarm_lh = lambda l, h: pred_CV_candidate(problem.goal,
        subset, threshold=problem.threshold, l=l, h=h, n_folds=problem.n_folds,
        bootstrap=problem.bootstrap)[0]
    ED_obj_lh = lambda l, h: ED_obj(alarm_lh(l,h), problem.goal, problem.threshold)
    opt = argmax(param_tuples, ED_obj_lh)
    ED_optimization_value = opt[0][0]
    return ED_optimization_value


def ed_objective_2(problem, subset):
    r"""Early detection objective function.

    The objective function for new version.

    Parameters
    ----------
    problem : surveillance optimization problem
        See Class `surveillance_optimization`.
    subset : list
        List of data sources on which to evaluate objective function. Each data
        source is a dictionary.

    Returns
    -------
    ED_optimization_value : number
        The objective value for early detection.

    """
    #  param_tuples = candidate_params(problem.gold_standard, subset,
            #  problem.threshold, problem.log_FPR_threshold)
    param_tuples = candidate_params(problem.gold_standard, subset,
            problem.threshold, problem.log_ATFS_threshold)
    alarm_lh = lambda l, h: pred_CV_candidate(problem.gold_standard,
        subset, threshold=problem.threshold, l=l, h=h, n_folds=problem.n_folds,
        bootstrap=problem.bootstrap)[0]
    ED_obj_lh = lambda l, h: ED_obj(alarm_lh(l,h), problem.gold_standard,
            problem.threshold)
    opt = argmax(param_tuples, ED_obj_lh)
    ED_optimization_value = opt[0][0]
    return ED_optimization_value


#  def pred_CV(goal_datum, data_sources, threshold, log_FPR_threshold,
        #  num_samples = 100, dl=0.1, tol=0.5, maxiter=10, n_folds=1,
        #  bootstrap=False):
def pred_CV(goal_datum, data_sources, threshold, log_ATFS_threshold,
        num_samples = 100, dl=0.1, tol=0.5, maxiter=10, n_folds=1,
        bootstrap=False):
    r"""Test the performance of the early detection model using
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
    threshold : float
        Event baseline. When the value is larger than the threshold, there is
        an event. Otherwise, there is no event.
    log_FPR_threshold : float
        Natural log of the False Positive Rate(FPR) threshold. The FPR is the
        inverse of Average Time Between False signals(ATFS).
    num_samples : integer, optional
        The number of samples. The default is 100.
    dl : number, optional
        Spacing between values for a series of smoothing parameters l. The
        default distance between two adjacent values l is 0.1.
    tol : number
        Tolerance value. The default is 1.
    maxiter : integer
        The max number of iterations. The default is 10.
    n_folds : integer
        Number of fold in k-fold cross-validation
    bootstrap : boolean
        Whether or not to use bootstrap data sets, which are created by sampling
        with replacement and the same size as the original training dataset.

    Returns
    -------
    alarm_opt : dictionary
        Including date and whether or not alarm is triggered at that date.
    alarm_start_opt : dict
        Including date and whether the alarm at that date is the first alarm in
        a cluster of alarms.
    ED_optimization_value : number
        The objective value for early detection.
    opt[1] : tuple
        The threshold h and smoothing parameter l that can maximize the
        early detection objective function.
    weeks_ahead_list : list
        How many weeks ahead in an event the alarm is triggered.

    """
    #  param_tuples = candidate_params(goal_datum, data_sources, threshold,
            #  log_FPR_threshold)
    param_tuples = candidate_params(goal_datum, data_sources, threshold,
            log_ATFS_threshold)
    alarm_lh = lambda l, h: pred_CV_candidate(goal_datum,
        data_sources, threshold=threshold, l=l, h=h, n_folds=n_folds,
        bootstrap=False)[0]
    ED_obj_lh = lambda l, h: ED_obj(alarm_lh(l,h), goal_datum, threshold)
    opt = argmax(param_tuples, ED_obj_lh)
    ED_optimization_value = opt[0][0]
    weeks_ahead_list = opt[0][1]
    #  alarm = opt[0][2]
    #  alarm_start = opt[0][3]
    alarm_opt = pred_CV_candidate(goal_datum, data_sources,
        threshold=threshold, l=opt[1][0], h=opt[1][1],
        n_folds=n_folds, bootstrap=False)[0]
    alarm_start_opt = start(alarm_opt)
    return alarm_opt, alarm_start_opt, ED_optimization_value, opt[1], weeks_ahead_list
    #  return alarm, alarm_start, ED_optimization_value, opt[1], weeks_ahead_list


############################################
# OLD EED METHODS
# NOTE: STILL USEFUL! REWRITE AS NECESSARY
############################################

def find_events_raw(raw_datum, threshold=2.0,
        min_time_above_threshold=datetime.timedelta(weeks=3),
        min_time_between_events=datetime.timedelta(weeks=5)):
    r"""Finds times at which events occur in the given datum.

    Parameters
    ----------
    raw_datum : dict in raw format
        Datum on which to find events.
    threshold : float
        Event baseline. When the value is larger than the threshold, there is
        an event. Otherwise, there is no event.
    min_time_above_threshold : datetime.timedelta
        Minimum amount of time above threshold required post-crossing to qualify
        as event.
    min_time_between_events : datetime.timedelta
        Minimum buffer between potential events.

    Returns
    -------
    events : list
        Event times as Pandas timestamps

    """
    d = raw_datum
    e = []
    for i in range(1,len(d['times'])):
        if d['values'][i-1] < threshold <= d['values'][i]:
            potential_event = d['times'][i]
            f = fs.filter_on_times_raw(d, after=potential_event,
                    before=potential_event + min_time_above_threshold)
            if all(val >= threshold for val in fs.select_raw(d,f)['values']):
                e.append(potential_event)
    if e:
        events = [e[0]]
        for j in range(1,len(e)):
            if e[j] - events[-1] >= min_time_between_events:
                events.append(e[j])
    else:
        events = []
    return events


def find_events(datum, threshold=2.0,
        min_time_above_threshold=datetime.timedelta(weeks=3),
        min_time_between_events=datetime.timedelta(weeks=5)):
    r"""Finds events for datum in standard format.

    Parameters
    ----------
    raw_datum : dict in standard format
        Datum on which to find events.
    threshold : float
        Event baseline. When the value is larger than the threshold, there is
        an event. Otherwise, there is no event.
    min_time_above_threshold : datetime.timedelta
        Minimum amount of time above threshold required post-crossing to qualify
        as event.
    min_time_between_events : datetime.timedelta
        Minimum buffer between potential events.

    Returns
    -------
    events : list
        Event times as Pandas timestamps

    """
    return find_events_raw(datum['data'],
            min_time_above_threshold=min_time_above_threshold,
            min_time_between_events=min_time_between_events)


def find_peaks_between(datum, events):
    boundaries = events + [datum['data']['times'][-1]]

    peaks = []
    for i in range(0,len(boundaries)-1):
        f = fs.filter_on_times(datum, after=boundaries[i],
                before=boundaries[i+1])
        _, peak_time = find_max(fs.select(datum, f))
        peaks.append(peak_time)

    return peaks

def find_max(datum):
    # return max value and first time associated with it
    max_value = max(datum['data']['values'])
    max_index = datum['data']['values'].index(max_value)
    max_time = datum['data']['times'][max_index]

    return max_value, max_time




### DELETE ALL THESE IF NOT NEEDED
def generate_goal(datum, threshold=2.0,
        min_time_above_threshold=datetime.timedelta(weeks=3),
        keep_data_within=datetime.timedelta(weeks=5),
        event_boundary=datetime.timedelta(weeks=1)):
    """Generates goal for early detection from datum in standard format.

    Parameters
    ----------
    :param datum: datum from which events are generated

    :type datum: data dictionary in standard format

    Remaining input parameters same as for find_events.

    :return: (goal datum restricted to event intervals, pair of filters
        corresponding to intervals before/after events)
    :rtype: (data dictionary, list of two boolean lists)
    """
    events = find_events(datum, threshold=threshold,
            min_time_above_threshold=min_time_above_threshold,
            min_time_between_events=keep_data_within)

    # Data after events
    filters_before = []
    filters_during = []
    filters_after = []
    for event in events:
        f0 = fs.filter_on_times(datum, after=event-keep_data_within,
                before=event-event_boundary)
        filters_before.append(f0)
        f1 = fs.filter_on_times(datum, after=event-event_boundary,
                before=event+event_boundary)
        filters_during.append(f1)
        f2 = fs.filter_on_times(datum, after=event+event_boundary,
                before=event+keep_data_within)
        filters_after.append(f2)

    f_before_merged = fs.merge_filters(filters_before, method='any')
    goal_before = fs.select(datum, f_before_merged)
    goal_before['data']['values'] = [0] * len(goal_before['data']['values'])
    f_during_merged = fs.merge_filters(filters_during, method='any')
    goal_during = fs.select(datum, f_during_merged)
    goal_during['data']['values'] = [1] * len(goal_during['data']['values'])
    f_after_merged = fs.merge_filters(filters_after, method='any')
    goal_after = fs.select(datum, f_after_merged)
    goal_after['data']['values'] = [2] * len(goal_after['data']['values'])

    goal = copy.deepcopy(datum)
    d0 = dm.raw_to_pandas(goal_before['data'])
    d1 = dm.raw_to_pandas(goal_during['data'])
    d2 = dm.raw_to_pandas(goal_after['data'])
    goal['data'] = dm.sort_raw(dm.pandas_to_raw(
        d0.add(d1.add(d2, fill_value=0), fill_value=0)))

    return (goal, [f_before_merged, f_during_merged, f_after_merged])

def generate_goal2(datum, threshold=2.0,
        min_time_above_threshold=datetime.timedelta(weeks=3),
        min_time_between_events=datetime.timedelta(weeks=5),
        event_boundary=datetime.timedelta(weeks=1)):
    """Generates goal for early detection from datum in standard format.

    Parameters
    ----------
    :param datum: datum from which events are generated

    :type datum: data dictionary in standard format

    Remaining input parameters same as for find_events.

    :return: (goal datum restricted to event intervals, pair of filters
        corresponding to intervals before/after events)
    :rtype: (data dictionary, list of two boolean lists)
    """
    events = find_events(datum, threshold=threshold,
            min_time_above_threshold=min_time_above_threshold,
            min_time_between_events=min_time_between_events)

    # Data after events
    filters_during = []
    for event in events:
        f0 = fs.filter_on_times(datum, after=event-event_boundary,
                before=event+event_boundary)
        filters_during.append(f0)

    f_during_merged = fs.merge_filters(filters_during, method='any')

    goal_outside = fs.exclude(datum, f_during_merged)
    goal_outside['data']['values'] = [0] * len(goal_outside['data']['values'])
    goal_during = fs.select(datum, f_during_merged)
    goal_during['data']['values'] = [1] * len(goal_during['data']['values'])

    goal = copy.deepcopy(datum)
    d0 = dm.raw_to_pandas(goal_outside['data'])
    d1 = dm.raw_to_pandas(goal_during['data'])
    goal['data'] = dm.sort_raw(dm.pandas_to_raw(d0.add(d1, fill_value=0)))

    return (goal, f_during_merged)

def generate_goal3(datum, threshold=2.0,
        min_time_above_threshold=datetime.timedelta(weeks=3),
        min_time_between_events=datetime.timedelta(weeks=5),
        event_boundary=datetime.timedelta(weeks=3)):
    """Generates goal for early detection from datum in standard format.

    Parameters
    ----------
    :param datum: datum from which events are generated

    :type datum: data dictionary in standard format

    Remaining input parameters same as for find_events.

    :return: (goal datum restricted to event intervals, pair of filters
        corresponding to intervals before/after events)
    :rtype: (data dictionary, list of two boolean lists)
    """
    events = find_events(datum, threshold=threshold,
            min_time_above_threshold=min_time_above_threshold,
            min_time_between_events=min_time_between_events)

    peaks = find_peaks_between(datum, events)

    # Data after events
    filters_after = []
    for i in range(0,len(events)):
        f1 = fs.filter_on_times(datum, after=events[i],
                before=peaks[i])
        filters_after.append(f1)

    f_after_merged = fs.merge_filters(filters_after, method='any')

    goal_before = fs.exclude(datum, f_after_merged)
    goal_before['data']['values'] = [0] * len(goal_before['data']['values'])
    goal_after = fs.select(datum, f_after_merged)
    goal_after['data']['values'] = [1] * len(goal_after['data']['values'])

    goal = copy.deepcopy(datum)
    d0 = dm.raw_to_pandas(goal_before['data'])
    d1 = dm.raw_to_pandas(goal_after['data'])
    goal['data'] = dm.sort_raw(dm.pandas_to_raw(d0.add(d1, fill_value=0)))

    return (goal, f_after_merged)



def generate_lagged_data(datum, goal_datum,
        min_lag=datetime.timedelta(weeks=3), max_history=5):
    """Formats datum for early detection.

    Parameters
    ----------
    :param datum: datum to be formatted
    :param goal_datum: goal datum generated from generate_goal
    :param min_lag: minimum amount of time to lag data
    :param max_history: maximum size of lagged dataset for each time point

    :type datum: data dictionary in standard format
    :type goal_datum: data dictionary
    :type min_lag: datatime.timedelta
    :type max_history: positive integer

    :return: formatted datum with lagged data appended to key 'lagged' in
        dictionary 'data'
    :rtype: data dictionary in (appended) standard format
    """
    datum = dm.fillna(datum, 0) #fill NAs with 0 automatically for now
    lagged_data = []
    for t in goal_datum['data']['times']:
        f = fs.filter_on_times(datum, before=(t - min_lag))
        v = fs.select(datum, f)['data']['values'][-max_history:]
        if len(v) < max_history:
            v = [0.0]*(max_history - len(v)) + v
        lagged_data.append(v)

    d = dm.fillna(dm.interpolate(datum, goal_datum), 0)
    d['data']['lagged'] = lagged_data
    #d['data']['lagged'] = [list(l) for l in zip(*lagged_data)]

    return d
