# -*- coding: utf-8 -*-

r"""
High level description of the module.
"""

# TODO:
#   * Fix docstrings.
#   * Re-factor to remove repetition.

import os
import copy
import numpy as np
import pandas as pd


def listify(x):
    r"""Converts element to list if not already a list.

    Parameters
    ----------
    x : type
        Description

    Returns
    -------
    x : type
        Description

    """
    if isinstance(x, list):
        return x
    else:
        return [x]


def info(data):
    r"""Displays basic metadata for datum in dataset.

    Parameters
    ----------
    data : list
        dataset to find information about

    Returns
    -------

    """
    data = listify(data)
    for i, datum in enumerate(data):
        print(str(i) + ": " + datum['metadata']['name']
                + " | " + datum['metadata']['subname'])


def search(data, key, values, verbose=False):
    r"""Searches dataset for data with matching metadata.

    Parameters
    ----------
    data : list
        dataset to search
    key : string
        key to search
    values : list
        values to match
    verbose : boolean
        DEFAULT = False
        determines if results printed to screen

    Returns
    -------
    output : list
        (indices of matches, matched data dictionary)
        (list of integers, list of data dictionaries)

    """
    values = listify(values)
    if verbose:
        print('Results with %s in %s...' % (key, values))
    matches = []
    match_data = []
    for i, datum in enumerate(data):
        if datum['metadata'][key] in set(values):
            matches.append(i)
            match_data.append(datum)
            if verbose:
                print(str(i) + ": " + datum['metadata']['name']
                        + " | " + datum['metadata']['subname'])
    output = matches, match_data
    return output


def last_modified(filename):
    r"""High level description.

    Parameters
    ----------
    filename : type
        Description

    Returns
    -------
    output : integer
        Description

    """
    t = os.path.getmtime(filename)
    output = int(t * 10**3)
    return output


def javaTime_to_pandasTime(javaTime):
    r"""High level description.

    Parameters
    ----------
    javaTime : type
        Description

    Returns
    -------
    pandasTime : type
        Description

    """
    pandasTime = pd.to_datetime(javaTime * 10**6)
    return pandasTime


def pandasTime_to_javaTime(pandasTime):
    r"""High level description.

    Parameters
    ----------
    pandasTime : type
        Description

    Returns
    -------
    javaTime : type
        Description

    """
    javaTime = pandasTime.value / 10**6
    return javaTime


def create_values(dates, values, positive=True):
    r"""High level description.

    Parameters
    ----------
    dates : type
        Description
    values : type
        Description
    positive : boolean
        Description

    Returns
    -------
    valueTuples : type
        Description

    """
    # convert to UNIX time in milliseconds (not seconds!)
    timestamps = map(pandasTime_to_javaTime, dates)
    #timestamps = map(int, dates.astype(np.int64) // 10**6)
    if positive:
        normalized = values / np.nanstd(values)
    else:
        normalized = (values - np.nanmean(values)) / np.nanstd(values)
    valueTuples = [list(z) for z in zip(timestamps, values, normalized)]
    #dateRange = [timestamps[0], timestamps[-1]]
    return valueTuples


def standard_to_json(datum, positive=True):
    r"""Converts time series datum from standard format to json format for
    values and times.

    Parameters
    ----------
    datum : dictionary
        time series to convert
    positive : boolean
        DEFAULT = True
        positivity of time series

    Returns
    -------
    d : dictionary
        time series in json format
        data dictionary with values and normalized values as floats and
        times as integers

    """
    d = copy.deepcopy(datum)
    dates = d['data']['times']
    values = d['data']['values']
    d['values'] = create_values(dates, values, positive=positive)
    d['metadata']['dateRange'] = map(pandasTime_to_javaTime,
            d['metadata']['dateRange'])
    d['metadata']['modified'] = pandasTime_to_javaTime(
            d['metadata']['modified'])
    del d['data']
    return d


def json_to_standard(datum):
    r"""Converts time series datum from json format to standard format (values
    as floats, times as Pandas timestamps).

    Parameters
    ----------
    datum : dictionary
        time series to convert

    Returns
    -------
    d : dictionary
        time series in standard format
        data dictionary with values as floats and times as Pandas
        timestamps

    """
    d = copy.deepcopy(datum)
    timestamps, values, normalized = map(list, zip(*d['values']))
    d['data'] = {}
    #d['data']['times'] = pd.DatetimeIndex([t * 10**6 for t in timestamps])
    d['data']['times'] = map(javaTime_to_pandasTime, timestamps)
    d['data']['values'] = values
    d['metadata']['dateRange'] = map(javaTime_to_pandasTime,
            d['metadata']['dateRange'])
    d['metadata']['modified'] = javaTime_to_pandasTime(
            d['metadata']['modified'])
    del d['values']
    return d


def standard_to_pandas(datum):
    r"""Converts time series datum from standard format to Pandas format (values
    and times given in a Pandas.Series object).

    Parameters
    ----------
    datum : dictionary
        time series in standard format

    Returns
    -------
    d : dictionary
        converted time series
        data dictionary with data a Pandas.Series object

    """
    d = copy.deepcopy(datum)
    d['data'] = pd.Series(data = datum['data']['values'],
            index = datum['data']['times'])
    return d


def pandas_to_standard(datum):
    r"""Converts time series datum from Pandas format to standard format (values
    as floats, times as Pandas timestamps).

    Parameters
    ----------
    datum : dictionary
        time series in Pandas format

    Returns
    -------
    d : dictionary
        converted time series
        data dictionary with values as floats and times as Pandas
        timestamps

    """
    d = copy.deepcopy(datum)
    d['data'] = {'values': map(float, datum['data'].values),
                 'times': list(datum['data'].index)}
    return d


def raw_to_pandas(raw_datum):
    r"""Converts time series datum from raw format to Pandas Series object.

    Parameters
    ----------
    raw_datum : dictionary
        time series in raw format (no metadata)
        dictionary of values as floats and times as Pandas timestamps

    Returns
    -------
    :return: converted time series
    :rtype: pandas.Series
    """
    return pd.Series(data = raw_datum['values'], index = raw_datum['times'])


def pandas_to_raw(series):
    r"""Converts Pandas Series object to raw format (values as floats, times as
    Pandas timestamps, no metadata).

    Parameters
    ----------
    :param series: time series in Pandas format

    :type series: pandas.Series

    :return: converted time series in raw format (no metadata)
    :rtype: dictionary of values as floats and times as Pandas timestamps
    """

    return {'values': map(float, series.values), 'times': list(series.index)}


# Methods that operate on raw datum
def sort_raw(raw_datum, *args, **kwargs):
    r"""Sorts raw datum.
    The arguments are the same as for the pandas.sort_by method.

    Parameters
    ----------

    """
    d = raw_to_pandas(raw_datum)
    d.sort_index(*args, **kwargs)

    return pandas_to_raw(d)


def normalize_raw(raw_datum, positive=True):
    r"""Normalizes raw datum to have mean 0 and standard deviation 1. NaN values
    are skipped by default when computing normalization.

    Parameters
    ----------
    :param raw_datum: time series in raw format (no metadata)

    :type raw_datum: dictionary of values as floats and times as Pandas
    timestamps

    :return: normalized time series in raw format
    :rtype: dictionary of values as floats and times as Pandas timestamps
    """
    d = raw_to_pandas(raw_datum)
    if positive:
        d = d / d.std()
    else:
        d = (d - d.mean()) / d.std()

    return pandas_to_raw(d)


def shift_raw(raw_datum, *args, **kwargs):
    r"""Shifts raw datum by specified length of time.
    The arguments are the same as for pandas.shift method.

    Parameters
    ----------
    :param raw_datum: time series in raw format (no metadata)

    :type raw_datum: dictionary of values as floats and times as Pandas
    timestamps

    :return: shifted time series in raw format
    :rtype: dictionary of values as floats and times as Pandas timestamps
    """
    d = raw_to_pandas(raw_datum)
    d = d.shift(*args, **kwargs)

    return pandas_to_raw(d)


def interpolate_raw(from_datum, to_datum, method='time', **kwargs):
    # TODO:
    # - If frequency of from_datum is less than that of to_datum, use
    #   interpolation as already given. However, if frequency of from_datum is
    #   greater than that of to_datum, smooth with kernel (i.e., integrate) and
    #   then interpolate as already given.
    # - Q: Implement above scheme as separate method that calls this existing
    #   one?
    r"""Interpolates raw datum to times of another raw datum.
    The arguments are the same as for pandas.interpolate method.

    Parameters
    ----------
    :param from_datum: time series in raw format (no metadata)
    :param to_datum: time series in raw format to interpolate to

    :type from_datum, to_datum: dictionary of values as floats and times as Pandas
    timestamps

    :return: interpolated time series in raw format
    :rtype: dictionary of values as floats and times as Pandas timestamps
    """

    d1, d2 = raw_to_pandas(from_datum), raw_to_pandas(to_datum)

    ## Resampling code to put d1 on same frequency as d2
    #to_freq = d2.index.inferred_freq
    #if to_freq:
    #    d1 = d1.resample(to_freq, how=np.sum) #or how=np.mean

    index_diff = d2.index.difference(d1.index)
    x = pd.Series(index=index_diff).add(d1, fill_value=0)
    d = x.interpolate(method=method, **kwargs)[d2.index]

    return pandas_to_raw(d)


def fillna_raw(raw_datum, *args, **kwargs):
    r"""Fills NaN values with specified values.
    The arguments are the same as for pandas.fillna method.

    Parameters
    ----------
    :param raw_datum: time series in raw format (no metadata)

    :type raw_datum: dictionary of values as floats and times as Pandas
    timestamps

    :return: filled time series in raw format
    :rtype: dictionary of values as floats and times as Pandas timestamps
    """
    d = raw_to_pandas(raw_datum)
    d = d.fillna(*args, **kwargs)

    return pandas_to_raw(d)


def derivative_raw(raw_datum, order=1, method='backward', scaling='w'):
    r"""High level description.

    Parameters
    ----------
    raw_datum : type
        Description
    order : integer
        DEFAULT = 1
        Description
    method : string
        DEFAULT = 'backward'
        Description
    scaling : string
        DEFAULT = 'w'
        Description
    """
    d = raw_datum

    if order == 0:
        return d

    # Determine number of seconds in one time period
    t0 = pd.Timestamp('1679-01-01 00:00:00')
    t1 = pd.DatetimeIndex([t0]).shift(1, scaling)[0]
    timescaling = (t1-t0).total_seconds()

    derivative = [(d['values'][i] - d['values'][i-1]) * timescaling
                    / (d['times'][i] - d['times'][i-1]).total_seconds()
                    for i in range(1,len(d['values']))]

    if method == 'backward':
        derivative = [float('nan')] + derivative
    else:
        derivative = derivative + [float('nan')]

    d_prime = {'values': derivative, 'times': d['times']}

    if order > 1:
        return derivative_raw(d_prime, order=order-1, method=method,
                scaling=scaling)

    return d_prime


def length_raw(raw_datum):
    r"""High level description.

    Parameters
    ----------
    raw_datum : type
        Description

    Returns
    -------
    output : integer
        Description

    """
    d = raw_datum
    output = len(d['values'])
    return output


# Methods to manipulate data in standard format
def sort(datum):
    r"""High level description.

    Parameters
    ----------
    datum : type
        Description

    Returns
    -------
    output : type
        Description

    """
    d = copy.deepcopy(datum)
    d['data'] = sort_raw(datum['data'])
    return d


def normalize(datum, positive=True):
    r"""High level description.

    Parameters
    ----------
    datum : type
        Description
    positive : boolean
        DEFAULT = True
        Description

    Returns
    -------
    d : type
        Description

    """
    d = copy.deepcopy(datum)
    d['data'] = normalize_raw(datum['data'], positive)
    return d


def interpolate(from_datum, to_datum, method='time', **kwargs):
    r"""High level description.

    Parameters
    ----------
    from_datum : type
        Description
    to_datum : type
        Description
    method : string
        DEFAULT = 'time'
        Description
    kwargs : dictionary
        Description

    Returns
    -------
    d : type
        Description

    """
    d = copy.deepcopy(from_datum)
    d['data'] = interpolate_raw(from_datum['data'], to_datum['data'],
            method=method, **kwargs)
    return d

def fillna(from_datum, *args, **kwargs):
    r"""High level description.

    Parameters
    ----------
    from_datum : type
        Description
    args : type
        Description
    kwargs : type
        Description

    Returns
    -------
    d : type
        Description

    """
    d = copy.deepcopy(from_datum)
    d['data'] = fillna_raw(from_datum['data'], *args, **kwargs)
    return d


def derivative(datum, order=1, method='backward', scaling='w'):
    r"""High level description.

    Parameters
    ----------
    datum : type
        Description
    order : integer
        DEFAULT = 1
        Description
    method : string
        DEFAULT = 'backward'
        Description
    scaling : string
        DEFAULT = 'w'
        Description

    Returns
    -------
    d : type
        Description

    """
    d = copy.deepcopy(datum)
    d['data'] = derivative_raw(datum['data'], order=order, method=method,
            scaling=scaling)
    return d


def length(datum):
    r"""High level description.

    Parameters
    ----------
    datum : type
        Description

    Returns
    -------
    output : type
        Description

    """
    output = length_raw(datum['data'])
    return output


#def attach_lags(datum, embedding_dim=10):
#    """
#    """
#    d = copy.deepcopy(datum)
#    v = [0.0]*(embedding_dim - 1) + d['data']
#    lagged_data = [v[i-embedding_dim:i] for i in range(1,len(d['data'])+1)]
#    d['data']['lagged'] = [list(l) for l in zip(*lagged_data)]
#
#    return d
#
#def attach_derivatives(datum, order=2):
#    """
#    """
#    pass
