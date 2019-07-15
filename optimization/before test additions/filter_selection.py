import copy
import numpy as np
import pandas as pd
from itertools import compress

# Methods on raw data
def filter_on_times_raw(raw_datum, after='1678-01-01', before='2261-12-31'):
    """Creates boolean mask on datum in raw format to select over interval of
    times.
    """
    # Force to pandas Timestamp to allow user input as strings
    after = pd.Timestamp(after)
    before = pd.Timestamp(before)
    f = map(lambda t: after <= t < before, raw_datum['times'])

    return f

def filter_on_values_raw(raw_datum, above=-np.inf, below=np.inf):
    """Creates boolean mask on datum in raw format to select over range of
    values.
    """
    f = map(lambda v: above <= v <= below, raw_datum['values'])

    return f

def select_raw(raw_datum, bool_mask):
    """Selects data in raw format according to boolean mask.
    """
    #d = copy.deepcopy(raw_datum)
    d = dict(raw_datum)
    for key in d.keys():
        d[key] = list(compress(raw_datum[key], bool_mask))

    return d

# Methods on standard data
def filter_on_times(datum, after='1678-01-01', before='2261-12-31'):
    """Creates filter on datum in standard format to select over interval of
    times.
    """

    return filter_on_times_raw(datum['data'], after, before)

def filter_on_values(datum, above=-np.inf, below=np.inf):
    """Creates filter on datum in standard format to select over range of
    values.
    """

    return filter_on_values_raw(datum['data'], above, below)

def select(datum, bool_mask):
    """Select datum in standard format, including according to boolean mask.
    """
    #d = copy.deepcopy(datum)
    d = dict(datum)
    d['data'] = select_raw(datum['data'], bool_mask)

    return d

def exclude(datum, bool_mask):
    """Select datum in standard format, excluding according to boolean mask.
    """
    include = [not b for b in bool_mask]

    return select(datum, include)

# Methods on boolean filters
def merge_filters(bool_mask_list, method='any'):
    """Merges multiple filters based on choice of merging method.

    Parameters
    ----------

    """
    transposed_list = np.array(bool_mask_list).T.tolist()
    f_merged = [eval(method)(l) for l in transposed_list]

    return f_merged
