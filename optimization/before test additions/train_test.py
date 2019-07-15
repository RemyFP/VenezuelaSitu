import numpy as np
import filter_selection as fs
import data_manipulation as dm
# from sklearn.cross_validation import KFold
from sklearn.model_selection import KFold


# function of convert list of indices to boolean mask
def index_to_filter(index, mask_length=0):
    """
    """
    if mask_length < np.max(index) + 1:
        mask_length = np.max(index) + 1
    f = [False]*mask_length
    for i in index:
        f[i] = True
    return f

# generalized functions for training/testing given boolean mask
def train_on_filter(train_func, goal_datum, data_sources, bool_mask):
    """
    """
    goal_datum_resample = fs.select(goal_datum, bool_mask)
    data_sources_resample = [fs.select(d, bool_mask) for d in data_sources]
    return train_func(goal_datum_resample, data_sources_resample)

def test_on_filter(test_func, data_sources, bool_mask):
    """
    """
    data_sources_resample = [fs.select(d, bool_mask) for d in data_sources]
    return test_func(data_sources_resample)


### NOTE: ATTEMPTED MODULARIZATION OF CROSS-VALIDATION, DO NOT USE YET
def cross_validate(train_func, test_func, goal_datum, data_sources, num_folds=1):
    # kf = KFold(dm.length(goal_datum), n_folds)
    kf_p = KFold(n_folds)
    kf = list(kf_p.split(range(dm.length(goal_datum))))

    pred_series_CV_list = []
    for train, test in kf:
        train_f, test_f = tt.index_to_filter(train), tt.index_to_filter(test)
        params = tt.train_on_filter(train_func, goal_datum, data_sources, train_f)
        pred_series_CV = tt.test_on_filter(partial(test_func, train_params=params), data_sources, test_f)
        pred_series_CV_list.append(pred_series_CV)
    pred_series_CV = np.concatenate(pred_series_CV_list, axis=0)
    return 0
