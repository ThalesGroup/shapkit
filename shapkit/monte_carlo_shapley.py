# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/monte_carlo_shapley.ipynb (unless otherwise specified).

__all__ = ['MonteCarloShapley']

# Cell
import numpy as np
import pandas as pd
from tqdm import tqdm

# Cell
def MonteCarloShapley(x, fc, ref, n_iter, callback=None):
    """
    Estimate the Shapley Values using an optimized Monte Carlo version.
    """

    # Get general information
    feature_names = list(x.index)
    d = len(feature_names) # dimension

    # Individual reference or dataset of references
    if isinstance(ref, pd.core.series.Series):
        individual_ref = True
        f_r = fc(ref.values)
    elif isinstance(ref, pd.core.frame.DataFrame):
        if ref.shape[0] == 1:
            ref = ref.iloc[0]
            individual_ref = True
            f_r = fc(ref.values)
        else:
            individual_ref = False
            n_ref = len(ref)

    if individual_ref:
        # If x[j] = r[j] => Φ[j] = 0 and we can reduce the dimension
        distinct_feature_names = list(x[x!=ref].index)
        if set(distinct_feature_names) == set(feature_names):
            distinct_feature_names = feature_names
            sub_d = d
            x_cp = x.copy()
            r_cp = ref.copy()
            reward = lambda z: fc(z)
            pass
        else:
            sub_d = len(distinct_feature_names) # new dimension
            x_cp = x[distinct_feature_names].copy()
            r_cp = ref[distinct_feature_names].copy()
            print("new dimension {0}".format(sub_d))
            def reward(z):
                z_tmp = ref.copy()
                z_tmp[distinct_feature_names] = z
                return fc(z_tmp.values)
    else:
        distinct_feature_names = feature_names
        sub_d = d
        x_cp = x.copy()
        reward = lambda z: fc(z)

    # Store all Shapley Values in a numpy array
    Φ_storage = np.empty((n_iter, sub_d))

    # Monte Carlo loop
    for m in tqdm(range(1, n_iter+1)):
        # Sample a random permutation order
        o = np.random.permutation(sub_d)
        # initiate useful variables for this iteration
        # if several references select at random one new ref at each iter
        if individual_ref:
            f_less_j = f_r
            x_plus_j = r_cp.values.copy()
        else:
            r_cp = ref.values[np.random.choice(n_ref, size=1)[0],:].copy()
            f_less_j = fc(r_cp)
            x_plus_j = r_cp.copy()
        # iterate through the permutation of features
        for j in o:
            x_plus_j[j] = x_cp.values[j]
            f_plus_j = reward(x_plus_j)
            # update Φ
            Φ_j = f_plus_j - f_less_j
            Φ_storage[m-1,j] = Φ_j
            # reassign f_less_j
            f_less_j = f_plus_j
        if callback:
            Φ = pd.Series(np.mean(Φ_storage[:m,:],axis=0), index=feature_names)
            callback(Φ)

    Φ_mean = np.mean(Φ_storage,axis=0)
    Φ = pd.Series(np.zeros(d), index=feature_names)
    Φ[distinct_feature_names] = Φ_mean
    return Φ