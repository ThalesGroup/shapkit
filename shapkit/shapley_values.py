# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/shapley_values.ipynb (unless otherwise specified).

__all__ = ['ShapleyValues']

# Cell
import numpy as np
import pandas as pd
from itertools import combinations
from math import factorial
from tqdm import tqdm

# Cell
def ShapleyValues(x, fc, ref):
    """
    Calculate the exact Shapley Values for an individual x
    in a game based on a reference r and the reward function fc.
    """

    # Get general information
    feature_names = list(x.index)
    d = len(feature_names) # dimension
    set_features = set(feature_names)

    # Store Shapley Values in a pandas Series
    Φ = pd.Series(np.zeros(d), index=feature_names)

    # Individual reference or dataset of references
    def output_single_ref(coalition, feature_names):
        z = np.array([x[col] if col in coalition else ref.loc[col] for col in feature_names])
        return fc(z)

    def output_several_ref(coalition, feature_names):
        rewards = []
        idxs = np.random.choice(ref.index, size=len(ref), replace=False)
        for idx in idxs:
            z = np.array([x[col] if col in coalition else ref.loc[idx, col] for col in feature_names])
            rewards.append(fc(z))
        return np.mean(rewards)

    if isinstance(ref, pd.core.series.Series):
        individual_ref = True
        output = output_single_ref
    elif isinstance(ref, pd.core.frame.DataFrame):
        if ref.shape[0] == 1:
            ref = ref.iloc[0]
            individual_ref = True
            output = output_single_ref
        else:
            individual_ref = False
            output = output_several_ref

    # Start computation (number of coalitions: 2**d - 1)
    for cardinal_S in tqdm(range(0, d)):
        # weight
        ω = factorial(cardinal_S) * (factorial(d - cardinal_S - 1))
        ω /= factorial(d)
        # iter over all combinations of size cardinal_S
        for S in combinations(feature_names, cardinal_S):
            S = list(S)
            f_S = output(S, feature_names)
            # Consider only features outside of S
            features_out_S = set_features - set(S)
            for j in features_out_S:
                S_union_j = S + [j]
                f_S_union_j = output(S_union_j, feature_names)
                # Update Shapley value of attribute i
                Φ[j] += ω * (f_S_union_j - f_S)

    return Φ