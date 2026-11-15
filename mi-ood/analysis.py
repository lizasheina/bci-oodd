
import numpy as np
import pandas as pd
import pingouin as pg
from scipy.stats import bootstrap, permutation_test

MI_METRICS = ("MI_BA", "MI_AUROC")


def rank_profiles(x):
    return x.rank().T.groupby(level="metric").mean().T

def adjusted_rho(x, ood_metric="OOD_AUROC"):
    z = pd.get_dummies(rank_profiles(x).reset_index(), columns=["dataset"], drop_first=True, dtype=float)
    return np.array([pg.partial_corr(data=z, x=m, y=ood_metric, covar=z.filter(like="dataset_").columns.tolist(), method="spearman")["r"].iloc[0] for m in MI_METRICS])

def weighted_rho(x, ood_metric="OOD_AUROC"):
    parts = [g for _, g in x.groupby(level="dataset")]
    return np.average([rank_profiles(g).corr(method="spearman").loc[list(MI_METRICS), ood_metric] for g in parts], axis=0, weights=[len(g) for g in parts])

def bootstrap_ci(x, n=4999, ood_metric="OOD_AUROC"):
    groups = tuple(x.groupby(level="dataset").indices.values())
    return bootstrap(groups, lambda *idx: adjusted_rho(x.iloc[np.concatenate(idx)], ood_metric=ood_metric), vectorized=False, n_resamples=n, method="percentile", rng=42).confidence_interval

def permutation_p(x, n=9999, statistic=adjusted_rho, ood_metric="OOD_AUROC"):
    rng = np.random.default_rng(42)
    groups = list(x.groupby(level="dataset").indices.values())
    null = []
    for _ in range(n):
        order = np.arange(len(x))
        for idx in groups:
            order[idx] = rng.permutation(idx)
        shuffled = x.copy()
        shuffled.loc[:, ood_metric] = x.iloc[order][ood_metric].to_numpy()
        null.append(statistic(shuffled, ood_metric=ood_metric))
    return (1 + (np.abs(null) >= np.abs(statistic(x, ood_metric=ood_metric)) - 1e-12).sum(axis=0)) / (n + 1)

def overall_agreement(x, n=9999):
    rng = np.random.default_rng(42)
    effects, nulls, weights = [], [], []
    for _, group in x.groupby(level="dataset"):
        ranks = group.rank().to_numpy().T - (len(group) + 1) / 2
        scale = len(ranks) * np.square(ranks).sum()
        test = permutation_test(tuple(ranks), lambda *r, axis: np.square(np.sum(r, axis=0)).sum(axis=axis) / scale, permutation_type="pairings", alternative="greater", n_resamples=n, batch=256, rng=rng)
        effects.append(test.statistic)
        nulls.append(test.null_distribution)
        weights.append(len(group))
    w = np.average(effects, weights=weights)
    null = np.average(nulls, axis=0, weights=weights)
    return {"n": sum(weights), "W": w, "p_perm": (1 + (null >= w - 1e-12).sum()) / (len(null) + 1)}
