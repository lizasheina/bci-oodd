from methods import *

from functools import partial

from moabb.pipelines.features import AugmentedDataset
from moabb.pipelines.utils import FilterBank

from pyriemann.estimation import Covariances
from pyriemann.spatialfilters import CSP
from pyriemann.tangentspace import TangentSpace

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import HistGradientBoostingClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import LocalOutlierFactor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, Normalizer, StandardScaler
from sklearn.svm import SVC, OneClassSVM

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

mi_pipelines = {
    "ACM+TS+SVM": make_pipeline(
        AugmentedDataset(order=2, lag=9),
        Covariances(estimator="cov"),
        TangentSpace(metric="riemann"),
        SVC(kernel="rbf", C=1.0),
    ),
    "FBCSP+ElasticNet": make_pipeline(
        FunctionTransformer(filter_bands),
        FilterBank(make_pipeline(Covariances(estimator="oas"), CSP(nfilter=4, metric="euclid", log=True))),
        LogisticRegression(solver="saga", l1_ratio=.5, C=1., max_iter=5000, random_state=42)
    ),
    "Log-bandpower+shLDA": make_pipeline(
        FunctionTransformer(log_bandpower),
        Normalizer(),
        LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"),
    ),
    "Microstates+GB": make_pipeline(
        MicrostateFeatures(), 
        HistGradientBoostingClassifier(learning_rate=.05, max_iter=200, max_leaf_nodes=7, min_samples_leaf=10, l2_regularization=1., early_stopping=False, random_state=42)
    ),
    "VAR+shLDA": make_pipeline(
        FunctionTransformer(var_features, kw_args={"order": 2}), 
        StandardScaler(),
        LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    ),
    "ShallowConvNet": ShallowConvNet(epochs=300, device=device),
    "EEGNet": EEGNetClassifier(epochs=300, device=device),
}

HANDS = ("left_hand", "right_hand")


ood_pipelines = {
    "ACM+TS+OCSVM": make_pipeline(
        AugmentedDataset(order=2, lag=9),
        Covariances(estimator="cov"),
        TangentSpace(metric="riemann"),
        StandardScaler(),
        OneClassSVM(kernel="rbf", gamma="scale", nu=0.1),
    ),
    "FBCSP+IsolationForest": make_pipeline(
        FunctionTransformer(filter_bands),
        FilterBank(make_pipeline(Covariances(estimator="oas"), CSP(nfilter=4, metric="euclid", log=True))),
        Normalizer(),
        IsolationForest(n_estimators=200, random_state=42, n_jobs=1),
    ),
    "Log-bandpower+Mahalanobis": make_pipeline(
        FunctionTransformer(log_bandpower), 
        StandardScaler(), 
        ClassMahalanobis()
    ),
    "Microstates+LOF": make_pipeline(
        MicrostateFeatures(), 
        StandardScaler(),
        LocalOutlierFactor(n_neighbors=20, novelty=True, n_jobs=1),
    ),
    "VAR+Mahalanobis": make_pipeline(
        FunctionTransformer(var_features, kw_args={"order": 2}), 
        StandardScaler(),
        ClassMahalanobis()
    ),
    "ShallowConvNet+kNN": ShallowOOD(method="knn", n_neighbors=10, device=device),
    "EEGNet+TempDens": EEGNetTempDens( epochs=100, device=device, threads=2 ),
}

