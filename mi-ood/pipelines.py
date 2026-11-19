from methods import *
from mulder_models import MulderMC

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
    "ACM+TS+shLDA": make_pipeline(
        AugmentedDataset(order=2, lag=9),
        Covariances(estimator="cov"),
        TangentSpace(metric="riemann"),
        StandardScaler(),
        LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    ),
    "FBCSP+ElasticNet": make_pipeline(
        FunctionTransformer(filter_bands),
        FilterBank(make_pipeline(Covariances(estimator="oas"), CSP(nfilter=4, metric="euclid", log=True))),
        LogisticRegression(solver="saga", l1_ratio=.5, C=1., max_iter=5000, random_state=42)
    ),
    "FBCSP+shLDA": make_pipeline(
        FunctionTransformer(filter_bands),
        FilterBank(make_pipeline(Covariances(estimator="oas"), CSP(nfilter=4, metric="euclid", log=True))),
        Normalizer(),
        LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    ),
    "FBCSP+Vote": make_pipeline(
        FunctionTransformer(filter_bands),
        FilterBank(make_pipeline(Covariances(estimator="oas"), CSP(nfilter=4, metric="euclid", log=True))),
        FeatureVote(n_models=11, feature_fraction=0.5, random_state=42),
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
    "MultiModel+SoftVote": MultiModel(mode="softvote"),
    "MultiModel+SoftVote(LDA)": MultiModel(mode="softvote", classifier="lda"),
    "MultiModel+SoftVote(SVM)": MultiModel(mode="softvote", classifier="svm"),
    "MultiModel+SoftVote(ExtraTrees)": MultiModel(mode="softvote", classifier="extra_trees"),
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
    "ACM+TS+kNN": make_pipeline(
        AugmentedDataset(order=2, lag=9),
        Covariances(estimator="cov"),
        TangentSpace(metric="riemann"),
        StandardScaler(),
        KNNDistance(k=10),
    ),
    "ACM+TS+Mahalanobis": make_pipeline(
        AugmentedDataset(order=2, lag=9),
        Covariances(estimator="cov"),
        TangentSpace(metric="riemann"),
        StandardScaler(),
        ClassMahalanobis()
    ),
    "ACM+TS+LOF": make_pipeline(
        AugmentedDataset(order=2, lag=9),
        Covariances(estimator="cov"),
        TangentSpace(metric="riemann"),
        StandardScaler(),
        LocalOutlierFactor(n_neighbors=20, novelty=True, n_jobs=1),
    ),
    "ACM+TS+IsolationForest": make_pipeline(
        AugmentedDataset(order=2, lag=9),
        Covariances(estimator="cov"),
        TangentSpace(metric="riemann"),
        IsolationForest(n_estimators=200, random_state=42, n_jobs=1),
    ),
    "FBCSP+IsolationForest": make_pipeline(
        FunctionTransformer(filter_bands),
        FilterBank(make_pipeline(Covariances(estimator="oas"), CSP(nfilter=4, metric="euclid", log=True))),
        Normalizer(),
        IsolationForest(n_estimators=200, random_state=42, n_jobs=1),
    ),
    "FBCSP(nonorm)+IsolationForest": make_pipeline(
        FunctionTransformer(filter_bands),
        FilterBank(make_pipeline(Covariances(estimator="oas"), CSP(nfilter=4, metric="euclid", log=True))),
        IsolationForest(n_estimators=200, random_state=42, n_jobs=1),
    ),
    "FBCSP+kNN": make_pipeline(
        FunctionTransformer(filter_bands),
        FilterBank(make_pipeline(Covariances(estimator="oas"), CSP(nfilter=4, metric="euclid", log=True))),
        Normalizer(),
        KNNDistance(k=10),
    ),
    "FBCSP+LOF": make_pipeline(
        FunctionTransformer(filter_bands),
        FilterBank(make_pipeline(Covariances(estimator="oas"), CSP(nfilter=4, metric="euclid", log=True))),
        Normalizer(),
        LocalOutlierFactor(n_neighbors=20, novelty=True, n_jobs=1),
    ),
    "FBCSP+OCSVM": make_pipeline(
        FunctionTransformer(filter_bands),
        FilterBank(make_pipeline(Covariances(estimator="oas"), CSP(nfilter=4, metric="euclid", log=True))),
        StandardScaler(),
        OneClassSVM(kernel="rbf", gamma="scale", nu=0.1),
    ),
    "FBCSP+Mahalanobis": make_pipeline(
        FunctionTransformer(filter_bands),
        FilterBank(make_pipeline(Covariances(estimator="oas"), CSP(nfilter=4, metric="euclid", log=True))),
        Normalizer(),
        ClassMahalanobis(),
    ),
    "FBCSP+Vote": make_pipeline(
        FunctionTransformer(filter_bands),
        FilterBank(make_pipeline(Covariances(estimator="oas"), CSP(nfilter=4, metric="euclid", log=True))),
        FeatureVote(n_models=11, feature_fraction=0.5, random_state=42),
    ),
    "FBCSP+Vote(entropy)": make_pipeline(
        FunctionTransformer(filter_bands),
        FilterBank(make_pipeline(Covariances(estimator="oas"), CSP(nfilter=4, metric="euclid", log=True))),
        FeatureVote(n_models=11, feature_fraction=0.5, use_entropy=True, random_state=42),
    ),
    "FBCSP+Vote(LDA)": make_pipeline(
        FunctionTransformer(filter_bands),
        FilterBank(make_pipeline(Covariances(estimator="oas"), CSP(nfilter=4, metric="euclid", log=True))),
        FeatureVote(classifier="lda", n_models=11),
    ),
    "FBCSP+Vote(SVM)": make_pipeline(
        FunctionTransformer(filter_bands),
        FilterBank(make_pipeline(Covariances(estimator="oas"), CSP(nfilter=4, metric="euclid", log=True))),
        FeatureVote(classifier="svm", n_models=11),
    ),
    "FBCSP+Vote(ExtraTrees)": make_pipeline(
        FunctionTransformer(filter_bands),
        FilterBank(make_pipeline(Covariances(estimator="oas"), CSP(nfilter=4, metric="euclid", log=True))),
        FeatureVote(classifier="extra_trees", n_models=11),
    ),
    "FBCSP+Vote(LR+LDA+SVM)": make_pipeline(
        FunctionTransformer(filter_bands),
        FilterBank(make_pipeline(Covariances(estimator="oas"), CSP(nfilter=4, metric="euclid", log=True))),
        FeatureVote(ensemble={"lr": 4, "lda": 4, "svm": 3}),
    ),
    "Log-bandpower+Mahalanobis": make_pipeline(
        FunctionTransformer(log_bandpower), 
        StandardScaler(), 
        ClassMahalanobis()
    ),
    "Log-bandpower+kNN": make_pipeline(
        FunctionTransformer(log_bandpower), 
        StandardScaler(), 
        KNNDistance(k=10),
    ),
    "Log-bandpower+LOF": make_pipeline(
        FunctionTransformer(log_bandpower),
        StandardScaler(),
        LocalOutlierFactor(n_neighbors=20, novelty=True, n_jobs=1),
    ),
    "Log-bandpower+OCSVM": make_pipeline(
        FunctionTransformer(log_bandpower),
        StandardScaler(),
        OneClassSVM(kernel="rbf", gamma="scale", nu=0.1),
    ),
    "Log-bandpower+IsolationForest": make_pipeline(
        FunctionTransformer(log_bandpower),
        IsolationForest(n_estimators=200, random_state=42, n_jobs=1),
    ),
    "Microstates+LOF": make_pipeline(
        MicrostateFeatures(), 
        StandardScaler(),
        LocalOutlierFactor(n_neighbors=20, novelty=True, n_jobs=1),
    ),
    "Microstates+kNN": make_pipeline(
        MicrostateFeatures(),
        StandardScaler(),
        KNNDistance(k=10),
    ),
    "Microstates+OCSVM": make_pipeline(
        MicrostateFeatures(),
        StandardScaler(),
        OneClassSVM(kernel="rbf", gamma="scale", nu=0.1),
    ),
    "Microstates+Mahalanobis": make_pipeline(
        MicrostateFeatures(),
        StandardScaler(),
        ClassMahalanobis(),
    ),
    "Microstates+IsolationForest": make_pipeline(
        MicrostateFeatures(),
        IsolationForest(n_estimators=200, random_state=42, n_jobs=1),
    ),
    "VAR+Mahalanobis": make_pipeline(
        FunctionTransformer(var_features, kw_args={"order": 2}), 
        StandardScaler(),
        ClassMahalanobis()
    ),
    "VAR+kNN": make_pipeline(
        FunctionTransformer(var_features, kw_args={"order": 2}), 
        StandardScaler(),
        KNNDistance(k=10),
    ),
    "VAR+LOF": make_pipeline(
        FunctionTransformer(var_features, kw_args={"order": 2}),
        StandardScaler(),
        LocalOutlierFactor(n_neighbors=20, novelty=True, n_jobs=1),
    ),
    "VAR+OCSVM": make_pipeline(
        FunctionTransformer(var_features, kw_args={"order": 2}),
        StandardScaler(),
        OneClassSVM(kernel="rbf", gamma="scale", nu=0.1),
    ),
    "VAR+IsolationForest": make_pipeline(
        FunctionTransformer(var_features, kw_args={"order": 2}),
        IsolationForest(n_estimators=200, random_state=42, n_jobs=1),
    ),
    "ShallowConvNet+kNN": ShallowOOD(method="knn", n_neighbors=10, device=device),
    "ShallowConvNet300+kNN": ShallowOOD(epochs=300, method="knn", n_neighbors=10, device=device),
    "EEGNet+TempDens": EEGNetTempDens( epochs=100, device=device, threads=2 ),
    "EEGNet300+TempDens": EEGNetTempDens( epochs=300, device=device, threads=2 ),
    "EEGNex+MCdropout": MulderMC(epochs=200, batch_size=32, n_mc=50, device=device),
    "MultiModel+SoftVote": MultiModel(mode="softvote"),
    "MultiModel+SoftVote(LDA)": MultiModel(mode="softvote", classifier="lda"),
    "MultiModel+SoftVote(SVM)": MultiModel(mode="softvote", classifier="svm"),
    "MultiModel+SoftVote(ExtraTrees)": MultiModel(mode="softvote", classifier="extra_trees"),
    "MultiModel+LOF": MultiModel(mode="lof", n_neighbors=20),
    "MultiModel+OCSVM": MultiModel(mode="ocsvm", nu=0.1, gamma="scale"),
    "MultiModel+LOF(top2)": MultiModel(mode="lof", agg="top2"),
    "MultiModel+OCSVM(top2)": MultiModel(mode="ocsvm", agg="top2"),
    "MultiModel+Mahalanobis": MultiModel(mode="mahalanobis"),
    "MultiModel+Mahalanobis(top2)": MultiModel(mode="mahalanobis", agg="top2"),
}

