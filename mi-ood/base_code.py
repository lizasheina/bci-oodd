import moabb.datasets as datasets

from moabb.paradigms import MotorImagery
from sklearn.model_selection import train_test_split

from moabb.pipelines.features import AugmentedDataset
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC

from pathlib import Path
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import balanced_accuracy_score

import random

import numpy as np
from functools import partial
from scipy.signal import butter, sosfiltfilt
from moabb.pipelines.utils import FilterBank
from pyriemann.spatialfilters import CSP
from sklearn.preprocessing import FunctionTransformer
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import HistGradientBoostingClassifier

import torch
from sklearn.base import BaseEstimator, ClassifierMixin
from braindecode import EEGClassifier
from braindecode.models import ShallowFBCSPNet
from braindecode.util import set_random_seeds

import mne

from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.covariance import LedoitWolf
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import roc_auc_score




mne.set_log_level("WARNING")

def filter_bands(X):
    bands = [(8, 12), (12, 16), (16, 20), (20, 24), (24, 28), (28, 32)]
    filters = [butter(4, band, btype="bandpass", fs=250, output="sos") for band in bands]
    return np.stack([sosfiltfilt(sos, X, axis=-1) for sos in filters], axis=-1)


def log_bandpower(X):
    X = filter_bands(X)
    power = np.mean(X ** 2, axis=2)
    return np.log(np.maximum(power, 1e-12)).reshape(len(X), -1)

class ShallowConvNet(ClassifierMixin, BaseEstimator):
    def __init__(self, epochs=100, device="cpu"):
        self.epochs, self.device = epochs, device

    def _prepare(self, X):
        return ((X - self.mean_) / self.std_).astype(np.float32)

    def fit(self, X, y):
        set_random_seeds(seed=42, cuda=self.device.startswith("cuda"))
        self.classes_, y = np.unique(y, return_inverse=True)
        self.mean_ = X.mean(axis=(0, 2), keepdims=True)
        self.std_ = np.maximum(X.std(axis=(0, 2), keepdims=True), 1e-6)
        self.net_ = EEGClassifier(
            ShallowFBCSPNet, module__n_chans=X.shape[1], module__n_times=X.shape[2],
            module__n_outputs=len(self.classes_), 
            optimizer=torch.optim.Adam,
            lr=0.001, batch_size=64, max_epochs=self.epochs, train_split=None,
            iterator_train__drop_last=False, device=self.device, verbose=0,
        )
        self.net_.fit(self._prepare(X), y.astype(np.int64))
        return self

    def predict(self, X):
        return self.classes_[self.net_.predict(self._prepare(X))]


device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

pipelines = {
    "ACM+TS+SVM": make_pipeline(
        AugmentedDataset(order=2, lag=9),
        Covariances(estimator="oas"),
        TangentSpace(metric="riemann"),
        SVC(kernel="rbf", C=1.0),
    ),
    "ACM(cov)+TS+SVM": make_pipeline(
        AugmentedDataset(order=2, lag=9),
        Covariances(estimator="cov"),
        TangentSpace(metric="riemann"),
        SVC(kernel="rbf", C=1.0),
    ),
    "FBCSP+shrinkLDA": make_pipeline(
        FunctionTransformer(filter_bands),
        FilterBank(make_pipeline(Covariances(estimator="oas"), CSP(nfilter=4, metric="euclid", log=True))),
        SelectKBest(partial(mutual_info_classif, random_state=42), k=10),
        LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"),
    ),
    "Log-bandpower+GB": make_pipeline(
        FunctionTransformer(log_bandpower),
        HistGradientBoostingClassifier(
            learning_rate=0.05, max_iter=200, max_leaf_nodes=7,
            min_samples_leaf=10, l2_regularization=1.0,
            early_stopping=False, random_state=42 )
    ),
    "ShallowConvNet": ShallowConvNet(epochs=100, device=device),
}

HANDS = ("left_hand", "right_hand")


class ClassMahalanobis(BaseEstimator):
    def fit(self, X, y):
        self.classes_, labels = np.unique(y, return_inverse=True)
        self.means_ = np.array([ X[labels == c].mean(axis=0) for c in range(len(self.classes_)) ])
        self.covariance_ = LedoitWolf(assume_centered=True).fit( X - self.means_[labels] )
        return self

    def score_samples(self, X):
        distances = [ self.covariance_.mahalanobis(X - mean) for mean in self.means_ ]
        return -np.min(distances, axis=0)


class ShallowOOD(BaseEstimator):
    def __init__(self, method="energy", epochs=100, device="cpu", n_neighbors=10, n_mc=30):
        self.method = method
        self.epochs = epochs
        self.device = device
        self.n_neighbors = n_neighbors
        self.n_mc = n_mc

    def fit(self, X, y, classifier=None):
        self.classifier_ = classifier

        if self.classifier_ is None:
            self.classifier_ = ShallowConvNet(epochs=self.epochs, device=self.device).fit(X, y)

        if self.method == "knn":
            features = normalize(self._forward(X, features=True).numpy())
            self.neighbors_ = NearestNeighbors(n_neighbors=min(self.n_neighbors, len(X))).fit(features)

        return self

    def _forward(self, X, features=False, mc=False):
        net = self.classifier_.net_.module_
        states = [ (module, module.training) for module in net.modules() ]
        outputs = []

        try:
            net.eval()

            # Включаем только Dropout, оставляя BatchNorm в eval.
            if mc:
                for module in net.modules():
                    if isinstance(module, torch.nn.Dropout):
                        module.train()

            prepared = self.classifier_._prepare(X)
            device = next(net.parameters()).device

            with torch.inference_mode():
                for start in range(0, len(prepared), 64):
                    batch = torch.as_tensor(prepared[start:start + 64], device=device)

                    if features:
                        for layer in list(net.children())[:-1]:
                            batch = layer(batch)
                        output = batch.flatten(start_dim=1)
                    else:
                        output = net(batch)

                    outputs.append(output.cpu())

            return torch.cat(outputs)

        finally:
            for module, training in states:
                module.training = training

    def score_samples(self, X):
        if self.method == "knn":
            features = normalize(self._forward(X, features=True).numpy())
            score = self.neighbors_.kneighbors(features)[0][:, -1]

        elif self.method == "mc_dropout":
            probabilities = torch.stack([ self._forward(X, mc=True).softmax(dim=1) for _ in range(self.n_mc) ])

            def entropy(p):
                return -(p * p.clamp_min(1e-8).log()).sum(dim=-1)

            score = (entropy(probabilities.mean(dim=0)) - entropy(probabilities).mean(dim=0)).clamp_min(0).numpy()

        else:
            logits = self._forward(X)

            if self.method == "energy":
                score = -torch.logsumexp(logits, dim=1).numpy()
            elif self.method == "msp":
                score = 1 - logits.softmax(dim=1).max(dim=1).values.numpy()
            else:
                raise ValueError(f"Unknown Shallow OOD method: {self.method}")

        return -score


ood_pipelines = {
    "ACM+TS+OCSVM": make_pipeline(
        clone(pipelines["ACM+TS+SVM"][:-1]),
        StandardScaler(),
        OneClassSVM(kernel="rbf", gamma="scale", nu=0.1),
    ),
    "FBCSP+Mahalanobis": make_pipeline(
        clone(pipelines["FBCSP+shrinkLDA"][:-1]),
        ClassMahalanobis(),
    ),
    "Log-bandpower+IF": make_pipeline(
        FunctionTransformer(log_bandpower),
        IsolationForest(n_estimators=200, random_state=42, n_jobs=1),
    ),
    #"ShallowConvNet+MSP": ShallowOOD(method="msp", device=device),
    #"ShallowConvNet+Energy": ShallowOOD(method="energy", device=device),
    #"ShallowConvNet+kNN": ShallowOOD(method="knn", n_neighbors=10, device=device),
    "ShallowConvNet+MC Dropout": ShallowOOD(method="mc_dropout", n_mc=30, device=device),
}


def OOD_AUROC(test_data, model, known_classes=HANDS):
    X_test, y_test = test_data
    is_ood = ~np.isin(y_test, known_classes)

    if np.unique(is_ood).size != 2:
        raise ValueError("Test must contain both ID and OOD trials.")

    scores = -np.asarray(model.score_samples(X_test))

    if scores.shape != (len(y_test),) or not np.isfinite(scores).all():
        raise ValueError("Expected one finite OOD score per test trial.")

    return float(roc_auc_score(is_ood, scores))

def train(train_data, pipeline):
    return clone(pipeline).fit(*train_data)

def predict(test_data, model):
    return model.predict(test_data[0])

def MI_balanced_accuracy(train_data, test_data, pipeline):
    model = train(train_data, pipeline)
    y_pred = predict(test_data, model)
    return balanced_accuracy_score(test_data[1], y_pred)

def result_path(result):
    return Path("results") / f"{result['dataset']}__sub-{result['subject']:02d}__{result['experiment']}__{result['method']}.csv"

def save_result(result):
    path = result_path(result)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([result]).to_csv(path, index=False)

def summarize_results(dataset, experiment):
    files = list(Path("results").glob(f"{dataset}__sub-*__{experiment}__*.csv"))
    if not files:
        return pd.DataFrame()
    data = pd.concat([pd.read_csv(path) for path in files], ignore_index=True)
    return data.groupby("method").agg(BA=("MI_balanced_accuracy", "mean"), n=("subject", "nunique"))

def get_subjects(name, n=None, seed=42):
    if name == "Yang2025_3C":
        dataset = datasets.Yang2025(paradigm_type="3C")
    else:
        dataset = getattr(datasets, name)()
    subjects = dataset.subject_list
    if n is None or n >= len(subjects):
        return subjects
    return sorted(random.Random(seed).sample(subjects, n))

def load_subject_data(name, subject, val_size=0, channels=None ):
    kwargs = {"sessions": [0, 1]} 

    if name == "Zhou2020":
        kwargs["cohort"] = "S" if subject <= 12 else "A"

    elif name == "Yang2025_3C":
        name = "Yang2025"
        kwargs["paradigm_type"] = "3C"

    dataset = getattr(datasets, name)(**kwargs)

    paradigm = MotorImagery(
        events=["left_hand", "right_hand"],
        n_classes=2,
        fmin=8,
        fmax=32,
        resample=250,
        channels=channels,  
        baseline=None,
        tmin=0,
        tmax=min(4, dataset.interval[1] - dataset.interval[0]),
    )

    X, y, metadata = paradigm.get_data(dataset, subjects=[subject])

    # 4s * 250 hz = 1000
    X = X[..., :1000]
    if X.shape[-1] != 1000:
        raise ValueError("Expected 1000.")

    sessions = sorted(metadata["session"].unique())

    train_mask = metadata["session"].eq(sessions[0]).to_numpy()
    test_mask = metadata["session"].eq(sessions[1]).to_numpy()
    train_data = (X[train_mask], y[train_mask])
    test_data = (X[test_mask], y[test_mask])
    val_data = None
    if val_size:
        X_train, X_val, y_train, y_val = train_test_split(*train_data, test_size=val_size, stratify=train_data[1], random_state=42)
        train_data, val_data = (X_train, y_train), (X_val, y_val)
    return train_data, val_data, test_data

def load_subject_data2(name, subject, val_size=0, channels=None, include_feet=False):
    kwargs = {"sessions": [0, 1]}

    if name == "Zhou2020":
        kwargs["cohort"] = "S" if subject <= 12 else "A"
    elif name == "Yang2025_3C":
        name = "Yang2025"
        kwargs["paradigm_type"] = "3C"

    dataset = getattr(datasets, name)(**kwargs)
    events = list(HANDS) + (["feet"] if include_feet else [])

    paradigm = MotorImagery(
        events=events,
        n_classes=len(events),
        fmin=8,
        fmax=32,
        resample=250,
        channels=channels,
        baseline=None,
        tmin=0,
        tmax=min(4, dataset.interval[1] - dataset.interval[0]),
    )

    X, y, metadata = paradigm.get_data(dataset, subjects=[subject])
    X = X[..., :1000]

    if X.shape[-1] != 1000:
        raise ValueError("Expected 1000.")

    sessions = sorted(metadata["session"].unique())
    train_mask = metadata["session"].eq(sessions[0]).to_numpy() & np.isin(y, HANDS)
    test_mask = metadata["session"].eq(sessions[1]).to_numpy()

    train_data = (X[train_mask], y[train_mask])
    test_data = (X[test_mask], y[test_mask])
    val_data = None

    if val_size:
        X_train, X_val, y_train, y_val = train_test_split(*train_data, test_size=val_size, stratify=train_data[1], random_state=42)
        train_data, val_data = (X_train, y_train), (X_val, y_val)

    return train_data, val_data, test_data