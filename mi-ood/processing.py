import random
from pathlib import Path

import numpy as np
import pandas as pd
from moabb.paradigms import MotorImagery
import moabb.datasets as datasets
from sklearn.base import clone
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

import dataset_ttk

RESULTS = Path( "results" )

HANDS = ("left_hand", "right_hand")

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

def MI_metrics( train_data, test_data, pipeline ):
    model = train(train_data, pipeline)
    y_pred = predict(test_data, model)
    if hasattr(model, "decision_function"):
        scores = model.decision_function(test_data[0])
    else:
        scores = model.predict_proba(test_data[0])[:, 1]
    return balanced_accuracy_score(test_data[1], y_pred), roc_auc_score( test_data[1] == model.classes_[1], scores)

def MI_(test_data, model):
    X_test, y_test = test_data
    if hasattr(model, "decision_function"):
        scores = model.decision_function(X_test)
    else:
        scores = model.predict_proba(X_test)[:, 1]
    return roc_auc_score(y_test == model.classes_[1], scores)

def result_path(result):
    filename = f"{result['dataset']}__sub-{result['subject']:02d}__{result['experiment']}__{result['method']}.csv"
    return Path("results") / result["experiment"] / filename

def save_result(result):
    path = result_path(result)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([result]).to_csv(path, index=False)

def create_exp_file( mi_exp, ood_exp, res_path ):
    parts = []
    for task, experiment in [("MI", mi_exp), ("OOD", ood_exp)]:
        files = sorted((res_path / experiment).glob("*.csv"))
        parts.extend(pd.read_csv(p).assign(task=task, source_file=p.name) for p in files)

    data = pd.concat(parts, ignore_index=True)
    output = res_path / f"{mi_exp}__{ood_exp}__all.csv"
    data.to_csv(output, index=False)
    print(f"Saved to {output}")
    return output


def summarize_results(dataset, experiment, metric="MI_BA"):
    files = list( (Path("results") / experiment).glob(f"{dataset}__sub-*__{experiment}__*.csv") )
    if not files:
        return pd.DataFrame()
    data = pd.concat([pd.read_csv(path) for path in files], ignore_index=True)
    return data.groupby("method").agg(metric=(metric, "mean"), n=("subject", "nunique"))

def get_subjects(name, n=None, seed=42):
    if name in {"Yang2025_3C", "Yang2025_3C_origwin", "Yang2025_3C_fixwin"}:
        dataset = datasets.Yang2025(paradigm_type="3C")
    else:
        dataset = getattr(datasets, name)()
    subjects = dataset.subject_list
    if n is None or n >= len(subjects):
        return subjects
    return sorted(random.Random(seed).sample(subjects, n))

def select_trials(data, n_trials=None, random_state=42):
    if n_trials is None:
        return data

    X, y = data
    classes, counts = np.unique(y, return_counts=True)

    if ( not isinstance(n_trials, (int, np.integer))
         or n_trials <= 0
         or n_trials % len(classes) ):
        raise ValueError("n_trials должно быть положительным и кратным числу классов.")

    per_class = n_trials // len(classes)
    if counts.min() < per_class:
        raise ValueError(
            f"Not enough trials: {dict(zip(classes, counts))}; need {per_class} per class."
        )

    rng = np.random.default_rng(random_state)
    indices = np.sort(np.concatenate([
        rng.choice(np.flatnonzero(y == label), per_class, replace=False)
        for label in classes
    ]))
    return X[indices], y[indices]

# n_trials 90
def load_subject_data(name, subject, val_size=0, n_trials=None, channels=None, 
                      include_feet=False, within_session=False, test_size=0.25, session_index=0, random_state=42 ):
    kwargs = {} if name == "Weibo2014" else {"sessions": [0, 1]}
    yang_fixed_window = name == "Yang2025_3C_fixwin"

    if name == "Zhou2020":
        kwargs["cohort"] = "S" if subject <= 12 else "A"
    elif name in {"Yang2025_3C", "Yang2025_3C_origwin", "Yang2025_3C_fixwin"}:
        name = "Yang2025"
        kwargs["paradigm_type"] = "3C"

    dataset = getattr(datasets, name)(**kwargs)
    if yang_fixed_window:
        dataset.interval = (0, 4)
    feet_events = ["left_foot", "right_foot"] if name == "TTK" else ["feet"]
    events = list(HANDS) + (feet_events if include_feet else [])

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

    # 4s * 250 hz = 1000
    X = X[..., :1000]

    if X.shape[-1] != 1000:
        raise ValueError("Expected 1000.")

    sessions = sorted(metadata["session"].unique())
    is_id = np.isin(y, HANDS)
    if within_session:
        in_session = metadata["session"].eq( sessions[session_index] ).to_numpy()
        id_idx = np.flatnonzero(in_session & is_id)
        train_idx, test_idx = train_test_split( id_idx, test_size=test_size, stratify=y[id_idx], random_state=42 )
        ood_idx = np.flatnonzero(in_session & ~is_id)
        test_idx = np.r_[test_idx, ood_idx]
    else:
        train_idx = np.flatnonzero( metadata["session"].eq(sessions[0]).to_numpy() & is_id )
        test_idx = np.flatnonzero( metadata["session"].eq(sessions[1]).to_numpy() )

    train_data = (X[train_idx], y[train_idx])
    test_data = (X[test_idx], y[test_idx])
    val_data = None

    if val_size:
        X_train, X_val, y_train, y_val = train_test_split(*train_data, test_size=val_size, stratify=train_data[1], random_state=42)
        train_data, val_data = (X_train, y_train), (X_val, y_val)

    train_data = select_trials(train_data, n_trials, random_state)
    return train_data, val_data, test_data
