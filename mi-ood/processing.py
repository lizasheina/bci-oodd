import random
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import patch

from IPython.display import display, HTML

import mne
import numpy as np
import pandas as pd
from moabb.paradigms import MotorImagery
import moabb.datasets as datasets
import moabb.datasets.wang2026 as wang_module
from scipy.io import loadmat
from sklearn.base import clone
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

import dataset_ttk

RESULTS = Path( "results" )
def experiment_dir(experiment, experiment_type, res_path=RESULTS):
    return Path(res_path) / experiment / experiment_type

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
    filename = ( f"{result['dataset']}__sub-{result['subject']:02d}__{result['method']}.csv" )
    return experiment_dir( result["experiment"], result["experiment_type"] ) / filename

def save_result(result):
    path = result_path(result)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([result]).to_csv(path, index=False)

def create_exp_file( experiment, experiment_types=None, res_path=RESULTS ):
    experiment_root = Path(res_path) / experiment
    if experiment_types is None:
        experiment_types = sorted( path.name for path in experiment_root.iterdir() if path.is_dir() )
    parts = []
    for experiment_type in experiment_types:
        files = sorted( ( res_path / experiment / experiment_type ).glob("*.csv") )
        parts.extend( pd.read_csv(p).assign( task=experiment_type, experiment=experiment, experiment_type=experiment_type, source_file=p.name ) for p in files )

    data = pd.concat(parts, ignore_index=True)
    output = res_path / f"{experiment}.csv"
    data.to_csv(output, index=False)
    print(f"Saved to {output}")
    return output


def summarize_results(dataset, experiment, metric="MI_BA", experiment_type="mi"):
    files = list( (Path("results") / experiment / experiment_type).glob(f"{dataset}__sub-*__*.csv") )
    if not files:
        return pd.DataFrame()
    data = pd.concat([pd.read_csv(path) for path in files], ignore_index=True)
    #return data.groupby("method").agg(metric=(metric, "mean"), n=("subject", "nunique"))
    return  data.groupby("method").agg(**{metric: (metric, "mean"), "n": ("subject", "nunique")})

def summarize_results_by_subject(dataset, experiment, metric="MI_BA", experiment_type="mi"):
    files = sorted( (Path("results") / experiment / experiment_type).glob(f"{dataset}__sub-*__*.csv") )
    if not files:
        return pd.DataFrame()

    data = pd.concat([pd.read_csv(p) for p in files], ignore_index=True)

    summary = data.groupby("method").agg(
        mean=(metric, "mean"),
        n=("subject", "nunique"),
    ).rename(columns={"mean": metric})

    by_subject = data.pivot( index="method", columns="subject", values=metric ).rename(columns=lambda s: f"sub-{int(s):02d}")

    return summary.join(by_subject).sort_values(metric, ascending=False)

#BEST_CELL_STYLE = "background-color: rgba(0, 128, 0, 0.50); font-weight: 600;" # background-color: green; color: #006100;
BEST_CELL_STYLE = (
    "background-color: rgba(0, 128, 0, 0.35); "
    "display: block; "
    "width: 100%; "
    "box-sizing: border-box; "
    "font-weight: 600;"
)
def _format_best_value(value, best):
    if pd.isna(value):
        return ""
    text = f"{value:.3f}"
    if value == best:
        return f'<span style="{BEST_CELL_STYLE}">{text}</span>'
    return text

def display_summary_by_subject(dataset, experiment, experiment_type="mi", metric="MI_BA"):
    table = summarize_results_by_subject(dataset, experiment, metric=metric, experiment_type=experiment_type)
    #subject_cols = [c for c in table.columns if c.startswith("sub-")]
    #display(table.round(3).style.highlight_max(axis=0, subset=subject_cols, color="green"))
    subject_cols = [c for c in table.columns if c.startswith("sub-")]
    html_table = table.round(3).copy()
    html_table[subject_cols] = html_table[subject_cols].astype(object)

    for col in subject_cols:
        best = html_table[col].max()
        html_table[col] = [ _format_best_value(value, best) for value in html_table[col] ]

    display(HTML(html_table.to_html(escape=False)))

def display_summary_by_dataset(experiment, mi_methods, ood_methods, experiment_specs=None, datasets=None, res_path=Path("results")):
    if experiment_specs is None:
        experiment_specs = [("MI BA", "mi", "MI_BA"), ("MI AUROC", "mi", "MI_AUROC"), ("OOD", "ood", "OOD_AUROC")]

    parts = []
    mi_methods, ood_methods = set(mi_methods), set(ood_methods)

    for task, experiment_type, metric in experiment_specs:
        files = sorted((res_path / experiment / experiment_type).glob("*.csv"))
        if not files:
            continue

        d = pd.concat([pd.read_csv(p) for p in files], ignore_index=True)
        methods = ood_methods if experiment_type.startswith("ood") else mi_methods
        d = d[d["method"].isin(methods)]
        if datasets is not None:
            d = d[d["dataset"].isin(datasets)]
        parts.append(d.assign(task=task, score=d[metric]))

    r = pd.concat(parts, ignore_index=True)
    means = r.groupby(["task", "method", "dataset"])["score"].mean().unstack("dataset")
    table = means.join(r.groupby(["task", "method"])["score"].agg(mean_subjects="mean", sd="std", n="count"))
    table["mean_datasets"] = means.mean(axis=1)

    dataset_cols = means.columns.tolist()
    table = table.sort_values(["task", "mean_subjects"], ascending=[True, False]).round(3)

    #def highlight_best(data):
    #    styles = pd.DataFrame("", index=data.index, columns=data.columns)

        #for task in data.index.get_level_values("task").unique():
        #    rows = data.index.get_level_values("task") == task
        #    best = data.loc[rows, dataset_cols].eq(data.loc[rows, dataset_cols].max())
        #    styles.loc[rows, dataset_cols] = best.replace({
        #        True: "background-color: green",
        #        False: "",
        #    })

        #return styles

    #display(table.style.apply(highlight_best, axis=None))
    
    html_table = table.copy()
    html_table[dataset_cols] = html_table[dataset_cols].astype(object)

    for task in html_table.index.get_level_values("task").unique():
        rows = html_table.index.get_level_values("task") == task

        for col in dataset_cols:
            best = table.loc[rows, col].max()
            html_table.loc[rows, col] = [ _format_best_value(value, best) for value in table.loc[rows, col] ]

    display(HTML(html_table.to_html(escape=False)))
    


WANG2026_45_SUBJECTS = sorted(set(range(1, 40)) - {1, 2, 4, 5, 6, 9, 10, 17, 24, 29})

def get_subjects(name, n=None, seed=42):
    if name == "Stieger2021_fix":
        name = "Stieger2021"
    wang_45 = name == "Wang2026_45"
    if name in {"Wang2026_fix", "Wang2026_45"}:
        name = "Wang2026"
    if name in {"Yang2025_3C", "Yang2025_3C_origwin", "Yang2025_3C_fixwin"}:
        dataset = datasets.Yang2025(paradigm_type="3C")
    else:
        dataset = getattr(datasets, name)()
    subjects = dataset.subject_list
    if wang_45:
        subjects = [s for s in subjects if s in WANG2026_45_SUBJECTS]
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
                      id_classes=HANDS, ood_classes=(), within_session=False, test_size=0.25,
                      session_index=0, sfreq=250, duration=4.0, random_state=42,
                      print_count=False,
                      fmin = 8, fmax=32 ):
    stieger_fix = name == "Stieger2021_fix"
    if stieger_fix: name = "Stieger2021"
    wang_45 = name == "Wang2026_45"
    wang_fix = name in {"Wang2026_fix", "Wang2026_45"}
    if wang_45 and subject not in WANG2026_45_SUBJECTS:
        raise ValueError(f"Wang2026_45: subject {subject} has no session 4/5.")

    if wang_fix:
        name = "Wang2026"
    if name == "Weibo2014":
        kwargs = {}
    elif name == "Stieger2021":
        kwargs = {"sessions": [2, 3], "interval": [0, duration]}
    elif name == "Wang2026":
        kwargs = {"sessions": [4, 5] if wang_45 else [3, 4]}
    else:
        kwargs = {"sessions": [0, 1]}

    yang_fixed_window = name == "Yang2025_3C_fixwin"

    if name == "Zhou2020":
        kwargs["cohort"] = "S" if subject <= 12 else "A"
    elif name in {"Yang2025_3C", "Yang2025_3C_origwin", "Yang2025_3C_fixwin"}:
        name = "Yang2025"
        kwargs["paradigm_type"] = "3C"

    dataset = getattr(datasets, name)(**kwargs)

    if wang_fix:
        raw_samples = int(round(duration * wang_module._SFREQ))
        dataset.interval = [0.0, (raw_samples - 1) / wang_module._SFREQ]
    if yang_fixed_window:
        dataset.interval = (0, 4)

    if isinstance(ood_classes, str):
        ood_classes = (ood_classes,)
    ood_classes = tuple(dict.fromkeys(ood_classes))

    if set(ood_classes) & set(id_classes):
        raise ValueError("ID and OOD should be different.")

    bnci_rest = name == "BNCI2014_001" and "rest" in ood_classes
    if bnci_rest and within_session:
        raise ValueError("BNCI rest: only for cross-session")
    zhou_rest = name == "Zhou2016" and "rest" in ood_classes
    if zhou_rest and (within_session or not 0 < duration <= 3):
        raise ValueError("Zhou2016 rest: cross-session, 0 < duration <= 3.")
    events = list(id_classes) + [c for c in ood_classes if not ((bnci_rest or zhou_rest) and c == "rest")]

    missing = set(events) - set(dataset.event_id)
    if missing:
        raise ValueError(
            f"{name}: no events {sorted(missing)}; "
            f"available {list(dataset.event_id)}."
        )
    tadd = 0.0
    if stieger_fix: tadd = 2.0
    paradigm = MotorImagery(
        events=events,
        n_classes=len(events),
        fmin=fmin,
        fmax=fmax,
        resample=sfreq,
        channels=channels,
        baseline=None,
        tmin=0+tadd,
        tmax=tadd+min(duration, dataset.interval[1] - dataset.interval[0]),
    )

    load_context = (
        patch.object(wang_module, "_NOMINAL_TRIAL_SAMPLES", raw_samples)
        if wang_fix else nullcontext()
    )

    with load_context:
        X, y, metadata = paradigm.get_data(
            dataset,
            subjects=[subject],
            cache_config={"use": False} if ( wang_fix or stieger_fix ) else None,
        )

    n_times = int(round(duration * sfreq))
    X = X[..., :n_times]

    if X.shape[-1] != n_times:
        raise ValueError(
            f"Expected {n_times} samples ({duration} s at {sfreq} Hz), "
            f"got {X.shape[-1]}."
        )

    sessions = sorted(metadata["session"].unique())
    is_id = np.isin(y, id_classes)
    is_ood = np.isin(y, ood_classes)

    if within_session:
        in_session = metadata["session"].eq(sessions[session_index]).to_numpy()
        id_idx = np.flatnonzero(in_session & is_id)

        train_idx, test_idx = train_test_split(id_idx, test_size=test_size, stratify=y[id_idx], random_state=random_state)

        ood_idx = np.flatnonzero(in_session & is_ood)
        test_idx = np.r_[test_idx, ood_idx]

    else:
        if len(sessions) < 2:
            raise ValueError(
                f"{name}: для cross-session нужны две сессии."
            )

        train_idx = np.flatnonzero(metadata["session"].eq(sessions[0]).to_numpy() & is_id)
        test_idx = np.flatnonzero(metadata["session"].eq(sessions[1]).to_numpy() & (is_id | is_ood))

    missing_test = set(events) - set(y[test_idx])
    if missing_test:
        raise ValueError(
            f"В test отсутствуют классы: {sorted(missing_test)}."
        )

    train_data = (X[train_idx], y[train_idx])
    test_data = (X[test_idx], y[test_idx])

    if bnci_rest:
        block = loadmat(dataset.data_path(subject)[1], simplify_cells=True)["data"][0]
        names = "Fz FC3 FC1 FCz FC2 FC4 C5 C3 C1 Cz C2 C4 C6 CP3 CP1 CPz CP2 CP4 P1 Pz P2 POz".split()
        raw = mne.io.RawArray(block["X"][:, :22].T * 1e-6, mne.create_info(names, block["fs"], "eeg"), verbose=False)
        raw.pick(names if channels is None else channels).filter(8, 32, method="iir", verbose=False)
        ev = mne.make_fixed_length_events(raw, start=2, stop=raw.times[-1] - 2, duration=duration)
        rest = mne.Epochs(raw, ev, tmin=0, tmax=duration, baseline=None, preload=True, verbose=False)
        rest = rest.resample(sfreq, verbose=False).get_data()[..., :n_times] * 1e6
        test_data = (np.concatenate([test_data[0], rest]), np.concatenate([test_data[1], np.full(len(rest), "rest")]))
    if zhou_rest:
        rest_paradigm = MotorImagery(events=list(dataset.event_id), n_classes=3, fmin=8, fmax=32, resample=sfreq, channels=channels, baseline=None, tmin=5.5, tmax=5.5 + duration)
        rest, _, rest_meta = rest_paradigm.get_data(dataset, subjects=[subject], cache_config={"use": False})
        rest = rest[rest_meta["session"].to_numpy() == sessions[1], :, :n_times]
        test_data = (np.concatenate([test_data[0], rest]), np.concatenate([test_data[1], np.full(len(rest), "rest")]))
    val_data = None

    if val_size:
        X_train, X_val, y_train, y_val = train_test_split(*train_data, test_size=val_size, stratify=train_data[1], random_state=random_state)
        train_data, val_data = (X_train, y_train), (X_val, y_val)

    train_data = select_trials(train_data, n_trials, random_state)
    if print_count:
        n_val = 0 if val_data is None else len(val_data[1])
        print(f"{name} {subject}: {len(train_data[1])}/{n_val}/{len(test_data[1])}")
    return train_data, val_data, test_data
