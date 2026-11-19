"""Mulder EEGNeX MC dropout, using the local author implementation in literature."""

from copy import deepcopy
import importlib.util
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split

model_path = Path(__file__).resolve().parents[1] / "literature/ood_bci/sources/mulder2026/repo/model/EEGNeX.py"
spec = importlib.util.spec_from_file_location("mulder_eegnex", model_path)
mulder = importlib.util.module_from_spec(spec); spec.loader.exec_module(mulder)


class MulderMC(ClassifierMixin, BaseEstimator):
    uses_validation = True
    def __init__(self, epochs=200, batch_size=32, n_mc=50, device="cpu", seed=42): 
        self.epochs, self.batch_size, self.n_mc, self.device, self.seed = epochs, batch_size, n_mc, device, seed
    
    def _tensor(self, X): 
        return torch.from_numpy(((X - self.mean_) / self.std_).astype(np.float32))[:, None]
    
    def fit(self, X, y, val_data=None):
        if val_data is None:
            X, X_val, y, y_val = train_test_split(X, y, test_size=0.1, stratify=y, random_state=self.seed)
            val_data = (X_val, y_val)
        self.n_fit_, self.n_validation_ = len(y), len(val_data[1])
        torch.manual_seed(self.seed)
        self.classes_, labels = np.unique(y, return_inverse=True)
        self.mean_ = X.mean(axis=(0, 2), keepdims=True)
        self.std_ = X.std(axis=(0, 2), keepdims=True).clip(1e-6)
        self.net_ = mulder.EEGNeX_8_32(len(self.classes_), X.shape[2], X.shape[1]).to(self.device)
        self.net_.conv5.apply_spectral()
        loader = DataLoader(TensorDataset(self._tensor(X), torch.as_tensor(labels, dtype=torch.long)), batch_size=self.batch_size, shuffle=True, generator=torch.Generator().manual_seed(self.seed))
        xv = self._tensor(val_data[0]).to(self.device)
        yv = torch.as_tensor(np.searchsorted(self.classes_, val_data[1]), dtype=torch.long, device=self.device)
        optimizer = torch.optim.Adam(self.net_.parameters(), lr=.001)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=.5, patience=5)
        best, wait, state = float("inf"), 0, None
        for _ in range(self.epochs):
            self.net_.train()
            for xb, yb in loader:
                optimizer.zero_grad(); loss = nn.functional.cross_entropy(self.net_(xb.to(self.device)), yb.to(self.device)); loss.backward(); optimizer.step()
            self.net_.eval()
            with torch.no_grad(): 
                value = nn.functional.cross_entropy(self.net_(xv), yv).item()
            scheduler.step(value)
            if value < best: 
                best, wait, state = value, 0, deepcopy(self.net_.state_dict())
            else: 
                wait += 1
            if wait >= 20: break
        self.net_.load_state_dict(state); self.net_.conv5.remove_spectral(); self.net_.eval(); return self
    
    def predict_proba(self, X):
        self.net_.eval()
        with torch.no_grad(): 
            return torch.cat([self.net_(b.to(self.device)).softmax(1).cpu() for b in self._tensor(X).split(self.batch_size)]).numpy()
    
    def predict(self, X): 
        return self.classes_[self.predict_proba(X).argmax(1)]
    
    def score_samples(self, X):
        buffers = {n: b.clone() for n, b in self.net_.named_buffers()}
        self.net_.eval()
        for m in self.net_.modules():
            if isinstance(m, nn.Dropout): 
                m.train()
        try:
            scores = []
            with torch.no_grad():
                for b in self._tensor(X).split(self.batch_size):
                    b = b.to(self.device)
                    p = torch.stack([self.net_(b).softmax(1) for _ in range(self.n_mc)]).mean(0)
                    scores.append((p * p.clamp_min(1e-10).log()).sum(1).cpu())
            return torch.cat(scores).numpy()
        finally:
            for n, b in self.net_.named_buffers(): 
                b.copy_(buffers[n])
            self.net_.eval()
