import mne
import numpy as np
import torch

from braindecode import EEGClassifier
from braindecode.models import EEGNet as EEGNetModel
from braindecode.models import ShallowFBCSPNet
from braindecode.util import set_random_seeds

from pycrostates.cluster import ModKMeans
from pycrostates.preprocessing import extract_gfp_peaks

from scipy.signal import butter, sosfiltfilt
from scipy.special import logsumexp

from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.covariance import LedoitWolf
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

from statsmodels.regression.linear_model import burg
from statsmodels.tsa.api import VAR

from torch import nn



mne.set_log_level("WARNING")

HANDS = ("left_hand", "right_hand")


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
    
    def predict_proba(self, X):
        return self.net_.predict_proba(self._prepare(X))

class EEGNetClassifier(ShallowConvNet):
    def fit(self, X, y):
        set_random_seeds(seed=42, cuda=self.device.startswith("cuda"))
        self.classes_, y = np.unique(y, return_inverse=True)
        self.mean_ = X.mean(axis=(0, 2), keepdims=True)
        self.std_ = np.maximum(X.std(axis=(0, 2), keepdims=True), 1e-6)
        self.net_ = EEGClassifier(
            EEGNetModel, module__n_chans=X.shape[1], module__n_times=X.shape[2], module__n_outputs=len(self.classes_),
            module__F1=8, module__D=2, module__F2=16, module__kernel_length=64, module__drop_prob=.5,
            optimizer=torch.optim.Adam, lr=.001, batch_size=64, max_epochs=self.epochs, train_split=None,
            iterator_train__drop_last=False, device=self.device, verbose=0,
        )
        self.net_.fit(self._prepare(X), y.astype(np.int64))
        return self

class MicrostateFeatures(TransformerMixin, BaseEstimator):
    def __init__(self, k=4, sfreq=250): self.k, self.sfreq = k, sfreq
    def _epochs(self, X):
        return mne.EpochsArray(X * 1e-6, self.info_, verbose=False).set_eeg_reference("average", verbose=False)
    def fit(self, X, y=None):
        self.info_ = mne.create_info(X.shape[1], self.sfreq, "eeg")
        self.model_ = ModKMeans(n_clusters=self.k, n_init=20, random_state=42)
        peaks = extract_gfp_peaks(self._epochs(X), min_peak_distance=max(1, round(.02 * self.sfreq)), verbose=False)
        self.model_.fit(peaks, n_jobs=1, verbose=False)
        return self
    def transform(self, X):
        epochs, rows = self._epochs(X), []
        for x in epochs.get_data(copy=False):
            seg = self.model_.predict(mne.io.RawArray(x, epochs.info, verbose=False), reject_edges=True, verbose=False)
            p = seg.compute_parameters()
            values = [p[f"{s}_{m}"] for s in self.model_.cluster_names for m in ("meandurs", "occurrences", "timecov", "mean_corr", "gev")]
            rows.append(np.r_[values, seg.compute_transition_matrix()[~np.eye(self.k, dtype=bool)]])
        return np.asarray(rows)

def ar_features(X, order=10):
    return np.array([np.concatenate([burg(x, order=order)[0] for x in trial]) for trial in X])

def ar_features_var(X, order=10):
    def features(x):
        a, v = burg(x, order=order)
        return np.r_[a, np.log(max(v, 1e-12))]
    return np.array([np.concatenate([features(x) for x in trial]) for trial in X])

def var_features(X, order=2):
    features = []
    for trial in X:
        data = (trial - trial.mean(axis=1, keepdims=True)).T
        model = VAR(data).fit(maxlags=order, trend="n")
        features.append(np.r_[model.coefs.ravel(), np.log(np.maximum(np.diag(model.sigma_u), 1e-12))])
    return np.asarray(features)

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

class TempDensEEGNet( nn.Module ):
    def __init__( self, channels ):
        super().__init__()
        self.block1 = nn.Sequential( 
            nn.ZeroPad2d( ( 124, 125, 0, 0 ) ), 
            nn.Conv2d( 1, 8, ( 1, 250 ), bias=False ), 
            nn.BatchNorm2d( 8 ), 
            nn.Conv2d( 8, 16, ( channels, 1 ), groups=8, bias=False ), 
            nn.BatchNorm2d( 16 ), 
            nn.ELU(), 
            nn.AvgPool2d( ( 1, 4 ) ), 
            nn.Dropout( .5 ) 
        )
        self.block2 = nn.Sequential( 
            nn.ZeroPad2d( ( 7, 8, 0, 0 ) ), 
            nn.Conv2d( 16, 16, ( 1, 16 ), groups=16, bias=False ), 
            nn.Conv2d( 16, 16, ( 1, 1 ), bias=False ), 
            nn.BatchNorm2d( 16 ), 
            nn.ELU(), 
            nn.AvgPool2d( ( 1, 8 ) ), 
            nn.Dropout( .5 ) 
        )
        self.classifier = nn.Linear( 240, 2 )

    def forward( self, X ):
        features = self.block2( self.block1( X.unsqueeze( 1 ) ) ).flatten( 1 )
        return self.classifier( features ), features


def tempdens_windows( X ):
    X = np.asarray( X, dtype=np.float32 )
    if X.ndim != 3 or X.shape[-1] < 500 or not np.isfinite( X ).all():
        raise ValueError( "Expected finite X with shape (trials, channels, time >= 500)" )
    return np.stack( [ X[..., start:start + 500] for start in range( 0, X.shape[-1] - 499, 31 ) ], axis=1 )


def tempdens_temporal( features ):
    scores = np.zeros( features.shape[:2], dtype=float )
    if features.shape[1] > 1:
        scores[:, 1] = np.linalg.norm( features[:, 1] - features[:, 0], axis=-1 )
    if features.shape[1] > 2:
        scores[:, 2:] = np.linalg.norm( features[:, 2:] - 2 * features[:, 1:-1] + features[:, :-2], axis=-1 )
    return scores


class EEGNetTempDens( BaseEstimator ):
    def __init__( self, epochs=50, batch_size=64, seed=42, device="cpu", threads=2, learning_rate=.001, weight_decay=.0001, cov_reg=.001, knn_k=10 ):
        self.epochs, self.batch_size, self.seed = epochs, batch_size, seed
        self.device, self.threads = device, threads
        self.learning_rate, self.weight_decay = learning_rate, weight_decay
        self.cov_reg, self.knn_k = cov_reg, knn_k

    def _encode( self, windows ):
        shape = windows.shape[:2]; flat = windows.reshape( -1, *windows.shape[2:] )
        logits, features = [], []
        self.network_.eval()
        with torch.inference_mode():
            for start in range( 0, len( flat ), self.batch_size ):
                X = ( flat[start:start + self.batch_size] - self.mean_ ) / self.std_
                z, f = self.network_( torch.from_numpy( X ).to( self.device ) )
                logits.append( z.cpu().numpy() ); features.append( f.cpu().numpy() )
        return np.concatenate( logits ).reshape( *shape, -1 ), np.concatenate( features ).reshape( *shape, -1 )

    def _components( self, logits, features, exclude_self=False ):
        shape = features.shape[:2]; flat = features.reshape( -1, features.shape[-1] ).astype( float )
        diff = flat[:, None] - self.class_means_[None]
        mahal = np.maximum( np.einsum( "nci,ij,ncj->nc", diff, self.precision_, diff, optimize=True ).min( axis=1 ), 0 )
        k = min( self.knn_k + int( exclude_self ), len( self.memory_ ) )
        distances = self.neighbors_.kneighbors( flat, n_neighbors=k, return_distance=True )[0]
        if exclude_self:
            distances = distances[:, 1:]
        density = .5 * mahal + .5 * distances.mean( axis=1 )
        return np.stack( [ -logsumexp( logits.astype( float ), axis=-1 ), density.reshape( shape ), tempdens_temporal( features ) ], axis=-1 )

    def fit( self, X, y ):
        state = torch.get_num_threads(), torch.are_deterministic_algorithms_enabled(), torch.is_deterministic_algorithms_warn_only_enabled()
        try:
            return self._fit( X, y )
        finally:
            torch.set_num_threads( state[0] )
            torch.use_deterministic_algorithms( state[1], warn_only=state[2] )

    def _fit( self, X, y ):
        y = np.asarray( y )
        if len( X ) != len( y ) or set( np.unique( y ) ) != set( HANDS ):
            raise ValueError( "Fit requires only left_hand and right_hand, with one label per trial" )
        torch.set_num_threads( self.threads ); torch.manual_seed( self.seed ); torch.use_deterministic_algorithms( True )
        windows = tempdens_windows( X ); self.classes_ = np.asarray( HANDS )
        self.network_ = TempDensEEGNet( windows.shape[2] ).to( self.device )
        flat = windows.reshape( -1, *windows.shape[2:] )
        self.mean_ = flat.mean( axis=( 0, 2 ), keepdims=True ).astype( np.float32 )
        self.std_ = flat.std( axis=( 0, 2 ), keepdims=True ).clip( 1e-6 ).astype( np.float32 )
        inputs = torch.from_numpy( ( flat - self.mean_ ) / self.std_ )
        labels = np.repeat( np.searchsorted( self.classes_, y ), windows.shape[1] )
        targets = torch.from_numpy( labels ).long()
        loader = torch.utils.data.DataLoader( torch.utils.data.TensorDataset( inputs, targets ), batch_size=self.batch_size, shuffle=True, generator=torch.Generator().manual_seed( self.seed ) )
        optimizer = torch.optim.Adam( self.network_.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay )

        for epoch in range( self.epochs ):
            self.network_.train()
            for bx, by in loader:
                bx, by = bx.to( self.device ), by.to( self.device )
                optimizer.zero_grad( set_to_none=True )
                logits, _ = self.network_( bx )
                nn.functional.cross_entropy( logits, by ).backward()
                optimizer.step()

        logits, features = self._encode( windows )
        self.memory_ = features.reshape( -1, features.shape[-1] ).astype( float )
        self.class_means_ = np.stack( [ self.memory_[labels == c].mean( axis=0 ) for c in range( 2 ) ] )
        covariance = np.cov( self.memory_, rowvar=False ) + self.cov_reg * np.eye( self.memory_.shape[1] )
        self.precision_ = np.linalg.pinv( covariance, hermitian=True )
        self.neighbors_ = NearestNeighbors( metric="euclidean", n_jobs=1 ).fit( self.memory_ )
        parts = self._components( logits, features, exclude_self=True ).reshape( -1, 3 )
        self.component_mean_ = parts.mean( axis=0 )
        std = parts.std( axis=0 ); self.component_std_ = np.where( std > 1e-6, std, 1. )
        return self

    def details( self, X ):
        logits, features = self._encode( tempdens_windows( X ) )
        parts = self._components( logits, features )
        scaled = ( parts - self.component_mean_ ) / self.component_std_
        return { "ood_score": scaled.sum( axis=-1 ).mean( axis=1 ), "class_prediction": self.classes_[logits.mean( axis=1 ).argmax( axis=1 )], "components": parts, "scaled_components": scaled, "logits": logits, "features": features }

    def score_samples( self, X ):
        return -self.details( X )["ood_score"]

    def predict( self, X ):
        return self.details( X )["class_prediction"]

class KNNDistance(BaseEstimator):
    def __init__(self, k=10): self.k = k
    def fit(self, X, y=None):
        self.nn_ = NearestNeighbors(n_neighbors=min(self.k, len(X)), n_jobs=1).fit(X)
        return self
    def score_samples(self, X): return -self.nn_.kneighbors(X)[0][:, -1]

