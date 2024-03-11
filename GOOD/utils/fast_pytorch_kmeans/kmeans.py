import warnings

import math
import torch
from time import time
import numpy as np
import pynvml
from .init_methods import init_methods


class KMeans:
    '''
    Kmeans clustering algorithm implemented with PyTorch

    Parameters:
      n_clusters: int,
        Number of clusters

      max_iter: int, default: 100
        Maximum number of iterations

      tol: float, default: 0.0001
        Tolerance

      verbose: int, default: 0
        Verbosity

      mode: {'euclidean', 'cosine'}, default: 'euclidean'
        Type of distance measure

      init_method: {'random', 'point', '++'}
        Type of initialization

      minibatch: {None, int}, default: None
        Batch size of MinibatchKmeans algorithm
        if None perform full KMeans algorithm

    Attributes:
      centroids: torch.Tensor, shape: [n_clusters, n_features]
        cluster centroids
    '''

    def __init__(self, n_clusters, max_iter=300, tol=0.0001, verbose=0, mode="euclidean", init_method="kmeans++",
                 minibatch=None, n_init=None, algorithm=None, device=None):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        self.mode = mode
        self.init_method = init_method
        self.minibatch = minibatch
        self._loop = False
        self._show = False

        self.n_init = n_init

        if algorithm is not None:
            warnings.warn("The parameter algorithm is not valid in this implementation of KMeans. Default: 'lloyd'")

        try:
            import pynvml
            self._pynvml_exist = True
        except ModuleNotFoundError:
            self._pynvml_exist = False

        self.device = device
        self.cluster_centers_ = None
        self.labels_ = None

    @staticmethod
    def cos_sim(a, b):
        """
          Compute cosine similarity of 2 sets of vectors

          Parameters:
          a: torch.Tensor, shape: [m, n_features]

          b: torch.Tensor, shape: [n, n_features]
        """
        a_norm = a.norm(dim=-1, keepdim=True)
        b_norm = b.norm(dim=-1, keepdim=True)
        a = a / (a_norm + 1e-8)
        b = b / (b_norm + 1e-8)
        return a @ b.transpose(-2, -1)

    @staticmethod
    def euc_sim(a, b):
        """
          Compute euclidean similarity of 2 sets of vectors

          Parameters:
          a: torch.Tensor, shape: [m, n_features]

          b: torch.Tensor, shape: [n, n_features]
        """
        return 2 * a @ b.transpose(-2, -1) - (a ** 2).sum(dim=1)[..., :, None] - (b ** 2).sum(dim=1)[..., None, :]

    def remaining_memory(self):
        """
          Get remaining memory in gpu
        """
        with torch.cuda.device(self.device):
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        if self._pynvml_exist:
            pynvml.nvmlInit()
            gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(self.device.index)
            info = pynvml.nvmlDeviceGetMemoryInfo(gpu_handle)
            remaining = info.free
        else:
            remaining = torch.cuda.memory_allocated()
        return remaining

    def max_sim(self, a, b):
        """
          Compute maximum similarity (or minimum distance) of each vector
          in a with all of the vectors in b

          Parameters:
          a: torch.Tensor, shape: [m, n_features]

          b: torch.Tensor, shape: [n, n_features]
        """
        batch_size = a.shape[0]
        if self.mode == 'cosine':
            sim_func = self.cos_sim
        elif self.mode == 'euclidean':
            sim_func = self.euc_sim

        if self.device == 'cpu':
            sim = sim_func(a, b)
            max_sim_v, max_sim_i = sim.max(dim=-1)
            return max_sim_v, max_sim_i
        else:
            if a.dtype == torch.double:
                expected = a.shape[0] * a.shape[1] * b.shape[0] * 8
            if a.dtype == torch.float:
                expected = a.shape[0] * a.shape[1] * b.shape[0] * 4
            elif a.dtype == torch.half:
                expected = a.shape[0] * a.shape[1] * b.shape[0] * 2
            ratio = math.ceil(expected / self.remaining_memory())
            subbatch_size = math.ceil(batch_size / ratio)
            msv, msi = [], []
            for i in range(ratio):
                if i * subbatch_size >= batch_size:
                    continue
                sub_x = a[i * subbatch_size: (i + 1) * subbatch_size]
                sub_sim = sim_func(sub_x, b)
                sub_max_sim_v, sub_max_sim_i = sub_sim.max(dim=-1)
                del sub_sim
                msv.append(sub_max_sim_v)
                msi.append(sub_max_sim_i)
            if ratio == 1:
                max_sim_v, max_sim_i = msv[0], msi[0]
            else:
                max_sim_v = torch.cat(msv, dim=0)
                max_sim_i = torch.cat(msi, dim=0)
            return max_sim_v, max_sim_i

    def fit_predict(self, X, sample_weight=None, centroids=None):
        """
          Combination of fit() and predict() methods.
          This is faster than calling fit() and predict() seperately.

          Parameters:
          X: torch.Tensor, shape: [n_samples, n_features]

          centroids: {torch.Tensor, None}, default: None
            if given, centroids will be initialized with given tensor
            if None, centroids will be randomly chosen from X

          Return:
          labels: torch.Tensor, shape: [n_samples]
        """
        assert isinstance(X, torch.Tensor), "input must be torch.Tensor"
        assert X.dtype in [torch.half, torch.float, torch.double], "input must be floating point"
        assert X.ndim == 2, "input must be a 2d tensor with shape: [n_samples, n_features] "

        batch_size, emb_dim = X.shape
        X = X.to(self.device)
        if sample_weight is None:
            sample_weight = torch.ones(batch_size, device=self.device, dtype=X.dtype)
        else:
            sample_weight = sample_weight.to(self.device)
        start_time = time()
        if centroids is None:
            cluster_centers_ = init_methods[self.init_method](X, self.n_clusters, self.minibatch)
        else:
            cluster_centers_ = centroids
        num_points_in_clusters = torch.ones(self.n_clusters, device=self.device, dtype=X.dtype)
        closest = None
        for i in range(self.max_iter):
            iter_time = time()
            if self.minibatch is not None:
                minibatch_idx = np.random.choice(batch_size, size=[self.minibatch], replace=False)
                x = X[minibatch_idx]
                sample_weight = sample_weight[minibatch_idx]
            else:
                x = X

            sim_score, closest = self.max_sim(a=x, b=cluster_centers_)
            matched_clusters, counts = closest.unique(return_counts=True)
            unmatched_clusters = torch.where(torch.ones(len(cluster_centers_), dtype=torch.bool, device=self.device).index_fill_(0, matched_clusters.long(), False) == True)[0]
            # reallocate unmatched clusters according to the machanism described
            # in https://github.com/scikit-learn/scikit-learn/blob/4af30870b0a09bf0a04d704bea4c5d861eae7c83/sklearn/cluster/_k_means_lloyd.pyx#L156
            while unmatched_clusters.shape[0] > 0:
                worst_x = x[sim_score.argmin(dim=0)]
                cluster_centers_[unmatched_clusters[0]] = worst_x
                sim_score, closest = self.max_sim(a=x, b=cluster_centers_)
                matched_clusters, counts = closest.unique(return_counts=True)
                unmatched_clusters = torch.where(
                    torch.ones(len(cluster_centers_), dtype=torch.bool, device=self.device).index_fill_(0, matched_clusters.long(),
                                                                                    False) == True)[0]

            c_grad = torch.zeros_like(cluster_centers_)
            expanded_closest = closest[None].expand(self.n_clusters, -1)
            mask = (expanded_closest == torch.arange(self.n_clusters, device=self.device)[:, None]).to(X.dtype)  # [n_clusters, minibatch] one-hot sample masks for each cluster
            mask = mask * sample_weight[None, :]
            c_grad = mask @ x / mask.sum(-1)[..., :, None]
            c_grad[c_grad != c_grad] = 0  # remove NaNs

            error = (c_grad - cluster_centers_).pow(2).sum()
            if self.minibatch is not None:
                lr = 1 / num_points_in_clusters[:, None] * 0.9 + 0.1
                # lr = 1/num_points_in_clusters[:,None]**0.1
            else:
                lr = 1
            num_points_in_clusters[matched_clusters] += counts
            cluster_centers_ = cluster_centers_ * (1 - lr) + c_grad * lr
            if self.verbose >= 2:
                print('iter:', i, 'error:', error.item(), 'time spent:', round(time() - iter_time, 4))
            if error <= self.tol:
                break

        if self.verbose >= 1:
            print(
                f'used {i + 1} iterations ({round(time() - start_time, 4)}s) to cluster {batch_size} items into {self.n_clusters} clusters')

        inertia = (sim_score * sample_weight).sum().neg()
        return cluster_centers_, closest, inertia

    def predict(self, X):
        """
          Predict the closest cluster each sample in X belongs to

          Parameters:
          X: torch.Tensor, shape: [n_samples, n_features]

          Return:
          labels: torch.Tensor, shape: [n_samples]
        """
        assert isinstance(X, torch.Tensor), "input must be torch.Tensor"
        assert X.dtype in [torch.half, torch.float, torch.double], "input must be floating point"
        assert X.ndim == 2, "input must be a 2d tensor with shape: [n_samples, n_features] "

        return self.max_sim(a=X, b=self.cluster_centers_)[1]

    def fit(self, X, sample_weight=None, centroids=None):
        """
          Perform kmeans clustering

          Parameters:
          X: torch.Tensor, shape: [n_samples, n_features]
        """
        assert isinstance(X, torch.Tensor), "input must be torch.Tensor"
        assert X.dtype in [torch.half, torch.float, torch.double], "input must be floating point"
        assert X.ndim == 2, "input must be a 2d tensor with shape: [n_samples, n_features] "

        self.cluster_centers_, self.labels_, self.inertia_ = [], [], []
        for i in range(self.n_init):
            cluster_centers, labels, inertia = self.fit_predict(X, sample_weight, centroids)
            self.cluster_centers_.append(cluster_centers.detach().cpu().numpy())
            self.labels_.append(labels.detach().cpu().numpy())
            self.inertia_.append(inertia.detach().cpu().numpy())
        best_cluster_idx = np.argmin(self.inertia_)
        self.cluster_centers_, self.labels_, self.inertia_ = self.cluster_centers_[best_cluster_idx], self.labels_[best_cluster_idx], self.inertia_[best_cluster_idx]
        return self